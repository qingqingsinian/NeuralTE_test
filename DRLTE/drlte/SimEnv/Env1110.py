import random
import numpy as np

from tensorflow.compat.v1.logging import log, log_first_n, set_verbosity
from tensorflow.compat.v1.logging import INFO, ERROR, DEBUG, WARN


# This Env is used for offline training.
class Env:
    def __init__(self, path_pre, file_name, topo_name, epoch, seed, start_index, tm_circle, len_circle, start_step):
        random.seed(seed)
        np.random.seed(seed)
        from os import path as osp
        self.__filepath = osp.join(path_pre, file_name + ".txt")  # e.g. GEA_trueTM_train4000.txt
        self.__topopath = osp.join(path_pre, topo_name + ".txt")
        log(INFO, f"filepath and topopath is {self.__filepath} and {self.__topopath}")

        self.__toponame = topo_name
        self.__epoch = epoch
        self.__episode = -1
        self.__nodenum = 0
        self.__edgenum = 0  # 备选路径中使用的边的数量(并不是topo图中的边),topo中的边都是双向的，但是这里可能只计入了一次甚至0次 --> 实际上该变量用于确定action的维度(没使用过的边完全可以计入action范围)
        self.__edgemap = []  # 联通矩阵
        self.__capaMatrix = []  # 带宽矩阵大小(对于GEA是B/5ms)
        self.__sessnum = 0  # ODpair数量
        self.__sesspaths = []  # 备选路径 (ODPairNum , SessionPathNum, EachHopOnPath)
        self.__pathnum = []  # 一维数组，每一个ODpair的备选路径数量
        self.__sessrate = []  # 当前的TM
        self.__sessrates = []  # 每个TM的ODPair之间的发送速率(乘上时间就是发送量) -- ( TMnum , ODPairNum*ODPairNum ) [需要进行单位转换]
        self.__flowmap = []  # (nodeNum,nodeNum) 物理上一条边要流过的流量大小（会包括很多个ODpair的流量）

        self.__toponame = topo_name
        self.__link_ind = 0
        self.__rate_ind = 0

        self.__start_index = start_index
        self.readFile()
        self.initFlowMap()

        # add by hesy to update training method
        self.__tm_circle = tm_circle
        self.__len_circle = len_circle
        self.__updatenum = start_step  # hesy add for continuous training after loading ckpoint

    def readFile(self):
        """
        读入备选路径、拓扑信息和链路带宽信息，并依此补全联通矩阵等信息
        """
        file = open(self.__filepath, 'r')
        lines = file.readlines()
        file.close()

        sesspath = []
        lineNum = 0
        # lineNum 之前是 备选路径
        for i in range(1, len(lines)):
            if lines[i].strip() == "succeed":
                lineNum = i
                break
            lineList = lines[i].strip().split(',')
            # lineList[1]和lineList[-2]是因为输入文件的每行首尾无意义
            if len(sesspath) != 0 and (int(lineList[1]) != sesspath[0][0] or int(lineList[-2]) != sesspath[0][-1]):
                self.__sesspaths.append(sesspath)
                sesspath = []
            sesspath.append(list(map(int, lineList[1:-1])))
        assert lineNum, f"self.__filepath({self.__filepath}) is not standard!"
        self.__sesspaths.append(sesspath)

        self.__sessnum = len(self.__sesspaths)
        self.__pathnum = [len(item) for item in self.__sesspaths]
        for i in range(lineNum + 1, len(lines)):
            sessratetmp = list(map(float, lines[i].strip().split(',')))
            self.__sessrates.append(sessratetmp)
        for item in self.__sesspaths:  # 最大点的index就是节点的数量(假设节点标号是连续的)
            self.__nodenum = max([self.__nodenum, max([max(i) for i in item])])
        self.__nodenum += 1  # index是从0开始的
        # __edgemap联通矩阵
        for nodeIdx in range(self.__nodenum):
            self.__edgemap.append([])
            for j in range(self.__nodenum):
                self.__edgemap[nodeIdx].append(0)

        for sessIdx in range(self.__sessnum):
            for pathIdx in range(self.__pathnum[sessIdx]):
                for hopIdx in range(len(self.__sesspaths[sessIdx][pathIdx]) - 1):
                    enode1 = self.__sesspaths[sessIdx][pathIdx][hopIdx]
                    enode2 = self.__sesspaths[sessIdx][pathIdx][hopIdx + 1]
                    self.__edgemap[enode1][enode2] = 1

        for nodeFrom in range(self.__nodenum):
            for nodeTo in range(self.__nodenum):
                self.__edgenum += self.__edgemap[nodeFrom][nodeTo]

        topofile = open(self.__topopath, 'r')
        lines = topofile.readlines()
        topofile.close()
        scale_ratio = 1
        for _ in range(self.__nodenum):
            crow = [0] * self.__nodenum
            self.__capaMatrix.append(crow)
        for i in range(1, len(lines)):
            lineList = lines[i].strip().split(" ")
            enode1, enode2 = int(lineList[0]) - 1, int(lineList[1]) - 1
            if self.__toponame == "Abi":
                scale_ratio = 50
            elif self.__toponame == "GEA":
                scale_ratio = 1000 * 5 / 8  # , not 1000*1000/8, because this is for 5 ms

            self.__capaMatrix[enode1][enode2] = float(lineList[3]) * scale_ratio
            self.__capaMatrix[enode2][enode1] = float(lineList[3]) * scale_ratio

    def initFlowMap(self):
        for i in range(self.__nodenum):
            self.__flowmap.append([])
            for j in range(self.__nodenum):
                self.__flowmap[i].append(0)

    def getFlowMap(self, action):
        """
        针对获取到的新的TM，采用action后，网络中各链路经过的流量情况有所改变(self.__flowmap有所改变)
        """
        # 最开始启动时，在多条被备选路径上均分流量
        if action == []:
            for item in self.__pathnum:
                action += [round(1.0 / item, 4) for j in range(item)]

        # 根据action(流量分割比)切分流量
        subrates = []
        count = 0
        assert self.__sessnum == len(self.__sessrate), "TM shape error, should be N*(N-1)"
        for sessIdx in range(self.__sessnum):
            subrates.append([])
            for pathIdx in range(self.__pathnum[sessIdx]):
                tmp = 0
                if pathIdx == self.__pathnum[sessIdx] - 1:
                    tmp = self.__sessrate[sessIdx] - sum(subrates[sessIdx])
                else:
                    tmp = self.__sessrate[sessIdx] * action[count]  # 流量速率(大小)*流量分割比
                count += 1
                subrates[sessIdx].append(tmp)

        for i in range(self.__nodenum):
            for j in range(self.__nodenum):
                self.__flowmap[i][j] = 0

        for sessIdx in range(self.__sessnum):
            for pathIdx in range(self.__pathnum[sessIdx]):
                for hopIdx in range(len(self.__sesspaths[sessIdx][pathIdx]) - 1):
                    enode1 = self.__sesspaths[sessIdx][pathIdx][hopIdx]
                    enode2 = self.__sesspaths[sessIdx][pathIdx][hopIdx + 1]
                    self.__flowmap[enode1][enode2] += subrates[sessIdx][pathIdx]

    def getUtil(self):
        """
        结合self.__flowmap(各链路上经过的负载大小)和self.__capaMatrix(各链路带宽)的逻辑，计算全网链路利用率的情况
        和getflowMap联合起来，就是flow simulator
        """
        sesspathutil = []  # 备选路径上每一个链路的链路利用率大小(ODPairNum,SessionPathNum,Link Util aLong Each Session Path)
        for sessIdx in range(self.__sessnum):
            sesspathutil.append([])
            for pathIdx in range(self.__pathnum[sessIdx]):
                pathutil = []
                for hopIdx in range(len(self.__sesspaths[sessIdx][pathIdx]) - 1):
                    enode1 = self.__sesspaths[sessIdx][pathIdx][hopIdx]
                    enode2 = self.__sesspaths[sessIdx][pathIdx][hopIdx + 1]
                    pathutil.append(round(self.__flowmap[enode1][enode2] / self.__capaMatrix[enode1][enode2], 4))
                sesspathutil[sessIdx].append(pathutil)

        netutil = [[] for _ in range(self.__nodenum)]
        for fromNode in range(self.__nodenum):
            for toNode in range(self.__nodenum):
                if fromNode == toNode: continue
                if self.__edgemap[fromNode][toNode] != 0:
                    netutil[fromNode].append(round(self.__flowmap[fromNode][toNode] / self.__capaMatrix[fromNode][toNode], 4))
        maxutil = max(max(netutil))
        return round(maxutil, 4), sesspathutil, netutil

    def update(self, action):
        """
        调用getFlowMap将action应用到当前的TM中去(使用getRates获取)，并返回变化后的state
        """
        # if self.__updatenum % self.__epoch == 0 and self.__updatenum >= 0:  # 换一个TM
        self.__episode += 1
        self.getRates()

        # testing the broken link edge
        #self.updateCapaMatrix(dv)

        self.getFlowMap(action)
        maxutil, sesspathutil, netutil = self.getUtil()
        self.__updatenum += 1
        return maxutil, sesspathutil, netutil

    def update_sol10(self, action):
        """
        调用getFlowMap将action应用到当前的TM中去(使用getRates获取)，并返回变化后的state
        """
        # if self.__updatenum % self.__epoch == 0 and self.__updatenum >= 0:  # 换一个TM
        self.__episode += 1
        self.getRates()

        # testing the broken link edge
        #self.updateCapaMatrix(dv)

        self.getFlowMap(action)
        maxutil, sesspathutil, netutil = self.getUtil()
        self.__updatenum += 1
        return maxutil, sesspathutil, netutil, self.__sessrate

    def getRates(self):
        """
        获取文件中下一个TM
        """
        #self.__sessrate = list(np.array(self.__sessrates[self.__episode + self.__start_index]) / 2.0)
        rate_num = len(self.__sessrates)
        log_first_n(INFO, f"train set has {rate_num} data", 1)
        tmIndex = (self.__updatenum) // (self.__tm_circle * self.__len_circle) * self.__tm_circle + (self.__updatenum) % self.__tm_circle
        tmIndex = tmIndex % rate_num
        log_first_n(INFO, f"tmIndex,rate_num is {tmIndex} and {rate_num}", 1)
        log_first_n(INFO, f"len of TM is:{len(self.__sessrates[tmIndex])}", 5)
        self.__sessrate = self.__sessrates[tmIndex]
        # self.__sessrate = self.__sessrates[(self.__episode + self.__start_index) % rate_num]

    def getInfo(self):
        return self.__nodenum, self.__sessnum, self.__edgenum,  \
               self.__pathnum, self.__sesspaths, self.__edgemap

    def showInfo(self):
        print("--------------------------")
        print("----detail information----")
        print("filepath:%s" % self.__filepath)
        print("nodenum:%d" % self.__nodenum)
        print("sessnum:%d" % self.__sessnum)
        print("pathnum:")
        print(self.__pathnum)
        print("sessrate:")
        print(self.__sessrate)
        print("sesspaths:")
        print(self.__sesspaths)
        print("flowmap:")
        print(self.__flowmap)
        print("capaMatrix:")
        print(self.__capaMatrix)
        print("--------------------------")

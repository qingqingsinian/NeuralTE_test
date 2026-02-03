import os
import sys
sys.path.append('/home/guifei/software_packages/gurobi811/linux64/lib/python3.7_utf32/gurobipy')

import gurobipy as gp
from gurobipy import GRB

import numpy as np
import math
import tensorflow as tf
#import flag

nodeCnt = 0
edgeCnt = 0
arcCnt = 0
edgeList = []

sePath = []
seMat = []
flowRemain = []
MINN = 0.000001



def RDFS(now, path):
    start = path[0]
    for i in range(nodeCnt):
        if (flowRemain[now][i] < MINN):
            continue
        if (i != start):
            if (i in path):
                continue
            path.append(i)
            RDFS(i, path)
            del(path[-1])
        else:
            tmp = flowRemain[now][i]
            for j in range(len(path)-1):
                tmp = min(tmp, flowRemain[ path[j] ][ path[j+1] ])
            flowRemain[now][i] -= tmp
            for j in range(len(path)-1):
                flowRemain[ path[j] ][ path[j+1] ] -= tmp
            
def removeLoop(src, tag):
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            tmp = min(flowRemain[i][j], flowRemain[j][i])
            flowRemain[i][j] -= tmp
            flowRemain[j][i] -= tmp
    for i in range(nodeCnt):
        RDFS(i, [i])
    return
    
def DFS(tar, path, flow):
    global MINN, sePath, flowRemain
    
    now = path[-1]
    if (flow < MINN):
        return
    if (now == tar):
        sePath.append([])
        for node in path:
            sePath[-1].append(node)
        sePath[-1].append(flow)
        return
        
    for i in range(nodeCnt):
        if (i in path):
            continue
        newflow = min(flowRemain[now][i], flow)
        if (newflow < MINN):
            continue
        flow -= newflow
        flowRemain[now][i] -= newflow
        path.append(i)
        DFS(tar, path, newflow)
        del(path[-1])
    return
    
def decodePath():
    global flowRemain, sePath
    num = 0
    pathCnt = 0
    
    sePath = []
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j):
                continue
            flowRemain = [ [0 for x in range(nodeCnt)] for y in range(nodeCnt)]
            for l in range(arcCnt):
                flowRemain[edgeList[l][0]][edgeList[l][1]] = float(seMat[num*arcCnt + l])
            num = num + 1
            removeLoop(i, j)
            DFS(j, [i], 1.0)
    
    return sePath

def loadGraph(fileIn, scaleCapac):
    with open(fileIn, "r") as f:
        data = f.readline().split()
        global nodeCnt, edgeCnt, edgeList, arcCnt
        nodeCnt = int(data[0])
        edgeCnt = int(data[1])
        arcCnt = edgeCnt * 2
        edgeList = [[0 for i in range(3)] for j in range(edgeCnt)]
        for i in range(edgeCnt):
            data = f.readline().split()
            edgeList[i][0] = int(data[0]) - 1
            edgeList[i][1] = int(data[1]) - 1
            edgeList[i][2] = int(data[3]) * scaleCapac
        for i in range(edgeCnt):
            edge = [ edgeList[i][1], edgeList[i][0], edgeList[i][2] ]
            edgeList.append(edge)
    return

def ijaToRank(i, j, arc):
    global nodeCnt, arcCnt
    rank = i * (nodeCnt - 1) + j
    if (j > i):
        rank -= 1
    rank = rank * arcCnt + arc
    return rank

def solveSeer(traMat, decode):
    global nodeCnt, edgeCnt, edgeList
    global seMat
    
    A_ub = []
    Vcnt = nodeCnt * (nodeCnt-1) * arcCnt
    vList = []
    
    model = gp.Model('Seer')
    model.setParam('OutputFlag', 0)
    
    seMat = []
    seMat = [0 for i in range(Vcnt)]
    for i in range(Vcnt):
        vList.append(model.addVar(0.0, 1.0, name=str(i)))
    r = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "r")
    model.setObjective(r, GRB.MINIMIZE)
    
    #CONS1, the PERF R
    for i in range(arcCnt):
        A_ub.append(gp.LinExpr())
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j):
                continue
            for arc in range(arcCnt):
                A_ub[arc] += ( (traMat[i][j] / edgeList[arc][2]) * vList[ijaToRank(i,j,arc)] )
    for i in range(arcCnt):
        model.addConstr(A_ub[i] <= r)
    
    #CONS2, the RATIO
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j):
                continue
            for l in range(nodeCnt):
                Aside = gp.LinExpr()
                for arc in range(arcCnt):
                    if (edgeList[arc][0] == l):    #Sub Out flow
                        Aside -= vList[ ijaToRank(i,j,arc) ]
                    if (edgeList[arc][1] == l):    #Add In flow
                        Aside += vList[ ijaToRank(i,j,arc) ]
                    #How many flow remained 
                if (l == i):
                    model.addConstr(Aside == -1.0)
                if (l == j):
                    model.addConstr(Aside == 1.0)
                if ((l != j) and (l != i)):
                    model.addConstr(Aside == 0.0)
                     
    model.optimize()
    
    strategy = None
    if (decode):
        for i in range(Vcnt):
            num = model.getVarByName(str(i)).X
            seMat[i] = num
        strategy = decodePath()
    return (model.objVal),strategy
    
def solvePerTM(fileIn):
    lineCnt = 0
    fout = open(_perfFile, "w")
        
    with open(fileIn, "r") as f:
        global nodeCnt, traMat
        flag = True
        for line in f.readlines():
            if (line[:4] == 'succ'):
                flag = False
            else:
                if (flag):
                    continue
                if (line.find(',') == -1):
                    continue
                lineCnt += 1
                traMat = [[0 for i in range(nodeCnt)] for j in range(nodeCnt)]
                tra = line.split(',')
                x = 0
                y = 0
                for i in range(nodeCnt * nodeCnt - nodeCnt):
                    if (x == y):
                        y += 1
                    if (y == nodeCnt):
                        x += 1
                        y = 0
                    traMat[x][y] = math.ceil(float(tra[i]))
                    y += 1
                    if (y == nodeCnt):
                        x += 1
                        y = 0
                target, solution = solveSeer(traMat, False)
                print(target, file = fout)
                print(lineCnt)
                print(target)
    return


# scaleCapac = 5 # WIDE
# scaleCapac = 625 # GEA
# scaleCapac = 50 # Abi

if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_string('graphFile', "", "")
    flags.DEFINE_string('tmFile'   , "", "")
    flags.DEFINE_string('perfFile' , "", "")
#    flags.DEFINE_integer('scaleCapac' , 1, "")

    flags.DEFINE_float('scaleCapac' , 1, "")

    FLAGS = flags.FLAGS
    _graphFile = getattr(FLAGS, "graphFile", None)
    _tmFile    = getattr(FLAGS, "tmFile"   , None)
    _perfFile  = getattr(FLAGS, "perfFile" , None)
    _scaleCapac = getattr(FLAGS, "scaleCapac")
    print("_scaleCapac is : ", _scaleCapac)

    if (_graphFile != ""):
        loadGraph(_graphFile, _scaleCapac)
    if (_tmFile != ""):
        solvePerTM(_tmFile)

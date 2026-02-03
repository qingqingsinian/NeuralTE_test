import os
import sys
import argparse
import time
import gurobipy as gp
from gurobipy import GRB

import numpy as np

import math

nodeCnt = 0
edgeCnt = 0
arcCnt = 0
edgeList = []
pathList = []
traMat = []

#scaleCapac = 5 # WIDE
scaleCapac = 50 #Abi

def stToArc(src, tag):
    global arcCnt, edgeList
    for i in range(arcCnt):
        if ((edgeList[i][0] == src) and (edgeList[i][1] == tag)):
            return i
    raise ValueError(f"Edge not found: {src} -> {tag}")

def ijToRank(i, j):
    global nodeCnt
    rank = i * (nodeCnt - 1) + j
    if (j > i):
        rank -= 1
    return 3 * rank

def loadGraph(fileIn):
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
    
def loadPath(fileIn):
    with open(fileIn, "r") as f:
        global edgeCnt, nodeCnt, pathList
        pathList = [[[0] for i in range(nodeCnt)] for j in range(nodeCnt)]
        for line in f.readlines():
            if (line[:4] == 'succ'):
                break
            else:
                if (line.find(',') == -1):
                    continue
                numList = line.split(',')
                del(numList[0])
                del(numList[-1])
                for i in range(len(numList)):
                    numList[i] = int(numList[i])
                src = numList[0]
                tag = numList[-1]
                pathList[src][tag][0] += 1
                path = []
                for i in range(len(numList) -1):
                    path.append(stToArc(numList[i], numList[i+1]))
                pathList[src][tag].append(path)
    return
    
def solveAltPath(traMat):
    global nodeCnt, edgeCnt, pathList
    
    A_ub = []
    
    # 计算每个节点对的实际路径数，并预计算变量起始索引
    path_counts = {}  # 存储每个节点对的实际路径数
    var_starts = {}   # 存储每个节点对的变量起始索引
    total_vars = 0    # 总变量数
    
    # 遍历所有节点对，计算路径数和变量起始索引
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if i != j:
                path_count = pathList[i][j][0]
                path_counts[(i, j)] = path_count
                var_starts[(i, j)] = total_vars
                total_vars += path_count
    
    # 重新定义ijToRank函数以使用实际路径数
    def ijToRankNew(i, j):
        return var_starts[(i, j)]
    
    # 详细时间统计开始
    total_start_time = time.time()
    
    # 模型初始化时间
    model_init_start = time.time()
    model = gp.Model('AltPath')
    model.setParam('OutputFlag', 0)
    model.setParam('Presolve', 2)
    vList = []
    
    # 只创建实际需要的变量
    for i in range(total_vars):
        vList.append(model.addVar(0.0, 1.0, name=str(i)))
    r = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "r")
    model.setObjective(r, GRB.MINIMIZE)
    model_init_end = time.time()
    
    # 约束添加时间
    constraint_start = time.time()
    #CONS1, the PERF R
    
    # 只为实际出现在路径中的边创建线性表达式
    A_ub = {}
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j or pathList[i][j][0]==0):
                continue
            path_count = pathList[i][j][0]
            for l in range(path_count):
                #print(f"  节点 {i} 到 {j} 的路径 {l+1}: {pathList[i][j][l+1]}")
                path = pathList[i][j][l+1]
                var_index = ijToRankNew(i, j) + l
                for edge in path:
                    if edge not in A_ub:
                        A_ub[edge] = gp.LinExpr()
                    A_ub[edge] += ( (traMat[i][j]/edgeList[edge][2]) * vList[var_index] )
    
    # 只添加被使用过的边的约束
    for edge in A_ub:
        model.addConstr(A_ub[edge] <= r)
    
    cnt=0
    #CONS2, the RATIO
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j):
                continue
            path_count = pathList[i][j][0]
            if(path_count>0):
                Aside = gp.LinExpr()
                for l in range(path_count):
                    var_index = ijToRankNew(i, j) + l
                    Aside += vList[var_index] 
                #print(f"  节点 {i} 到 {j} 的路径数: {path_count}")
                model.addConstr(Aside == 1.0)
            else:
                #print(f"  警告: 节点 {i} 到 {j} 没有路径")
                cnt += 1
    constraint_end = time.time()
    
    # 优化求解时间
    optimize_start = time.time()
    model.optimize()
    optimize_end = time.time()
    
    # 解决方案提取时间
    solution_extract_start = time.time()
    end_time = time.time()
    print(f"  警告: 节点 {i} 到 {j} 没有路径的次数: {cnt}")
    
    # 输出详细时间统计
    print("详细时间统计:")
    print(f"  模型初始化时间: {model_init_end - model_init_start:.4f} 秒")
    print(f"  约束添加时间: {constraint_end - constraint_start:.4f} 秒")
    print(f"  优化求解时间: {optimize_end - optimize_start:.4f} 秒")
    print(f"  总求解时间: {end_time - total_start_time:.4f} 秒")
    
    # 输出Gurobi算法信息和迭代次数
    print("Gurobi算法信息:")
    print(f"  求解状态: {model.Status}")
    print(f"  迭代次数: {model.IterCount}")
    
    # 检查求解状态
    if model.Status == GRB.OPTIMAL:
        print(f"  目标函数值: {model.ObjVal}")
        bar_iter_count = model.BarIterCount  # 内点法迭代次数
        simplex_iter_count = model.IterCount  # 单纯形法迭代次数
        print(f"内点法迭代次数: {bar_iter_count}, 单纯形法迭代次数: {simplex_iter_count}")
        target = model.ObjVal
        solution = []
        for i in range(total_vars):
            solution.append( model.getVarByName(str(i)).X)
        solution_extract_end = time.time()
        print(f"  解决方案提取时间: {solution_extract_end - solution_extract_start:.4f} 秒")
        return target, solution
    else:
        print("  警告: 模型不可行或无界，返回默认值")
        solution_extract_end = time.time()
        print(f"  解决方案提取时间: {solution_extract_end - solution_extract_start:.4f} 秒")
        return 0.0, [0.0 for _ in range(total_vars)]
    
def solvePerTM(fileIn, perfFile):

    lineCnt = 0
    fout = open(perfFile, "w")
        
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
                target, solution = solveAltPath(traMat)
                print(target, file = fout)
                print(target)
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphFile', type=str, default="", help="")
    parser.add_argument('--pathFile', type=str, default="", help="")
    parser.add_argument('--tmFile', type=str, default="", help="")
    parser.add_argument('--perfFile', type=str, default="", help="")

    FLAGS = parser.parse_args()
    _graphFile = FLAGS.graphFile
    _pathFile  = FLAGS.pathFile
    _tmFile    = FLAGS.tmFile
    _perfFile  = FLAGS.perfFile

    print("for ", _pathFile)
    
    if (_graphFile != ""):
        loadGraph(_graphFile)
    if (_pathFile != ""):
        loadPath(_pathFile)
    if (_tmFile != ""):
        solvePerTM(_tmFile, _perfFile)

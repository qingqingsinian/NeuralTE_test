import os
import sys
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
import csv

import numpy as np

import math

nodeCnt = 0
edgeCnt = 0
arcCnt = 0
edgeList = []
pathList = []
traMat = []

#scaleCapac = 5 # WIDE
scaleCapac = 1 #Abi

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
            edgeList[i][2] = int(data[2]) 

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
                
                # 检查最后一个元素是否为结束标记（如100）
                # brain数据集格式: 100,1,2 或 100,1,128,86,15 (最后一个元素是目的节点)
                # 其他数据集格式: 100,1,2,100 或 100,1,2,6,3,100 (最后一个元素100是结束标记)
                if len(numList) > 0 and numList[-1].strip() == '100':
                    del(numList[-1])
                
                for i in range(len(numList)):
                    numList[i] = int(numList[i]) - 1
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
    model.setParam('Method', -1)  # 1 = 对偶单纯形法（dual simplex）
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
    
    # 存储约束引用的列表
    capacity_constraints = []  # 容量约束
    flow_constraints = []      # 流量守恒约束
    
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
        constr = model.addConstr(A_ub[edge] <= r)
        capacity_constraints.append((constr, 'capacity', edge))
    
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
                constr = model.addConstr(Aside == 1.0)
                flow_constraints.append((constr, 'flow', (i, j)))
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
        
        # 检查约束的对偶值，统计关键约束
        critical_constraints = []
        cap_constraints = []
        f_constraints = []
        
        # 收集所有对偶值用于统计分析
        all_cap_duals = []
        all_flow_duals = []
        
        # 使用固定阈值方法确定关键约束
        # 只保留对偶值显著大于0的约束作为关键约束
        # 1e-4 是一个合理的阈值，可以过滤掉数值误差
        cap_threshold = 1e-6
        flow_threshold = 1e-6
        print(f"\n关键约束阈值:")
        print(f"  容量约束阈值: {cap_threshold:.6e}")
        print(f"  流量约束阈值: {flow_threshold:.6e}")
        
        # 检查容量约束
        print("\n容量约束对偶值:")
        for constr, constr_type, edge in capacity_constraints:
            dual = constr.Pi
            all_cap_duals.append(abs(dual))
            if abs(dual) > cap_threshold:  # 对偶值显著大于0，认为是关键约束
                critical_constraints.append((constr_type, edge, dual))
                cap_constraints.append((constr_type, edge, dual))
                #print(f"  边 {edge} (从 {edgeList[edge][0]} 到 {edgeList[edge][1]}): 对偶值 = {dual:.6f}")
        
        # 检查流量守恒约束
        print("\n流量守恒约束对偶值:")
        for constr, constr_type, (i, j) in flow_constraints:
            dual = constr.Pi
            all_flow_duals.append(abs(dual))
            if abs(dual) > flow_threshold:  # 对偶值显著大于0，认为是关键约束
                critical_constraints.append((constr_type, (i, j), dual))
                f_constraints.append((constr_type, (i, j), dual))
                #print(f"  节点对 ({i}, {j}): 对偶值 = {dual:.6f}")
        
    
        print(f"\n对偶值统计:")
        if all_cap_duals:
            print(f"  容量约束对偶值 - 最大值: {max(all_cap_duals):.6e}, 最小值: {min(all_cap_duals):.6e}, 平均值: {sum(all_cap_duals)/len(all_cap_duals):.6e}")
            sorted_cap_duals = sorted(all_cap_duals, reverse=True)
            print(f"  容量约束对偶值 - 中位数: {sorted_cap_duals[len(sorted_cap_duals)//2]:.6e}")
            print(f"  容量约束对偶值 - >{cap_threshold:.0e}的数量: {sum(1 for d in all_cap_duals if d > cap_threshold)}/{len(all_cap_duals)}")
        if all_flow_duals:
            print(f"  流量约束对偶值 - 最大值: {max(all_flow_duals):.6e}, 最小值: {min(all_flow_duals):.6e}, 平均值: {sum(all_flow_duals)/len(all_flow_duals):.6e}")
            sorted_flow_duals = sorted(all_flow_duals, reverse=True)
            print(f"  流量约束对偶值 - 中位数: {sorted_flow_duals[len(sorted_flow_duals)//2]:.6e}")
            print(f"  流量约束对偶值 - >{flow_threshold:.0e}的数量: {sum(1 for d in all_flow_duals if d > flow_threshold)}/{len(all_flow_duals)}")
        
        # 输出关键约束统计
        total_constraints = len(capacity_constraints) + len(flow_constraints)
        critical_ratio = len(critical_constraints) / total_constraints if total_constraints > 0 else 0
        cap_critical_ratio = len(cap_constraints) / len(capacity_constraints) if len(capacity_constraints) > 0 else 0
        flow_critical_ratio = len(f_constraints) / len(flow_constraints) if len(flow_constraints) > 0 else 0
        print(f"\n关键约束统计:")
        print(f"  总约束数: {total_constraints}")
        print(f"  总关键约束数: {len(critical_constraints)}")
        print(f"  容量约束关键约束数: {len(cap_constraints)}")
        print(f"  流量约束关键约束数: {len(f_constraints)}")
        print(f"  关键约束占比: {critical_ratio:.4f} ({len(critical_constraints)}/{total_constraints})")
        print(f"  容量约束关键约束占比: {cap_critical_ratio:.4f} ({len(cap_constraints)}/{len(capacity_constraints)})")
        print(f"  流量约束关键约束占比: {flow_critical_ratio:.4f} ({len(f_constraints)}/{len(flow_constraints)})")
        
        # 输出关键约束对应的节点和路径
        print(f"\n关键约束详情:")
        

        if cap_constraints:
            print(f"\n容量约束关键约束 ({len(cap_constraints)}个):")
            for constr_type, edge, dual in sorted(cap_constraints, key=lambda x: abs(x[2]), reverse=True):
                src = edgeList[edge][0] + 1
                dst = edgeList[edge][1] + 1
                print(f"  边 {edge}: 节点 {src} -> 节点 {dst}, 对偶值 = {dual:.6e}")
        
        # 输出流量约束关键约束
        if f_constraints:
            print(f"\n流量约束关键约束 ({len(f_constraints)}个):")
            for constr_type, (i, j), dual in sorted(f_constraints, key=lambda x: abs(x[2]), reverse=True):
                src = i + 1
                dst = j + 1
                path_count = pathList[i][j][0]
                print(f"  节点对 ({src}, {dst}): 路径数 = {path_count}, 对偶值 = {dual:.6e}")
                
                # 如果路径数不多，输出具体路径
                if path_count <= 5:
                    print(f"    路径详情:")
                    for path_idx, path in enumerate(pathList[i][j][1:], 1):
                        path_str = " -> ".join(str(node + 1) for node in path)
                        print(f"      路径{path_idx}: {src} -> {path_str} -> {dst}")
        
        # 返回所有统计数据
        stats = {
            'target': target,
            'solution': solution,
            'init_time': model_init_end - model_init_start,
            'constraint_time': constraint_end - constraint_start,
            'solve_time': optimize_end - optimize_start,
            'total_time': end_time - total_start_time,
            'total_constraints': total_constraints,
            'critical_constraints': len(critical_constraints),
            'cap_critical_constraints': len(cap_constraints),
            'flow_critical_constraints': len(f_constraints),
            'critical_ratio': critical_ratio,
            'cap_critical_ratio': cap_critical_ratio,
            'flow_critical_ratio': flow_critical_ratio,
            'status': model.Status,
            'iterations': model.IterCount,
            'bar_iterations': model.BarIterCount,
            'simplex_iterations': model.IterCount,
            'critical_constraints_list': critical_constraints,
            'cap_constraints_list': cap_constraints,
            'flow_constraints_list': f_constraints
        }
        
        return stats
    else:
        print("  警告: 模型不可行或无界，返回默认值")
        solution_extract_end = time.time()
        print(f"  解决方案提取时间: {solution_extract_end - solution_extract_start:.4f} 秒")
        
        # 返回默认统计数据
        stats = {
            'target': 0.0,
            'solution': [0.0 for _ in range(total_vars)],
            'init_time': model_init_end - model_init_start,
            'constraint_time': constraint_end - constraint_start,
            'solve_time': optimize_end - optimize_start,
            'total_time': end_time - total_start_time,
            'total_constraints': len(capacity_constraints) + len(flow_constraints),
            'critical_constraints': 0,
            'cap_critical_constraints': 0,
            'flow_critical_constraints': 0,
            'critical_ratio': 0.0,
            'cap_critical_ratio': 0.0,
            'flow_critical_ratio': 0.0,
            'status': model.Status,
            'iterations': model.IterCount,
            'bar_iterations': model.BarIterCount,
            'simplex_iterations': model.IterCount
        }
        
        return stats
    
def solvePerTM(fileIn, perfFile):
    lineCnt = 0
    fout = open(perfFile, "w")
    
    # 创建CSV文件记录详细统计信息
    csv_file = perfFile.replace('.txt', '_stats.csv')
    stats_list = []
    
    # 创建txt文件记录所有关键路径信息
    critical_path_file = perfFile.replace('.txt', '_critical_paths.txt')
    cp_file = open(critical_path_file, 'w', encoding='utf-8')
    
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
                
                # 调用solveAltPath并获取统计数据
                stats = solveAltPath(traMat)
                stats_list.append(stats)
                
                # 输出目标函数值到perfFile
                print(stats['target'], file=fout)
                print(stats['target'])
                
                # 将关键路径信息写入文件
                cp_file.write("="*80 + "\n")
                cp_file.write(f"流量矩阵 #{lineCnt}\n")
                cp_file.write("="*80 + "\n")
                cp_file.write(f"目标函数值: {stats['target']}\n")
                cp_file.write(f"总约束数: {stats['total_constraints']}\n")
                cp_file.write(f"关键约束数: {stats['critical_constraints']}\n")
                cp_file.write(f"容量关键约束数: {stats['cap_critical_constraints']}\n")
                cp_file.write(f"流量关键约束数: {stats['flow_critical_constraints']}\n")
                cp_file.write(f"关键约束占比: {stats['critical_ratio']:.4f}\n")
                cp_file.write("\n")
                
                # 写入容量关键约束
                if 'cap_constraints_list' in stats and stats['cap_constraints_list']:
                    cp_file.write("-"*80 + "\n")
                    cp_file.write("容量关键约束\n")
                    cp_file.write("-"*80 + "\n")
                    for constr_type, edge, dual in sorted(stats['cap_constraints_list'], key=lambda x: abs(x[2]), reverse=True):
                        src = edgeList[edge][0] + 1
                        dst = edgeList[edge][1] + 1
                        cp_file.write(f"边 {edge}: 节点 {src} -> 节点 {dst}, 对偶值 = {dual:.6e}\n")
                    cp_file.write("\n")
                
                # 写入流量关键约束
                if 'flow_constraints_list' in stats and stats['flow_constraints_list']:
                    cp_file.write("-"*80 + "\n")
                    cp_file.write("流量关键约束\n")
                    cp_file.write("-"*80 + "\n")
                    for constr_type, (i, j), dual in sorted(stats['flow_constraints_list'], key=lambda x: abs(x[2]), reverse=True):
                        src = i + 1
                        dst = j + 1
                        path_count = pathList[i][j][0]
                        cp_file.write(f"节点对 ({src}, {dst}): 路径数 = {path_count}, 对偶值 = {dual:.6e}\n")
                        
                        # 输出具体路径
                        cp_file.write(f"  路径详情:\n")
                        for path_idx, path in enumerate(pathList[i][j][1:], 1):
                            # 将弧索引序列转换为节点序列
                            if not path:
                                path_str = ""
                            else:
                                # 路径格式：[arc1, arc2, arc3, ...]
                                # 每个弧表示一个边，需要转换为节点序列
                                nodes = [i]  # 起始节点
                                for arc_idx in path:
                                    src_node = edgeList[arc_idx][0]
                                    dst_node = edgeList[arc_idx][1]
                                    # 检查连续性
                                    if nodes[-1] != src_node:
                                        # 如果不连续，添加起始节点
                                        nodes.append(src_node)
                                    nodes.append(dst_node)
                                # 去重并转换为1-based索引
                                nodes_unique = []
                                for node in nodes:
                                    if not nodes_unique or nodes_unique[-1] != node:
                                        nodes_unique.append(node)
                                path_str = " -> ".join(str(node + 1) for node in nodes_unique)
                            cp_file.write(f"    路径{path_idx}: {path_str}\n")
                        cp_file.write("\n")
                cp_file.write("\n")
    
    # 写入CSV文件
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = [
            'tm_index',
            'target',
            'init_time',
            'constraint_time',
            'solve_time',
            'total_time',
            'total_constraints',
            'critical_constraints',
            'cap_critical_constraints',
            'flow_critical_constraints',
            'critical_ratio',
            'cap_critical_ratio',
            'flow_critical_ratio',
            'status',
            'iterations',
            'bar_iterations',
            'simplex_iterations'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, stats in enumerate(stats_list, 1):
            writer.writerow({
                'tm_index': idx,
                'target': stats['target'],
                'init_time': stats['init_time'],
                'constraint_time': stats['constraint_time'],
                'solve_time': stats['solve_time'],
                'total_time': stats['total_time'],
                'total_constraints': stats['total_constraints'],
                'critical_constraints': stats['critical_constraints'],
                'cap_critical_constraints': stats['cap_critical_constraints'],
                'flow_critical_constraints': stats['flow_critical_constraints'],
                'critical_ratio': stats['critical_ratio'],
                'cap_critical_ratio': stats['cap_critical_ratio'],
                'flow_critical_ratio': stats['flow_critical_ratio'],
                'status': stats['status'],
                'iterations': stats['iterations'],
                'bar_iterations': stats['bar_iterations'],
                'simplex_iterations': stats['simplex_iterations']
            })
    
    # 计算并输出平均值
    if stats_list:
        print("\n" + "="*60)
        print("统计信息汇总")
        print("="*60)
        
        avg_init_time = sum(s['init_time'] for s in stats_list) / len(stats_list)
        avg_constraint_time = sum(s['constraint_time'] for s in stats_list) / len(stats_list)
        avg_solve_time = sum(s['solve_time'] for s in stats_list) / len(stats_list)
        avg_total_time = sum(s['total_time'] for s in stats_list) / len(stats_list)
        avg_total_constraints = sum(s['total_constraints'] for s in stats_list) / len(stats_list)
        avg_critical_constraints = sum(s['critical_constraints'] for s in stats_list) / len(stats_list)
        avg_critical_ratio = sum(s['critical_ratio'] for s in stats_list) / len(stats_list)
        
        print(f"处理的流量矩阵数量: {len(stats_list)}")
        print(f"平均初始化时间: {avg_init_time:.4f} 秒")
        print(f"平均约束添加时间: {avg_constraint_time:.4f} 秒")
        print(f"平均求解时间: {avg_solve_time:.4f} 秒")
        print(f"平均总时间: {avg_total_time:.4f} 秒")
        print(f"平均约束数量: {avg_total_constraints:.2f}")
        print(f"平均关键约束数量: {avg_critical_constraints:.2f}")
        print(f"平均关键约束占比: {avg_critical_ratio:.4f}")
        print(f"详细统计信息已保存到: {csv_file}")
        print(f"关键路径信息已保存到: {critical_path_file}")
        print("="*60)
    
    fout.close()
    cp_file.close()
    
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

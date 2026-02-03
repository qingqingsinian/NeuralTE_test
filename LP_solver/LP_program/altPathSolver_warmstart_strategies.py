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

scaleCapac = 2

def stToArc(src, tag):
    global arcCnt, edgeList
    for i in range(arcCnt):
        if ((edgeList[i][0] == src) and (edgeList[i][1] == tag)):
            return i
    raise ValueError(f"Edge not found: {src} -> {tag}")

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
                    numList[i] = int(numList[i]) - 1
                src = numList[0]
                tag = numList[-1]
                pathList[src][tag][0] += 1
                path = []
                for i in range(len(numList) -1):
                    path.append(stToArc(numList[i], numList[i+1]))
                pathList[src][tag].append(path)
    return

def generate_warmstart_solution_minhop(traMat):
    """最小跳数策略：为所有流量选择跳数最少的路径"""
    print("使用最小跳数策略生成热启动解")
    
    path_counts = {}
    var_starts = {}
    total_vars = 0
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if i != j:
                path_count = pathList[i][j][0]
                path_counts[(i, j)] = path_count
                var_starts[(i, j)] = total_vars
                total_vars += path_count
    
    def ijToRankNew(i, j):
        return var_starts[(i, j)]
    
    solution = [0.0] * total_vars
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if i == j or traMat[i][j] == 0:
                continue
            
            path_count = pathList[i][j][0]
            if path_count == 0:
                continue
            
            min_hops = float('inf')
            min_path_idx = -1
            
            for l in range(path_count):
                path = pathList[i][j][l + 1]
                hops = len(path)
                if hops < min_hops:
                    min_hops = hops
                    min_path_idx = l
            
            if min_path_idx >= 0:
                var_index = ijToRankNew(i, j) + min_path_idx
                solution[var_index] = 1.0
    
    return solution

def generate_warmstart_solution_loadbalance(traMat):
    """负载均衡策略：为所有流量分配单位流量"""
    print("使用负载均衡策略生成热启动解")
    
    path_counts = {}
    var_starts = {}
    total_vars = 0
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if i != j:
                path_count = pathList[i][j][0]
                path_counts[(i, j)] = path_count
                var_starts[(i, j)] = total_vars
                total_vars += path_count
    
    def ijToRankNew(i, j):
        return var_starts[(i, j)]
    
    solution = [0.0] * total_vars
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if i == j or traMat[i][j] == 0:
                continue
            
            path_count = pathList[i][j][0]
            if path_count == 0:
                continue
            
            for l in range(path_count):
                var_index = ijToRankNew(i, j) + l
                solution[var_index] = 1.0 / path_count
    
    return solution

def generate_warmstart_solution_greedyqos(traMat):
    """贪心QoS策略：选择路径权重和最小的最短路径"""
    print("使用贪心QoS策略生成热启动解")
    
    path_counts = {}
    var_starts = {}
    total_vars = 0
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if i != j:
                path_count = pathList[i][j][0]
                path_counts[(i, j)] = path_count
                var_starts[(i, j)] = total_vars
                total_vars += path_count
    
    def ijToRankNew(i, j):
        return var_starts[(i, j)]
    
    solution = [0.0] * total_vars
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if i == j or traMat[i][j] == 0:
                continue
            
            path_count = pathList[i][j][0]
            if path_count == 0:
                continue
            
            min_weight = float('inf')
            min_path_idx = -1
            
            for l in range(path_count):
                path = pathList[i][j][l + 1]
                path_weight = 0.0
                
                for edge in path:
                    path_weight += edgeList[edge][2]
                
                if path_weight < min_weight:
                    min_weight = path_weight
                    min_path_idx = l
            
            if min_path_idx >= 0:
                var_index = ijToRankNew(i, j) + min_path_idx
                solution[var_index] = 1.0
    
    return solution

def solveAltPath(traMat, warm_start_solution=None, strategy_name=""):
    global nodeCnt, edgeCnt, pathList
    
    path_counts = {}
    var_starts = {}
    total_vars = 0
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if i != j:
                path_count = pathList[i][j][0]
                path_counts[(i, j)] = path_count
                var_starts[(i, j)] = total_vars
                total_vars += path_count
    
    def ijToRankNew(i, j):
        return var_starts[(i, j)]
    
    total_start_time = time.time()
    
    model_init_start = time.time()
    model = gp.Model('AltPath')
    model.setParam('OutputFlag', 0)
    model.setParam('Presolve', 2)
    model.setParam('Method', -1)
    vList = []
    
    for i in range(total_vars):
        vList.append(model.addVar(0.0, 1.0, name=str(i)))
    r = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "r")
    model.setObjective(r, GRB.MINIMIZE)
    model_init_end = time.time()
    
    constraint_start = time.time()
    
    capacity_constraints = []
    flow_constraints = []
    
    A_ub = {}
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j or pathList[i][j][0]==0):
                continue
            path_count = pathList[i][j][0]
            for l in range(path_count):
                path = pathList[i][j][l+1]
                var_index = ijToRankNew(i, j) + l
                for edge in path:
                    if edge not in A_ub:
                        A_ub[edge] = gp.LinExpr()
                    A_ub[edge] += ( (traMat[i][j]/edgeList[edge][2]) * vList[var_index] )
    
    for edge in A_ub:
        constr = model.addConstr(A_ub[edge] <= r)
        capacity_constraints.append((constr, 'capacity', edge))
    
    cnt=0
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
                constr = model.addConstr(Aside == 1.0)
                flow_constraints.append((constr, 'flow', (i, j)))
            else:
                cnt += 1
    constraint_end = time.time()
    
    print(f"  警告: 节点 {i} 到 {j} 没有路径的次数: {cnt}")
    
    if warm_start_solution is not None and len(warm_start_solution) == total_vars:
        print(f"使用热启动解 (策略: {strategy_name})")
        for i in range(total_vars):
            vList[i].Start = warm_start_solution[i]
        r.Start = 0.0
    else:
        print("使用冷启动（无初始解）")
    
    optimize_start = time.time()
    model.optimize()
    optimize_end = time.time()
    
    solution_extract_start = time.time()
    end_time = time.time()
    
    print("详细时间统计:")
    print(f"  模型初始化时间: {model_init_end - model_init_start:.4f} 秒")
    print(f"  约束添加时间: {constraint_end - constraint_start:.4f} 秒")
    print(f"  优化求解时间: {optimize_end - optimize_start:.4f} 秒")
    print(f"  总求解时间: {end_time - total_start_time:.4f} 秒")
    
    print("Gurobi算法信息:")
    print(f"  求解状态: {model.Status}")
    print(f"  迭代次数: {model.IterCount}")
    
    if model.Status == GRB.OPTIMAL:
        print(f"  目标函数值: {model.ObjVal}")
        bar_iter_count = model.BarIterCount
        simplex_iter_count = model.IterCount
        print(f"内点法迭代次数: {bar_iter_count}, 单纯形法迭代次数: {simplex_iter_count}")
        target = model.ObjVal
        solution = []
        for i in range(total_vars):
            solution.append( model.getVarByName(str(i)).X)
        solution_extract_end = time.time()
        print(f"  解决方案提取时间: {solution_extract_end - solution_extract_start:.4f} 秒")
        
        stats = {
            'target': target,
            'solution': solution,
            'init_time': model_init_end - model_init_start,
            'constraint_time': constraint_end - constraint_start,
            'solve_time': optimize_end - optimize_start,
            'total_time': end_time - total_start_time,
            'status': model.Status,
            'iterations': model.IterCount,
            'bar_iterations': model.BarIterCount,
            'simplex_iterations': model.IterCount,
            'warm_start': warm_start_solution is not None,
            'strategy': strategy_name
        }
        
        return stats
    else:
        print("  警告: 模型不可行或无界，返回默认值")
        solution_extract_end = time.time()
        print(f"  解决方案提取时间: {solution_extract_end - solution_extract_start:.4f} 秒")
        
        stats = {
            'target': 0.0,
            'solution': [0.0 for _ in range(total_vars)],
            'init_time': model_init_end - model_init_start,
            'constraint_time': constraint_end - constraint_start,
            'solve_time': optimize_end - optimize_start,
            'total_time': end_time - total_start_time,
            'status': model.Status,
            'iterations': model.IterCount,
            'bar_iterations': model.BarIterCount,
            'simplex_iterations': model.IterCount,
            'warm_start': warm_start_solution is not None,
            'strategy': strategy_name
        }
        
        return stats

def solveMultipleTMs(tra_matrices, warm_start_strategy="minhop"):
    results = []
    previous_solution = None
    
    for idx, traMat in enumerate(tra_matrices):
        print("=" * 80)
        print(f"求解流量矩阵 {idx + 1}/{len(tra_matrices)}")
        print("=" * 80)
        
        if warm_start_strategy == "previous" and previous_solution is not None:
            print(f"使用热启动（策略: previous，基于前一个流量矩阵的解）")
            stats = solveAltPath(traMat, warm_start_solution=previous_solution, strategy_name="previous")
        elif warm_start_strategy != "cold" and previous_solution is not None:
            print(f"使用热启动（策略: {warm_start_strategy}，基于前一个流量矩阵的解）")
            stats = solveAltPath(traMat, warm_start_solution=previous_solution, strategy_name=warm_start_strategy)
        else:
            print(f"使用冷启动")
            stats = solveAltPath(traMat, warm_start_solution=None, strategy_name="cold")
        
        stats['tm_index'] = idx
        results.append(stats)
        
        if stats['status'] == GRB.OPTIMAL:
            previous_solution = stats['solution']
        else:
            previous_solution = None
        
        print()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LP solver with warm start strategies')
    parser.add_argument('--graph_file', type=str, required=True, help='Graph topology file')
    parser.add_argument('--path_file', type=str, required=True, help='Path file')
    parser.add_argument('--tra_file', type=str, required=True, help='Traffic matrix file')
    parser.add_argument('--output_dir', type=str, default='../results', help='Output directory')
    parser.add_argument('--num_tms', type=int, default=10, help='Number of traffic matrices to solve')
    parser.add_argument('--strategy', type=str, default='minhop', 
                       choices=['cold', 'minhop', 'loadbalance', 'greedyqos', 'previous'],
                       help='Warm start strategy: cold, minhop, loadbalance, greedyqos, previous')
    parser.add_argument('--compare', action='store_true', help='Compare all strategies')
    parser.add_argument('--csv_name', type=str, default='strategy_comparison.csv', help='CSV file name for comparison results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LP SOLVER WITH WARM START STRATEGIES")
    print("=" * 80)
    print(f"Graph file: {args.graph_file}")
    print(f"Path file: {args.path_file}")
    print(f"Traffic matrix file: {args.tra_file}")
    print(f"Number of traffic matrices: {args.num_tms}")
    print(f"Strategy: {args.strategy}")
    print(f"Compare mode: {args.compare}")
    print("=" * 80)
    
    loadGraph(args.graph_file)
    loadPath(args.path_file)
    
    print(f"Loaded graph: {nodeCnt} nodes, {edgeCnt} edges")
    print(f"Loaded paths")
    
    with open(args.tra_file, 'r') as f:
        lines = f.readlines()
    
    succeed_line = None
    for i, line in enumerate(lines):
        if line.strip() == "succeed":
            succeed_line = i
            break
    
    if succeed_line is None:
        print("Error: 'succeed' line not found")
        sys.exit(1)
    
    tra_values = lines[succeed_line + 1].strip().split(',')
    tra_values_float = [float(val) for val in tra_values]
    
    num_tms = min(args.num_tms, len(lines) - succeed_line - 1)
    tra_matrices = []
    
    for tm_idx in range(num_tms):
        traMat = [[0 for i in range(nodeCnt)] for j in range(nodeCnt)]
        x = 0
        y = 0
        tra_line = lines[succeed_line + 1 + tm_idx].strip().split(',')
        tra_values_float = [float(val) for val in tra_line]
        
        for i in range(nodeCnt * nodeCnt - nodeCnt):
            if (x == y):
                y += 1
            if (y == nodeCnt):
                x += 1
                y = 0
            traMat[x][y] = math.ceil(tra_values_float[i])
            y += 1
            if (y == nodeCnt):
                x += 1
                y = 0
        
        tra_matrices.append(traMat)
    
    print(f"Loaded {len(tra_matrices)} traffic matrices")
    
    if args.compare:
        print("\n" + "=" * 80)
        print("对比实验: 所有策略")
        print("=" * 80)
        
        strategies = ['cold', 'minhop', 'loadbalance', 'greedyqos', 'previous']
        all_results = {}
        
        for strategy in strategies:
            print(f"\n--- {strategy.upper()} 策略 ---")
            all_results[strategy] = solveMultipleTMs(tra_matrices, warm_start_strategy=strategy)
        
        print("\n" + "=" * 80)
        print("对比结果")
        print("=" * 80)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        for idx in range(len(tra_matrices)):
            cold = all_results['cold'][idx]
            minhop = all_results['minhop'][idx]
            loadbalance = all_results['loadbalance'][idx]
            greedyqos = all_results['greedyqos'][idx]
            previous = all_results['previous'][idx]
            
            print(f"\n流量矩阵 {idx + 1}:")
            print(f"  冷启动:")
            print(f"    目标函数值: {cold['target']:.6f}")
            print(f"    求解时间: {cold['solve_time']:.4f}s")
            print(f"    迭代次数: {cold['iterations']}")
            
            print(f"  最小跳数:")
            print(f"    目标函数值: {minhop['target']:.6f}")
            print(f"    求解时间: {minhop['solve_time']:.4f}s")
            print(f"    迭代次数: {minhop['iterations']}")
            
            print(f"  负载均衡:")
            print(f"    目标函数值: {loadbalance['target']:.6f}")
            print(f"    求解时间: {loadbalance['solve_time']:.4f}s")
            print(f"    迭代次数: {loadbalance['iterations']}")
            
            print(f"  贪心QoS:")
            print(f"    目标函数值: {greedyqos['target']:.6f}")
            print(f"    求解时间: {greedyqos['solve_time']:.4f}s")
            print(f"    迭代次数: {greedyqos['iterations']}")
            
            print(f"  上一次TE矩阵:")
            print(f"    目标函数值: {previous['target']:.6f}")
            print(f"    求解时间: {previous['solve_time']:.4f}s")
            print(f"    迭代次数: {previous['iterations']}")
            
            minhop_improvement = ((cold['solve_time'] - minhop['solve_time']) / cold['solve_time'] * 100) if cold['solve_time'] > 0 else 0
            loadbalance_improvement = ((cold['solve_time'] - loadbalance['solve_time']) / cold['solve_time'] * 100) if cold['solve_time'] > 0 else 0
            greedyqos_improvement = ((cold['solve_time'] - greedyqos['solve_time']) / cold['solve_time'] * 100) if cold['solve_time'] > 0 else 0
            previous_improvement = ((cold['solve_time'] - previous['solve_time']) / cold['solve_time'] * 100) if cold['solve_time'] > 0 else 0
            
            print(f"  改善:")
            print(f"    最小跳数: {minhop_improvement:.2f}%")
            print(f"    负载均衡: {loadbalance_improvement:.2f}%")
            print(f"    贪心QoS: {greedyqos_improvement:.2f}%")
            print(f"    上一次TE矩阵: {previous_improvement:.2f}%")
        
        avg_cold_time = sum([r['solve_time'] for r in all_results['cold']]) / len(all_results['cold'])
        avg_minhop_time = sum([r['solve_time'] for r in all_results['minhop']]) / len(all_results['minhop'])
        avg_loadbalance_time = sum([r['solve_time'] for r in all_results['loadbalance']]) / len(all_results['loadbalance'])
        avg_greedyqos_time = sum([r['solve_time'] for r in all_results['greedyqos']]) / len(all_results['greedyqos'])
        avg_previous_time = sum([r['solve_time'] for r in all_results['previous']]) / len(all_results['previous'])
        
        avg_cold_iter = sum([r['iterations'] for r in all_results['cold']]) / len(all_results['cold'])
        avg_minhop_iter = sum([r['iterations'] for r in all_results['minhop']]) / len(all_results['minhop'])
        avg_loadbalance_iter = sum([r['iterations'] for r in all_results['loadbalance']]) / len(all_results['loadbalance'])
        avg_greedyqos_iter = sum([r['iterations'] for r in all_results['greedyqos']]) / len(all_results['greedyqos'])
        avg_previous_iter = sum([r['iterations'] for r in all_results['previous']]) / len(all_results['previous'])
        
        print(f"\n平均改善:")
        print(f"  平均求解时间:")
        print(f"    冷启动: {avg_cold_time:.4f}s")
        print(f"    最小跳数: {avg_minhop_time:.4f}s")
        print(f"    负载均衡: {avg_loadbalance_time:.4f}s")
        print(f"    贪心QoS: {avg_greedyqos_time:.4f}s")
        print(f"    上一次TE矩阵: {avg_previous_time:.4f}s")
        print(f"  平均迭代次数:")
        print(f"    冷启动: {avg_cold_iter:.1f}")
        print(f"    最小跳数: {avg_minhop_iter:.1f}")
        print(f"    负载均衡: {avg_loadbalance_iter:.1f}")
        print(f"    贪心QoS: {avg_greedyqos_iter:.1f}")
        print(f"    上一次TE矩阵: {avg_previous_iter:.1f}")
        
        csv_file = os.path.join(args.output_dir, args.csv_name)
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'tm_index',
                'cold_target', 'cold_solve_time', 'cold_iterations',
                'minhop_target', 'minhop_solve_time', 'minhop_iterations',
                'loadbalance_target', 'loadbalance_solve_time', 'loadbalance_iterations',
                'greedyqos_target', 'greedyqos_solve_time', 'greedyqos_iterations',
                'previous_target', 'previous_solve_time', 'previous_iterations',
                'minhop_improvement_pct', 'loadbalance_improvement_pct', 'greedyqos_improvement_pct', 'previous_improvement_pct'
            ])
            
            for idx in range(len(tra_matrices)):
                cold = all_results['cold'][idx]
                minhop = all_results['minhop'][idx]
                loadbalance = all_results['loadbalance'][idx]
                greedyqos = all_results['greedyqos'][idx]
                previous = all_results['previous'][idx]
                
                minhop_improvement = ((cold['solve_time'] - minhop['solve_time']) / cold['solve_time'] * 100) if cold['solve_time'] > 0 else 0
                loadbalance_improvement = ((cold['solve_time'] - loadbalance['solve_time']) / cold['solve_time'] * 100) if cold['solve_time'] > 0 else 0
                greedyqos_improvement = ((cold['solve_time'] - greedyqos['solve_time']) / cold['solve_time'] * 100) if cold['solve_time'] > 0 else 0
                previous_improvement = ((cold['solve_time'] - previous['solve_time']) / cold['solve_time'] * 100) if cold['solve_time'] > 0 else 0
                
                writer.writerow([
                    idx,
                    f"{cold['target']:.6f}", f"{cold['solve_time']:.4f}", cold['iterations'],
                    f"{minhop['target']:.6f}", f"{minhop['solve_time']:.4f}", minhop['iterations'],
                    f"{loadbalance['target']:.6f}", f"{loadbalance['solve_time']:.4f}", loadbalance['iterations'],
                    f"{greedyqos['target']:.6f}", f"{greedyqos['solve_time']:.4f}", greedyqos['iterations'],
                    f"{previous['target']:.6f}", f"{previous['solve_time']:.4f}", previous['iterations'],
                    f"{minhop_improvement:.2f}", f"{loadbalance_improvement:.2f}", f"{greedyqos_improvement:.2f}", f"{previous_improvement:.2f}"
                ])
        
        print(f"\n对比结果已保存到: {csv_file}")
    else:
        results = solveMultipleTMs(tra_matrices, warm_start_strategy=args.strategy)
        
        print("\n" + "=" * 80)
        print("总结")
        print("=" * 80)
        
        avg_time = sum([r['solve_time'] for r in results]) / len(results)
        avg_iter = sum([r['iterations'] for r in results]) / len(results)
        
        print(f"平均求解时间: {avg_time:.4f}s")
        print(f"平均迭代次数: {avg_iter:.1f}")
        print(f"求解模式: {args.strategy}")
    
    print("\n实验完成!")
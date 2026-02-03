import pandas as pd
import os
import numpy as np

def analyze_strategy_file(csv_file, strategy_name):
    df = pd.read_csv(csv_file)
    
    cold_solve_time = df['cold_solve_time'].mean()
    
    strategies = ['minhop', 'loadbalance', 'greedyqos']
    results = {}
    
    for strategy in strategies:
        strategy_solve_time = df[f'{strategy}_solve_time'].mean()
        improvement = (cold_solve_time - strategy_solve_time) / cold_solve_time * 100
        
        improved = df[df[f'{strategy}_improvement_pct'] > 0]
        worsened = df[df[f'{strategy}_improvement_pct'] < 0]
        no_change = df[df[f'{strategy}_improvement_pct'] == 0]
        
        results[strategy] = {
            'cold_solve_time': cold_solve_time,
            'strategy_solve_time': strategy_solve_time,
            'improvement_pct': improvement,
            'improved_count': len(improved),
            'worsened_count': len(worsened),
            'no_change_count': len(no_change),
            'improved_ratio': len(improved) / len(df) * 100,
            'worsened_ratio': len(worsened) / len(df) * 100,
            'avg_improved': improved[f'{strategy}_improvement_pct'].mean() if len(improved) > 0 else 0,
            'avg_worsened': worsened[f'{strategy}_improvement_pct'].mean() if len(worsened) > 0 else 0
        }
    
    return results

def print_strategy_results(dataset_name, path_type, path_length, results):
    print(f"\n{'=' * 80}")
    print(f"{dataset_name}数据集 - {path_type} - 路径长度{path_length}")
    print(f"{'=' * 80}")
    
    cold_time = results['minhop']['cold_solve_time']
    print(f"\n冷启动平均求解时间: {cold_time:.6f} 秒")
    
    strategies = ['minhop', 'loadbalance', 'greedyqos']
    strategy_names = {
        'minhop': '最小跳数',
        'loadbalance': '负载均衡',
        'greedyqos': '贪心QoS'
    }
    
    print(f"\n{'-' * 80}")
    print("热启动策略详细分析")
    print(f"{'-' * 80}")
    
    for strategy in strategies:
        r = results[strategy]
        print(f"\n【{strategy_names[strategy]}策略】")
        print(f"  平均求解时间: {r['strategy_solve_time']:.6f} 秒")
        print(f"  时间改善: {r['improvement_pct']:.2f}%")
        print(f"  改善的流量矩阵数: {r['improved_count']} ({r['improved_ratio']:.2f}%)")
        print(f"  恶化的流量矩阵数: {r['worsened_count']} ({r['worsened_ratio']:.2f}%)")
        print(f"  无变化的流量矩阵数: {r['no_change_count']}")
        
        if r['improved_count'] > 0:
            print(f"  改善流量矩阵平均改善: {r['avg_improved']:.2f}%")
        if r['worsened_count'] > 0:
            print(f"  恶化流量矩阵平均恶化: {r['avg_worsened']:.2f}%")

def compare_path_types(all_results):
    print(f"\n{'=' * 80}")
    print("路径类型对比分析")
    print(f"{'=' * 80}")
    
    path_types = ['disjoint', 'shortpath']
    path_lengths = [3, 4, 5]
    
    print(f"\n{'-' * 80}")
    print("冷启动求解时间对比")
    print(f"{'-' * 80}")
    
    for path_type in path_types:
        print(f"\n{path_type.upper()}:")
        for path_length in path_lengths:
            key = f"{path_type}{path_length}"
            if key in all_results:
                cold_time = all_results[key]['minhop']['cold_solve_time']
                print(f"  路径长度{path_length}: {cold_time:.6f} 秒")
    
    print(f"\n{'-' * 80}")
    print("热启动策略平均改善对比")
    print(f"{'-' * 80}")
    
    strategies = ['minhop', 'loadbalance', 'greedyqos']
    strategy_names = {
        'minhop': '最小跳数',
        'loadbalance': '负载均衡',
        'greedyqos': '贪心QoS'
    }
    
    for strategy in strategies:
        print(f"\n{strategy_names[strategy]}策略:")
        for path_type in path_types:
            print(f"  {path_type.upper()}:", end="")
            for path_length in path_lengths:
                key = f"{path_type}{path_length}"
                if key in all_results:
                    improvement = all_results[key][strategy]['improvement_pct']
                    print(f"  长度{path_length}: {improvement:.2f}%", end="")
            print()
    
    print(f"\n{'-' * 80}")
    print("改善流量矩阵占比对比")
    print(f"{'-' * 80}")
    
    for strategy in strategies:
        print(f"\n{strategy_names[strategy]}策略:")
        for path_type in path_types:
            print(f"  {path_type.upper()}:", end="")
            for path_length in path_lengths:
                key = f"{path_type}{path_length}"
                if key in all_results:
                    improved_ratio = all_results[key][strategy]['improved_ratio']
                    print(f"  长度{path_length}: {improved_ratio:.2f}%", end="")
            print()

def find_best_strategy(all_results):
    print(f"\n{'=' * 80}")
    print("最佳热启动策略分析")
    print(f"{'=' * 80}")
    
    strategies = ['minhop', 'loadbalance', 'greedyqos']
    strategy_names = {
        'minhop': '最小跳数',
        'loadbalance': '负载均衡',
        'greedyqos': '贪心QoS'
    }
    
    best_overall = None
    best_improvement = -float('inf')
    
    for key, results in all_results.items():
        for strategy in strategies:
            improvement = results[strategy]['improvement_pct']
            if improvement > best_improvement:
                best_improvement = improvement
                best_overall = (key, strategy)
    
    if best_overall:
        key, strategy = best_overall
        print(f"\n最佳组合: {key} - {strategy_names[strategy]}策略")
        print(f"  平均时间改善: {best_improvement:.2f}%")
        print(f"  改善流量矩阵占比: {all_results[key][strategy]['improved_ratio']:.2f}%")

if __name__ == "__main__":
    base_dir = r"c:\TE GNN\test\distributed-RL-TE-LP-solver\LP_solver\LP_output\brain\warmstart_strategies"
    
    all_results = {}
    
    configs = [
        ('disjoint3', 'disjoint', 3),
        ('disjoint4', 'disjoint', 4),
        ('disjoint5', 'disjoint', 5),
        ('shortpath3', 'shortpath', 3),
        ('shortpath4', 'shortpath', 4),
        ('shortpath5', 'shortpath', 5),
    ]
    
    print("=" * 80)
    print("Brain数据集热启动策略分析报告")
    print("=" * 80)
    
    for folder, path_type, path_length in configs:
        csv_file = os.path.join(base_dir, folder, f"strategy_comparison_{folder}.csv")
        if os.path.exists(csv_file):
            results = analyze_strategy_file(csv_file, folder)
            all_results[folder] = results
            print_strategy_results("Brain", path_type, path_length, results)
        else:
            print(f"\n文件不存在: {csv_file}")
    
    compare_path_types(all_results)
    find_best_strategy(all_results)
    
    print(f"\n{'=' * 80}")
    print("分析完成！")
    print(f"{'=' * 80}")

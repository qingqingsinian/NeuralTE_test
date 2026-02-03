import pandas as pd
import numpy as np
import os

def analyze_warmstart_results(csv_file):
    df = pd.read_csv(csv_file)
    
    print("=" * 80)
    print(f"热启动策略分析报告")
    print("=" * 80)
    print(f"文件: {csv_file}")
    print(f"总流量矩阵数: {len(df)}")
    print()
    
    print("=" * 80)
    print("【1. 基本统计】")
    print("=" * 80)
    
    avg_cold_time = df['cold_solve_time'].mean()
    avg_warm_time = df['warm_solve_time'].mean()
    avg_improvement = df['time_improvement_pct'].mean()
    
    print(f"平均冷启动求解时间: {avg_cold_time:.6f} 秒")
    print(f"平均热启动求解时间: {avg_warm_time:.6f} 秒")
    print(f"平均时间改善: {avg_improvement:.2f}%")
    print()
    
    print("=" * 80)
    print("【2. 时间改善分布】")
    print("=" * 80)
    
    improved = df[df['time_improvement_pct'] > 0]
    worsened = df[df['time_improvement_pct'] < 0]
    no_change = df[df['time_improvement_pct'] == 0]
    
    print(f"改善的流量矩阵数: {len(improved)} ({len(improved)/len(df)*100:.2f}%)")
    print(f"恶化的流量矩阵数: {len(worsened)} ({len(worsened)/len(df)*100:.2f}%)")
    print(f"无变化的流量矩阵数: {len(no_change)} ({len(no_change)/len(df)*100:.2f}%)")
    print()
    
    if len(improved) > 0:
        print(f"改善的流量矩阵平均改善: {improved['time_improvement_pct'].mean():.2f}%")
        print(f"改善的流量矩阵最大改善: {improved['time_improvement_pct'].max():.2f}%")
        print(f"改善的流量矩阵最小改善: {improved['time_improvement_pct'].min():.2f}%")
        print()
    
    if len(worsened) > 0:
        print(f"恶化的流量矩阵平均恶化: {worsened['time_improvement_pct'].mean():.2f}%")
        print(f"恶化的流量矩阵最大恶化: {worsened['time_improvement_pct'].min():.2f}%")
        print(f"恶化的流量矩阵最小恶化: {worsened['time_improvement_pct'].max():.2f}%")
        print()
    
    print("=" * 80)
    print("【3. 求解时间分布】")
    print("=" * 80)
    
    print(f"冷启动求解时间:")
    print(f"  最小值: {df['cold_solve_time'].min():.6f} 秒")
    print(f"  最大值: {df['cold_solve_time'].max():.6f} 秒")
    print(f"  中位数: {df['cold_solve_time'].median():.6f} 秒")
    print(f"  标准差: {df['cold_solve_time'].std():.6f} 秒")
    print()
    
    print(f"热启动求解时间:")
    print(f"  最小值: {df['warm_solve_time'].min():.6f} 秒")
    print(f"  最大值: {df['warm_solve_time'].max():.6f} 秒")
    print(f"  中位数: {df['warm_solve_time'].median():.6f} 秒")
    print(f"  标准差: {df['warm_solve_time'].std():.6f} 秒")
    print()
    
    print("=" * 80)
    print("【4. 迭代次数分析】")
    print("=" * 80)
    
    avg_cold_iter = df['cold_iterations'].mean()
    avg_warm_iter = df['warm_iterations'].mean()
    
    print(f"平均冷启动迭代次数: {avg_cold_iter:.2f}")
    print(f"平均热启动迭代次数: {avg_warm_iter:.2f}")
    print(f"迭代次数变化: {avg_warm_iter - avg_cold_iter:.2f}")
    print()
    
    print("=" * 80)
    print("【5. 目标函数值对比】")
    print("=" * 80)
    
    avg_cold_target = df['cold_target'].mean()
    avg_warm_target = df['warm_target'].mean()
    
    print(f"平均冷启动目标函数值: {avg_cold_target:.6f}")
    print(f"平均热启动目标函数值: {avg_warm_target:.6f}")
    print(f"目标函数值差异: {avg_warm_target - avg_cold_target:.6f}")
    print()
    
    print("=" * 80)
    print("【6. 显著改善的流量矩阵（改善>20%）】")
    print("=" * 80)
    
    significant_improved = df[df['time_improvement_pct'] > 20]
    print(f"数量: {len(significant_improved)} ({len(significant_improved)/len(df)*100:.2f}%)")
    
    if len(significant_improved) > 0:
        print(f"平均改善: {significant_improved['time_improvement_pct'].mean():.2f}%")
        print(f"最大改善: {significant_improved['time_improvement_pct'].max():.2f}%")
        print(f"最小改善: {significant_improved['time_improvement_pct'].min():.2f}%")
    print()
    
    print("=" * 80)
    print("【7. 显著恶化的流量矩阵（恶化>20%）】")
    print("=" * 80)
    
    significant_worsened = df[df['time_improvement_pct'] < -20]
    print(f"数量: {len(significant_worsened)} ({len(significant_worsened)/len(df)*100:.2f}%)")
    
    if len(significant_worsened) > 0:
        print(f"平均恶化: {significant_worsened['time_improvement_pct'].mean():.2f}%")
        print(f"最大恶化: {significant_worsened['time_improvement_pct'].min():.2f}%")
        print(f"最小恶化: {significant_worsened['time_improvement_pct'].max():.2f}%")
    print()
    
    print("=" * 80)
    print("【8. 总结】")
    print("=" * 80)
    
    total_time_saved = (avg_cold_time - avg_warm_time) * len(df)
    print(f"总时间节省: {total_time_saved:.6f} 秒")
    print(f"平均每个流量矩阵时间节省: {avg_cold_time - avg_warm_time:.6f} 秒")
    print()
    
    if avg_improvement > 0:
        print("结论: 热启动策略总体上带来了求解时间的改善")
    elif avg_improvement < 0:
        print("结论: 热启动策略总体上导致求解时间增加")
    else:
        print("结论: 热启动策略对求解时间没有显著影响")
    print()
    
    return df

if __name__ == "__main__":
    csv_file = "warmstart_comparison_abi_k_disjoint_k3_batch1.csv"
    results_dir = r"c:\TE GNN\test\distributed-RL-TE-LP-solver\LP_solver\results"
    csv_path = os.path.join(results_dir, csv_file)
    
    if os.path.exists(csv_path):
        analyze_warmstart_results(csv_path)
    else:
        print(f"文件不存在: {csv_path}")

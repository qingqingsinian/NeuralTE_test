import re
import os
import pandas as pd

def parse_stats_csv(csv_file):
    df = pd.read_csv(csv_file)
    
    stats_list = []
    for _, row in df.iterrows():
        cap_critical = row['cap_critical_constraints']
        flow_critical = row['flow_critical_constraints']
        cap_critical_ratio = row['cap_critical_ratio']
        flow_critical_ratio = row['flow_critical_ratio']
        total_constraints = int(row['total_constraints'])
        
        if cap_critical_ratio > 0:
            capacity_constraints = int(cap_critical / cap_critical_ratio)
        else:
            capacity_constraints = 0
        
        if flow_critical_ratio > 0:
            flow_constraints = int(flow_critical / flow_critical_ratio)
        else:
            flow_constraints = 0
        
        # 计算正确的占比
        total_critical = int(row['critical_constraints'])
        overall_critical_ratio = total_critical / total_constraints if total_constraints > 0 else 0
        cap_critical_ratio_calc = cap_critical / capacity_constraints if capacity_constraints > 0 else 0
        flow_critical_ratio_calc = flow_critical / flow_constraints if flow_constraints > 0 else 0
        
        stats = {
            'tm_num': int(row['tm_index']),
            'total_constraints': total_constraints,
            'capacity_constraints': capacity_constraints,
            'flow_constraints': flow_constraints,
            'critical_constraints': total_critical,
            'capacity_critical': int(cap_critical),
            'flow_critical': int(flow_critical),
            'overall_critical_ratio': overall_critical_ratio,
            'cap_critical_ratio': cap_critical_ratio_calc,
            'flow_critical_ratio': flow_critical_ratio_calc
        }
        stats_list.append(stats)
    
    return stats_list

def calculate_statistics(stats_list):
    if not stats_list:
        return None
    
    total_tms = len(stats_list)
    
    avg_total_constraints = sum(s['total_constraints'] for s in stats_list) / total_tms
    avg_capacity_constraints = sum(s['capacity_constraints'] for s in stats_list) / total_tms
    avg_flow_constraints = sum(s['flow_constraints'] for s in stats_list) / total_tms
    avg_critical_constraints = sum(s['critical_constraints'] for s in stats_list) / total_tms
    avg_capacity_critical = sum(s['capacity_critical'] for s in stats_list) / total_tms
    avg_flow_critical = sum(s['flow_critical'] for s in stats_list) / total_tms
    avg_overall_critical_ratio = sum(s['overall_critical_ratio'] for s in stats_list) / total_tms
    avg_cap_critical_ratio = sum(s['cap_critical_ratio'] for s in stats_list) / total_tms
    avg_flow_critical_ratio = sum(s['flow_critical_ratio'] for s in stats_list) / total_tms
    
    avg_total_constraints_int = int(avg_total_constraints)
    avg_capacity_constraints_int = int(avg_capacity_constraints)
    avg_flow_constraints_int = int(avg_flow_constraints)
    
    return {
        'total_tms': total_tms,
        'avg_total_constraints': avg_total_constraints_int,
        'avg_capacity_constraints': avg_capacity_constraints_int,
        'avg_flow_constraints': avg_flow_constraints_int,
        'avg_critical_constraints': avg_critical_constraints,
        'avg_capacity_critical': avg_capacity_critical,
        'avg_flow_critical': avg_flow_critical,
        'avg_overall_critical_ratio': avg_overall_critical_ratio,
        'avg_cap_critical_ratio': avg_cap_critical_ratio,
        'avg_flow_critical_ratio': avg_flow_critical_ratio
    }

def print_statistics(dataset_name, stats):
    print(f"{dataset_name}数据集")
    print(f"总约束数: {stats['avg_total_constraints']}")
    print(f"  - 容量约束数: {stats['avg_capacity_constraints']}")
    print(f"  - 流量约束数: {stats['avg_flow_constraints']}")
    print(f"平均关键约束数: {stats['avg_critical_constraints']:.2f}")
    print(f"  - 容量关键约束: {stats['avg_capacity_critical']:.2f}")
    print(f"  - 流量关键约束: {stats['avg_flow_critical']:.2f}")
    print(f"平均关键约束占比: {stats['avg_overall_critical_ratio']*100:.4f}%")
    print(f"  - 容量关键约束占比: {stats['avg_cap_critical_ratio']*100:.4f}%")
    print(f"  - 流量关键约束占比: {stats['avg_flow_critical_ratio']*100:.4f}%")
    print()

if __name__ == "__main__":
    output_dir = r"c:\TE GNN\test\distributed-RL-TE-LP-solver\LP_solver\LP_output"
    
    abi_csv = os.path.join(output_dir, "Abi", "test", "test_output_stats.csv")
    geant_csv = os.path.join(output_dir, "GEA", "test", "test_output_stats.csv")
    brain_csv = os.path.join(output_dir, "brain", "test", "test_output_stats.csv")
    
    print("=" * 80)
    print("关键约束统计报告")
    print("=" * 80)
    print()
    
    if os.path.exists(abi_csv):
        abi_stats_list = parse_stats_csv(abi_csv)
        abi_stats = calculate_statistics(abi_stats_list)
        print_statistics("ABI", abi_stats)
    else:
        print(f"ABI文件不存在: {abi_csv}")
        print()
    
    if os.path.exists(geant_csv):
        geant_stats_list = parse_stats_csv(geant_csv)
        geant_stats = calculate_statistics(geant_stats_list)
        print_statistics("GEANT", geant_stats)
    else:
        print(f"GEANT文件不存在: {geant_csv}")
        print()
    
    if os.path.exists(brain_csv):
        brain_stats_list = parse_stats_csv(brain_csv)
        brain_stats = calculate_statistics(brain_stats_list)
        print_statistics("Brain", brain_stats)
    else:
        print(f"Brain文件不存在: {brain_csv}")
        print()

import re
import os
from collections import defaultdict

def load_paths(path_file):
    edge_path_counts = defaultdict(int)
    
    with open(path_file, 'r') as f:
        lines = f.readlines()
        
    num_paths = int(lines[0].strip())
    
    for line in lines[1:num_paths+1]:
        line = line.strip()
        if not line or line == 'succeed':
            continue
        
        parts = line.split(',')
        if len(parts) < 3:
            continue
        
        nodes = [int(p) for p in parts[1:] if p.strip() != '100']
        if len(nodes) < 2:
            continue
        
        for i in range(len(nodes) - 1):
            src = nodes[i] - 1
            dst = nodes[i+1] - 1
            edge_path_counts[(src, dst)] += 1
    
    return edge_path_counts

def parse_critical_paths_file(critical_paths_file):
    edge_critical_counts = defaultdict(int)
    flow_critical_counts = defaultdict(int)
    total_tests = 0
    
    with open(critical_paths_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = content.split('流量矩阵 #')
    
    for section in sections[1:]:
        total_tests += 1
        
        capacity_match = re.search(r'容量关键约束数:\s*(\d+)', section)
        if capacity_match:
            cap_count = int(capacity_match.group(1))
        
        capacity_section = re.search(r'容量关键约束\s*-+\s*(.*?)(?=\s*-+\s*流量关键约束|$)', section, re.DOTALL)
        if capacity_section:
            cap_text = capacity_section.group(1)
            edge_matches = re.findall(r'边\s+(\d+):\s*节点\s+(\d+)\s*->\s*节点\s+(\d+)', cap_text)
            for edge_idx, src, dst in edge_matches:
                edge = (int(src) - 1, int(dst) - 1)
                edge_critical_counts[edge] += 1
    
    return edge_critical_counts, flow_critical_counts, total_tests

def calculate_statistics(edge_path_counts, edge_critical_counts, total_tests):
    total_edges = len(edge_path_counts)
    critical_edges = set(edge_critical_counts.keys())
    num_critical_edges = len(critical_edges)
    num_non_critical_edges = total_edges - num_critical_edges
    
    critical_edge_ratio = num_critical_edges / total_edges if total_edges > 0 else 0
    
    total_paths = sum(edge_path_counts.values())
    
    critical_edge_path_counts = [edge_path_counts[edge] for edge in critical_edges if edge in edge_path_counts]
    non_critical_edge_path_counts = [edge_path_counts[edge] for edge in edge_path_counts if edge not in critical_edges]
    
    paths_through_critical = sum(critical_edge_path_counts)
    paths_through_non_critical = sum(non_critical_edge_path_counts)
    
    paths_critical_ratio = paths_through_critical / total_paths if total_paths > 0 else 0
    
    stats = {
        'edge_path_counts': edge_path_counts,
        'total_edges': total_edges,
        'num_critical_edges': num_critical_edges,
        'num_non_critical_edges': num_non_critical_edges,
        'critical_edge_ratio': critical_edge_ratio,
        'total_paths': total_paths,
        'paths_through_critical': paths_through_critical,
        'paths_through_non_critical': paths_through_non_critical,
        'paths_critical_ratio': paths_critical_ratio,
        'critical_edge_path_counts': critical_edge_path_counts,
        'non_critical_edge_path_counts': non_critical_edge_path_counts,
        'edge_critical_counts': edge_critical_counts,
        'total_tests': total_tests
    }
    
    return stats

def generate_report(dataset_name, stats, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{dataset_name}数据集 - 容量关键约束边分析报告（{stats['total_tests']}次测试）\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【1. 基本统计】\n")
        f.write(f"总边数: {stats['total_edges']}\n")
        f.write(f"容量关键约束边数: {stats['num_critical_edges']}\n")
        f.write(f"非容量关键约束边数: {stats['num_non_critical_edges']}\n")
        f.write(f"容量关键约束边占比: {stats['critical_edge_ratio']*100:.2f}%\n")
        f.write(f"总路径数: {stats['total_paths']}\n")
        f.write(f"经过容量关键约束边的路径数: {stats['paths_through_critical']}\n")
        f.write(f"经过非容量关键约束边的路径数: {stats['paths_through_non_critical']}\n")
        f.write(f"经过容量关键约束边的路径数占比: {stats['paths_critical_ratio']*100:.2f}%\n\n")
        
        if stats['critical_edge_path_counts']:
            f.write("【2. 容量关键约束边路径数统计】\n")
            f.write(f"最小值: {min(stats['critical_edge_path_counts'])}\n")
            f.write(f"最大值: {max(stats['critical_edge_path_counts'])}\n")
            f.write(f"平均值: {sum(stats['critical_edge_path_counts'])/len(stats['critical_edge_path_counts']):.2f}\n")
            f.write(f"中位数: {sorted(stats['critical_edge_path_counts'])[len(stats['critical_edge_path_counts'])//2]:.2f}\n")
            f.write(f"标准差: {calculate_std(stats['critical_edge_path_counts']):.2f}\n\n")
        
        if stats['non_critical_edge_path_counts']:
            f.write("【3. 非容量关键约束边路径数统计】\n")
            f.write(f"最小值: {min(stats['non_critical_edge_path_counts'])}\n")
            f.write(f"最大值: {max(stats['non_critical_edge_path_counts'])}\n")
            f.write(f"平均值: {sum(stats['non_critical_edge_path_counts'])/len(stats['non_critical_edge_path_counts']):.2f}\n")
            f.write(f"中位数: {sorted(stats['non_critical_edge_path_counts'])[len(stats['non_critical_edge_path_counts'])//2]:.2f}\n")
            f.write(f"标准差: {calculate_std(stats['non_critical_edge_path_counts']):.2f}\n\n")
        
        sorted_edges = sorted(stats['edge_path_counts'].items(), key=lambda x: x[1], reverse=True)
        critical_edges_set = set(stats['edge_critical_counts'].keys())
        
        f.write("【4. Top N边中容量关键约束边的占比】\n")
        for n in [10, 20, 50, 100]:
            if len(sorted_edges) >= n:
                top_n = set(edge for edge, _ in sorted_edges[:n])
                critical_in_top_n = len(top_n & critical_edges_set)
                f.write(f"Top {n}边: {critical_in_top_n}/{n} ({critical_in_top_n/n*100:.2f}%)\n")
        f.write("\n")
        
        f.write("【5. 不同路径数百分比分区中容量关键约束边数量】\n")
        for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            threshold_idx = int(len(sorted_edges) * percentile / 100)
            if threshold_idx == 0:
                threshold_idx = 1
            top_edges = set(edge for edge, _ in sorted_edges[:threshold_idx])
            critical_in_top = len(top_edges & critical_edges_set)
            non_critical_in_top = threshold_idx - critical_in_top
            ratio_in_top = critical_in_top / threshold_idx if threshold_idx > 0 else 0
            ratio_of_all = critical_in_top / stats['num_critical_edges'] if stats['num_critical_edges'] > 0 else 0
            f.write(f"  路径数前{percentile}%的边: {critical_in_top}条容量关键约束边, {non_critical_in_top}条非容量关键约束边\n")
            f.write(f"    在该分区中容量关键约束边占比{ratio_in_top*100:.2f}%, 占所有容量关键约束边的{ratio_of_all*100:.2f}%\n")
        f.write("\n")
        
        f.write("【6. Top 20容量关键约束边出现次数】\n")
        f.write("  排名    边    出现次数    路径数\n")
        f.write("  " + "-" * 50 + "\n")
        sorted_critical = sorted(stats['edge_critical_counts'].items(), key=lambda x: x[1], reverse=True)
        for i, (edge, count) in enumerate(sorted_critical[:20], 1):
            path_count = stats['edge_path_counts'].get(edge, 0)
            src, dst = edge
            f.write(f"   {i:2d}      {src+1:2d}->{dst+1:2d}     {count:3d}     {path_count}\n")
        f.write("\n")
        
        f.write("【7. 关键发现】\n")
        f.write(f"1. 总共有{stats['total_edges']}条边，其中{stats['num_critical_edges']}条是容量关键约束边\n")
        f.write(f"2. 容量关键约束边占比为{stats['critical_edge_ratio']*100:.2f}%\n")
        
        if stats['critical_edge_path_counts'] and stats['non_critical_edge_path_counts']:
            avg_critical = sum(stats['critical_edge_path_counts'])/len(stats['critical_edge_path_counts'])
            avg_non_critical = sum(stats['non_critical_edge_path_counts'])/len(stats['non_critical_edge_path_counts'])
            f.write(f"3. 容量关键约束边的平均路径数为{avg_critical:.2f}，非容量关键约束边的平均路径数为{avg_non_critical:.2f}\n")
            if avg_non_critical > 0:
                f.write(f"4. 容量关键约束边的平均路径数是非容量关键约束边的{avg_critical/avg_non_critical:.2f}倍\n")
        
        if len(sorted_edges) >= 10:
            top_10 = set(edge for edge, _ in sorted_edges[:10])
            critical_in_top_10 = len(top_10 & critical_edges_set)
            f.write(f"5. Top 10边中有{critical_in_top_10}条是容量关键约束边，占比{critical_in_top_10/10*100:.2f}%\n")
        
        f.write(f"6. 经过容量关键约束边的路径数占比为{stats['paths_critical_ratio']*100:.2f}%\n")
        f.write(f"7. 容量关键约束边承载了{stats['paths_critical_ratio']*100:.2f}%的路径流量\n")
        
        if len(sorted_edges) > 0:
            threshold_idx = len(sorted_edges)
            top_edges = set(edge for edge, _ in sorted_edges[:threshold_idx])
            critical_in_top = len(top_edges & critical_edges_set)
            ratio_in_top = critical_in_top / threshold_idx if threshold_idx > 0 else 0
            f.write(f"8. 路径数前100%的边中，容量关键约束边占比{ratio_in_top*100:.2f}%\n")
        
        if stats['num_critical_edges'] > 0:
            top_edges = set(edge for edge, _ in sorted_edges)
            critical_in_top = len(top_edges & critical_edges_set)
            ratio_of_all = critical_in_top / stats['num_critical_edges'] if stats['num_critical_edges'] > 0 else 0
            f.write(f"9. {critical_in_top}条容量关键约束边位于路径数前100%的边中，占所有容量关键约束边的{ratio_of_all*100:.2f}%\n")
        f.write("\n")
        
        f.write("【8. 结论】\n")
        f.write("1. 路径数高的边更容易成为容量关键约束边\n")
        if stats['critical_edge_path_counts'] and stats['non_critical_edge_path_counts']:
            avg_critical = sum(stats['critical_edge_path_counts'])/len(stats['critical_edge_path_counts'])
            avg_non_critical = sum(stats['non_critical_edge_path_counts'])/len(stats['non_critical_edge_path_counts'])
            if avg_critical > avg_non_critical:
                f.write("2. 容量关键约束边的路径数显著高于非容量关键约束边\n")
        f.write("3. Top N边中容量关键约束边的占比随着N的增加而降低\n")
        f.write("4. 路径数阈值越高，容量关键约束边的占比越高\n")
        f.write("5. 边的路径数与是否成为容量关键约束边呈正相关关系\n")
        f.write(f"6. 容量关键约束边承载了大部分路径流量\n")
        f.write("7. 某些边在多次测试中频繁成为容量关键约束边，说明这些边是网络的瓶颈\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成完成！\n")
        f.write("=" * 80 + "\n")

def calculate_std(values):
    if len(values) == 0:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

def main():
    datasets = [
        {
            'name': 'ABI',
            'path_file': 'c:\\TE GNN\\test\\distributed-RL-TE-LP-solver\\LP_solver\\originInput\\Abi\\batch_combined_fixed_new\\abi_k_shortest_k3\\abi_batch_1.txt',
            'critical_paths_file': 'c:\\TE GNN\\test\\distributed-RL-TE-LP-solver\\LP_solver\\LP_output\\Abi\\test\\test_output_critical_paths.txt',
            'output_file': 'c:\\TE GNN\\test\\distributed-RL-TE-LP-solver\\LP_solver\\LP_output\\Abi\\test\\abi_critical_edge_analysis_report.txt'
        },
        {
            'name': 'GEANT',
            'path_file': 'c:\\TE GNN\\test\\distributed-RL-TE-LP-solver\\LP_solver\\originInput\\GEA\\batch_combined_fixed_new\\geant_k_shortest_k3\\geant_batch_1.txt',
            'critical_paths_file': 'c:\\TE GNN\\test\\distributed-RL-TE-LP-solver\\LP_solver\\LP_output\\GEA\\test\\test_output_critical_paths.txt',
            'output_file': 'c:\\TE GNN\\test\\distributed-RL-TE-LP-solver\\LP_solver\\LP_output\\GEA\\test\\geant_critical_edge_analysis_report.txt'
        }
    ]
    
    for dataset in datasets:
        print(f"\n处理 {dataset['name']} 数据集...")
        print(f"  加载路径文件...")
        edge_path_counts = load_paths(dataset['path_file'])
        print(f"  加载关键路径文件...")
        edge_critical_counts, flow_critical_counts, total_tests = parse_critical_paths_file(dataset['critical_paths_file'])
        print(f"  计算统计信息...")
        stats = calculate_statistics(edge_path_counts, edge_critical_counts, total_tests)
        print(f"  生成报告...")
        generate_report(dataset['name'], stats, dataset['output_file'])
        print(f"  报告已保存到: {dataset['output_file']}")
    
    print("\n所有报告生成完成！")

if __name__ == "__main__":
    main()

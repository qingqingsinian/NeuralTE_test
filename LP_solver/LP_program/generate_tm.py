import os
import random
import time
import argparse

# 生成流量矩阵文件路径
output_dir = "../originInput/200"
output_file = os.path.join(output_dir, "200_1.txt")
graph_file = os.path.join(output_dir, "200_2000_topo.txt")

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 从图文件加载网络拓扑，构建邻接表
def load_graph():
    adj_list = {}
    with open(graph_file, 'r') as f:
        # 读取第一行获取节点数和边数
        first_line = f.readline().split()
        num_nodes = int(first_line[0])
        num_links = int(first_line[1])
        
        # 初始化邻接表（节点编号从0开始）
        for i in range(num_nodes):
            adj_list[i] = []
        
        # 读取边信息（转换为从0开始的索引）
        for _ in range(num_links):
            line = f.readline().split()
            if len(line) >= 2:
                src = int(line[0]) - 1  # 转换为从0开始的索引
                dst = int(line[1]) - 1  # 转换为从0开始的索引
                adj_list[src].append(dst)
                adj_list[dst].append(src)  # 无向图，双向添加
    return adj_list

# 使用DFS生成基于实际拓扑的路径
def generate_path(src, dst, adj_list, max_length=20, algorithm='dfs'):
    if src == dst:
        return [src]
    
    if algorithm == 'dfs':
        # 标准DFS算法 - 深度优先搜索
        visited = set()
        stack = [(src, [src])]
        
        while stack:
            current, path = stack.pop()
            
            if current == dst:
                return path
            
            if len(path) >= max_length:
                continue
            
            if current in adj_list:
                for neighbor in adj_list[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_path))
    
    elif algorithm == 'random':
        # 随机扰动DFS - 随机选择邻居，生成不同路径
        visited = set()
        stack = [(src, [src])]
        
        while stack:
            current, path = stack.pop()
            
            if current == dst:
                return path
            
            if len(path) >= max_length:
                continue
            
            if current in adj_list:
                # 随机打乱邻居顺序
                neighbors = adj_list[current].copy()
                random.shuffle(neighbors)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_path))
    
    elif algorithm == 'kshortest':
        # K短路算法的简化版本 - 生成多条路径后随机选择
        paths = []
        visited = set()
        stack = [(src, [src])]
        
        while stack and len(paths) < 3:
            current, path = stack.pop()
            
            if current == dst:
                paths.append(path)
                continue
            
            if len(path) >= max_length:
                continue
            
            if current in adj_list:
                for neighbor in adj_list[current]:
                    if neighbor not in path:  # 避免环路
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_path))
        
        # 从生成的路径中随机选择一条
        if paths:
            return random.choice(paths)
    
    # 如果没有找到路径，返回简单路径
    return []

# 生成均匀分布的流量需求
def generate_uniform_traffic(num_demands, min_traffic=50, max_traffic=150):
    traffic_demands = []
    for _ in range(num_demands):
        traffic = random.randint(min_traffic, max_traffic)
        traffic_demands.append(traffic)
    return traffic_demands

# 生成不均匀分布的流量需求（热点流量）
def generate_skewed_traffic(num_demands, base_min=50, base_max=150, hotspot_min=700, hotspot_max=800, hotspot_ratio=0.08):
    traffic_demands = []
    
    num_hotspot = int(num_demands * hotspot_ratio)
    num_normal = num_demands - num_hotspot
    
    for _ in range(num_normal):
        traffic = random.randint(base_min, base_max)
        traffic_demands.append(traffic)
    
    for _ in range(num_hotspot):
        traffic = random.randint(hotspot_min, hotspot_max)
        traffic_demands.append(traffic)
    
    random.shuffle(traffic_demands)
    return traffic_demands

# 生成流量矩阵
def generate_traffic_matrix(traffic_distribution='uniform'):
    # 加载网络拓扑
    adj_list = load_graph()
    num_nodes = len(adj_list)
    #200_2 2-80 12
    #500_2 2-80 12
    #500_1 2-10 80
    #200_1 2-10 80
    # 可控参数
    generate_all_pairs = False  # 是否为每个节点对生成路径
    min_paths_per_pair = 2  # 每个源-目的节点对的最小路径数
    max_paths_per_pair = 10  # 每个源-目的节点对的最大路径数
    min_path_length = 2    # 最小路径长度
    max_path_length = 100   # 最大路径长度
    path_algorithms = ['random']  # 可用的路径生成算法（包含随机性）
    max_dests_per_src = 80 # 每个源节点选择的最大目的节点数
    
    # 记录哪些节点对之间存在路径
    has_path = set()
    
    with open(output_file, 'w') as f:
        # 1. 生成路径信息（模仿Abi的格式）
        paths = []
        # 已生成的路径集合，避免重复
        generated_paths = set()
        # 已覆盖的节点集合，确保连通性
        covered_nodes = set()
        
        print(f"Generating paths with configuration:")
        print(f"- Generate all node pairs: {generate_all_pairs}")
        if generate_all_pairs:
            print(f"- Min paths per node pair: {min_paths_per_pair}")
            print(f"- Max paths per node pair: {max_paths_per_pair}")
        else:
            print(f"- Max destinations per source: {max_dests_per_src}")
        print(f"- Path length range: {min_path_length}-{max_path_length}")
        
        # 为每个源-目的节点对生成指定数量的路径
        total_pairs = 0
        processed_pairs = 0
    
        print("\nGenerating paths for selected node pairs...")
        # 先计算总节点对数量
        temp_total = 0
        for src in range(num_nodes):
            available_dsts = [dst for dst in range(num_nodes) if dst != src]
            temp_total += min(random.randint(1, max_dests_per_src), len(available_dsts))
        total_pairs = temp_total
        
        print(f"Total node pairs to process: {total_pairs}")
        
        
        print("\nGenerating paths for other nodes...")
        for src in range(num_nodes):
            
            #print(f"Processing source node: {src}/{num_nodes-1}")
            available_dsts = [dst for dst in range(num_nodes) if dst != src]
            # 随机选择max_dests_per_src个目的节点
            selected_dsts = random.sample(available_dsts, min(random.randint(1, max_dests_per_src), len(available_dsts)))
            
            for dst in selected_dsts:
                processed_pairs += 1
                # 显示进度
                if processed_pairs % 1000 == 0 or processed_pairs == total_pairs:
                    progress = (processed_pairs / total_pairs) * 100
                    print(f"Progress: {processed_pairs}/{total_pairs} ({progress:.1f}%) - Source node: {src}/{num_nodes-1}")
                
                # 为每个选择的节点对生成随机数量的路径
                paths_count = random.randint(min_paths_per_pair, max_paths_per_pair)
                #print(f"Generating {paths_count} paths for src: {src}, dst: {dst}")
                for path_idx in range(paths_count):
                    # 为每条路径选择不同的算法
                    algorithm = path_algorithms[path_idx % len(path_algorithms)]
                    
                    # 生成基于实际拓扑的路径，限制最大长度
                    path = generate_path(src, dst, adj_list, max_length=max_path_length, algorithm=algorithm)
                    
                    # 确保路径长度合理（在最小和最大之间）
                    if min_path_length <= len(path) <= max_path_length:
                        # 添加路径结束标记
                        path_with_end = path.copy()
                        path_with_end.append(100)
                        
                        # 生成路径唯一标识符（使用路径内容作为唯一标识）
                        path_content_key = (src, dst, tuple(path))
                        if path_content_key not in generated_paths:
                            # 保存路径信息
                            line = f"100,{src}," + ",".join(map(str, path_with_end[1:])) + "\n"
                            paths.append(line)
                            generated_paths.add(path_content_key)
                            
                            # 标记节点对之间存在路径
                            if (src, dst) not in has_path:
                                has_path.add((src, dst))
                                has_path.add((dst, src))  # 无向图，双向都有路径
                            
                            # 标记路径上的节点为已覆盖
                            for node in path:
                                covered_nodes.add(node)
        
        # 总结路径生成情况
        print(f"\nPath generation completed:")
        print(f"- Total node pairs: {total_pairs}")
        print(f"- Total generated paths: {len(generated_paths)}")
        print(f"- Total node pairs with paths: {len(has_path) // 2}")  # 除以2因为双向都有记录
        print(f"- Covered nodes: {len(covered_nodes)}/{num_nodes}")
        
        # 写入路径总数目
        f.write(f"{len(paths)}\n")
        
        # 写入路径信息
        for line in paths:
            f.write(line)
        
        # 2. 写入 succeed 标记
        f.write("succeed\n")
        
        # 3. 生成流量矩阵数据（生成50个流量矩阵，每个占一行）
        print(f"\nGenerating 50 traffic matrices with {traffic_distribution} distribution...")
        
        # 收集所有需要生成流量的节点对
        demand_positions = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and (i, j) in has_path:
                    demand_positions.append((i, j))
        
        for tm_idx in range(50):
            # 根据分布类型生成流量需求
            if traffic_distribution == 'uniform':
                traffic_demands = generate_uniform_traffic(len(demand_positions))
            elif traffic_distribution == 'skewed':
                traffic_demands = generate_skewed_traffic(len(demand_positions))
            else:
                raise ValueError(f"Unknown traffic distribution: {traffic_distribution}")
            
            # 创建流量需求字典
            demand_dict = dict(zip(demand_positions, traffic_demands))
            
            # 模仿Abi的流量矩阵格式，整个流量矩阵写在同一行，只包含非对角线元素
            all_traffic = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        # 只有当两个节点间存在路径时，才生成流量需求
                        if (i, j) in has_path:
                            traffic = demand_dict[(i, j)]
                            all_traffic.append(f"{traffic}.0")
                        else:
                            # 如果两个节点间没有路径，流量需求为0
                            all_traffic.append("0.0")
            
            # 写入整个流量矩阵为一行
            line = ",".join(all_traffic) + "\n"
            f.write(line)
        
        print(f"Generated 50 traffic matrices with {traffic_distribution} distribution")
        
        # 输出流量统计信息
        if traffic_distribution == 'uniform':
            print(f"Traffic range: 50-300")
        elif traffic_distribution == 'skewed':
            print(f"Normal traffic range: 50-150 (90%)")
            print(f"Hotspot traffic range: 400-800 (10%)")

# 生成流量矩阵
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate traffic matrix with different distributions')
    parser.add_argument('--output_dir', type=str, default="../originInput/500", help='Output directory')
    parser.add_argument('--output_file', type=str, default="500_traffic.txt", help='Output file name')
    parser.add_argument('--graph_file', type=str, default="../originInput/500/500_1000_topo.txt", help='Graph topology file')
    parser.add_argument('--distribution', type=str, default='uniform', choices=['uniform', 'skewed'], 
                       help='Traffic distribution type: uniform or skewed (hotspot traffic)')
    
    args = parser.parse_args()
    
    # 使用时间作为随机种子
    seed = int(time.time())
    random.seed(seed)
    print(f"Random seed: {seed} (based on current time)")
    
    print("=" * 80)
    print("TRAFFIC MATRIX GENERATION")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_file}")
    print(f"Topology file: {graph_file}")
    print(f"Traffic distribution: {args.distribution}")
    print("=" * 80)
    
    generate_traffic_matrix(traffic_distribution=args.distribution)
    
    print("\nTraffic matrix generation completed!")
    
    # 统计文件大小
    file_size = os.path.getsize(output_file)
    print(f"File size: {file_size / (1024 * 1024):.2f} MB")

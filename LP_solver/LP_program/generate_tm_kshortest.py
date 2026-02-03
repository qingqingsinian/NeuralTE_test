import os
import random
import time
import argparse
import heapq
from collections import defaultdict

# 生成流量矩阵文件路径
output_dir = "../originInput/50"
output_file = os.path.join(output_dir, "50_300.txt")
graph_file = os.path.join(output_dir, "50_300_topo.txt")

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 从图文件加载网络拓扑，构建邻接表
def load_graph():
    adj_list = defaultdict(list)
    edge_weights = {}
    
    with open(graph_file, 'r') as f:
        first_line = f.readline().split()
        num_nodes = int(first_line[0])
        num_links = int(first_line[1])
        
        # 读取边信息（转换为从0开始的索引）
        for _ in range(num_links):
            line = f.readline().split()
            if len(line) >= 2:
                src = int(line[0]) - 1
                dst = int(line[1]) - 1
                weight = int(line[2]) if len(line) > 2 else 1
                
                adj_list[src].append((dst, weight))
                adj_list[dst].append((src, weight))
                edge_weights[(src, dst)] = weight
                edge_weights[(dst, src)] = weight
    
    return adj_list, num_nodes, edge_weights

# Dijkstra算法 - 找到最短路径
def dijkstra(adj_list, src, dst):
    if src == dst:
        return [src]
    
    distances = {node: float('infinity') for node in adj_list}
    distances[src] = 0
    previous = {node: None for node in adj_list}
    visited = set()
    
    priority_queue = [(0, src)]
    
    while priority_queue:
        current_dist, current = heapq.heappop(priority_queue)
        
        if current in visited:
            continue
        
        if current == dst:
            break
        
        visited.add(current)
        
        for neighbor, weight in adj_list[current]:
            if neighbor in visited:
                continue
            
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(priority_queue, (distance, neighbor))
    
    if previous[dst] is None:
        return None
    
    path = []
    current = dst
    while current is not None:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    return path

# Yen's K最短路算法
def yen_k_shortest_paths(adj_list, src, dst, K=5):
    if src == dst:
        return [[src]]
    
    A = []
    B = []
    
    shortest_path = dijkstra(adj_list, src, dst)
    if shortest_path is None:
        return []
    
    A.append(shortest_path)
    
    for k in range(1, K):
        for i in range(len(A[-1]) - 1):
            spur_node = A[-1][i]
            root_path = A[-1][:i+1]
            
            removed_edges = set()
            removed_nodes = set()
            
            for path in A:
                if len(path) > i and path[:i+1] == root_path:
                    edge = (path[i], path[i+1])
                    removed_edges.add(edge)
            
            for node in root_path[:-1]:
                removed_nodes.add(node)
            
            original_adj_list = {}
            for node in adj_list:
                original_adj_list[node] = [(n, w) for n, w in adj_list[node]]
            
            for edge in removed_edges:
                if edge[0] in adj_list:
                    adj_list[edge[0]] = [(n, w) for n, w in adj_list[edge[0]] if n != edge[1]]
            
            spur_path = dijkstra(adj_list, spur_node, dst)
            
            for node in original_adj_list:
                adj_list[node] = original_adj_list[node]
            
            if spur_path is not None:
                total_path = root_path[:-1] + spur_path
                if total_path not in B:
                    B.append(total_path)
        
        if not B:
            break
        
        B.sort(key=lambda x: len(x))
        A.append(B.pop(0))
    
    return A

# BFS算法 - 广度优先搜索
def bfs(adj_list, src, dst, max_length=20):
    if src == dst:
        return [src]
    
    visited = {src}
    queue = [(src, [src])]
    
    while queue:
        current, path = queue.pop(0)
        
        if current == dst:
            return path
        
        if len(path) >= max_length:
            continue
        
        for neighbor, _ in adj_list[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
    
    return None

# 生成路径（使用K最短路算法）
def generate_path(src, dst, adj_list, algorithm='kshortest', max_paths=5):
    if src == dst:
        return [src]
    
    if algorithm == 'kshortest':
        paths = yen_k_shortest_paths(adj_list, src, dst, K=max_paths)
        if paths:
            return random.choice(paths)
    elif algorithm == 'dijkstra':
        path = dijkstra(adj_list, src, dst)
        if path:
            return path
    elif algorithm == 'bfs':
        path = bfs(adj_list, src, dst)
        if path:
            return path
    
    return None

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

# 生成不均匀分布的流量需求（热点流量，均匀分布）
def generate_skewed_traffic_uniform(num_demands, base_min=50, base_max=150, hotspot_min=700, hotspot_max=800, hotspot_ratio=0.08):
    traffic_demands = [0] * num_demands
    
    num_hotspot = int(num_demands * hotspot_ratio)
    num_normal = num_demands - num_hotspot
    
    # 随机选择高流量节点对的位置
    all_positions = list(range(num_demands))
    hotspot_positions = random.sample(all_positions, num_hotspot)
    
    # 为高流量位置分配高流量值
    for pos in hotspot_positions:
        traffic_demands[pos] = random.randint(hotspot_min, hotspot_max)
    
    # 为其他位置分配正常流量值
    normal_positions = [pos for pos in all_positions if pos not in hotspot_positions]
    for pos in normal_positions:
        traffic_demands[pos] = random.randint(base_min, base_max)
    
    return traffic_demands

# 生成流量矩阵
def generate_traffic_matrix(traffic_distribution='uniform'):
    adj_list, num_nodes, edge_weights = load_graph()
    
    # 可控参数
    min_paths_per_pair = 2
    max_paths_per_pair = 80
    min_path_length = 2
    max_path_length = 100
    
    # 记录哪些节点对之间存在路径
    has_path = set()
    
    with open(output_file, 'w') as f:
        paths = []
        generated_paths = set()
        covered_nodes = set()
        
        print(f"Generating paths with configuration:")
        print(f"- Path generation algorithm: kshortest")
        print(f"- Min paths per node pair: {min_paths_per_pair}")
        print(f"- Max paths per node pair: {max_paths_per_pair}")
        print(f"- Path length range: {min_path_length}-{max_path_length}")
        
        # 为所有源-目的节点对生成路径
        total_pairs = num_nodes * (num_nodes - 1)
        processed_pairs = 0
        
        print(f"\nGenerating paths for all node pairs (total: {total_pairs})...")
        
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src == dst:
                    continue
                
                processed_pairs += 1
                
                if processed_pairs % 1000 == 0 or processed_pairs == total_pairs:
                    progress = (processed_pairs / total_pairs) * 100
                    print(f"Progress: {processed_pairs}/{total_pairs} ({progress:.1f}%) - Source node: {src}/{num_nodes-1}")
                
                paths_count = random.randint(min_paths_per_pair, max_paths_per_pair)
                
                # 使用kshortest算法生成指定数量的路径
                k_shortest_paths = yen_k_shortest_paths(adj_list, src, dst, K=paths_count)
                
                for path in k_shortest_paths:
                    if min_path_length <= len(path) <= max_path_length:
                        path_with_end = path.copy()
                        path_with_end.append(100)
                        
                        path_content_key = (src, dst, tuple(path))
                        if path_content_key not in generated_paths:
                            line = f"100,{src}," + ",".join(map(str, path_with_end[1:])) + "\n"
                            paths.append(line)
                            generated_paths.add(path_content_key)
                            
                            if (src, dst) not in has_path:
                                has_path.add((src, dst))
                                has_path.add((dst, src))
                            
                            for node in path:
                                covered_nodes.add(node)
        
        print(f"\nPath generation completed:")
        print(f"- Total node pairs: {total_pairs}")
        print(f"- Total generated paths: {len(generated_paths)}")
        print(f"- Total node pairs with paths: {len(has_path) // 2}")
        print(f"- Covered nodes: {len(covered_nodes)}/{num_nodes}")
        
        f.write(f"{len(paths)}\n")
        
        for line in paths:
            f.write(line)
        
        f.write("succeed\n")
        
        print(f"\nGenerating 50 traffic matrices with {traffic_distribution} distribution...")
        
        demand_positions = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and (i, j) in has_path:
                    demand_positions.append((i, j))
        
        for tm_idx in range(300):
            if traffic_distribution == 'uniform':
                traffic_demands = generate_uniform_traffic(len(demand_positions))
            elif traffic_distribution == 'skewed':
                traffic_demands = generate_skewed_traffic_uniform(len(demand_positions))
            else:
                raise ValueError(f"Unknown traffic distribution: {traffic_distribution}")
            
            demand_dict = dict(zip(demand_positions, traffic_demands))
            
            all_traffic = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        if (i, j) in has_path:
                            traffic = demand_dict[(i, j)]
                            all_traffic.append(f"{traffic}.0")
                        else:
                            all_traffic.append("0.0")
            
            line = ",".join(all_traffic) + "\n"
            f.write(line)
        
        print(f"Generated 50 traffic matrices with {traffic_distribution} distribution")
        
        if traffic_distribution == 'uniform':
            print(f"Traffic range: 50-150")
        elif traffic_distribution == 'skewed':
            print(f"Normal traffic range: 50-150 (92%)")
            print(f"Hotspot traffic range: 700-800 (8%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate traffic matrix with K-shortest paths')
    parser.add_argument('--output_dir', type=str, default="../originInput/50", help='Output directory')
    parser.add_argument('--output_file', type=str, default="50_1.txt", help='Output file name')
    parser.add_argument('--graph_file', type=str, default="../originInput/50/50_300_topo.txt", help='Graph topology file')
    parser.add_argument('--distribution', type=str, default='uniform', choices=['uniform', 'skewed'], 
                       help='Traffic distribution type: uniform or skewed (hotspot traffic)')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_file = os.path.join(output_dir, args.output_file)
    graph_file = args.graph_file
    
    seed = int(time.time())
    random.seed(seed)
    print(f"Random seed: {seed} (based on current time)")
    
    print("=" * 80)
    print("TRAFFIC MATRIX GENERATION WITH K-SHORTEST PATHS")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_file}")
    print(f"Topology file: {graph_file}")
    print(f"Traffic distribution: {args.distribution}")
    print("=" * 80)
    
    generate_traffic_matrix(traffic_distribution=args.distribution)
    
    print("\nTraffic matrix generation completed!")
    
    file_size = os.path.getsize(output_file)
    print(f"File size: {file_size / (1024 * 1024):.2f} MB")

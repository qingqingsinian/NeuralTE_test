import os
import subprocess
import time
from datetime import datetime

def run_comparison_test(graph_file, path_file, tra_file, output_dir, num_tms, test_name, csv_name):
    """运行对比测试（所有策略）"""
    print(f"\n{'='*80}")
    print(f"运行对比测试: {test_name} - 所有策略")
    print(f"{'='*80}")
    
    cmd = [
        "python", "altPathSolver_warmstart_strategies.py",
        "--graph_file", graph_file,
        "--path_file", path_file,
        "--tra_file", tra_file,
        "--output_dir", output_dir,
        "--num_tms", str(num_tms),
        "--compare",
        "--csv_name", csv_name
    ]
    
    print(f"命令: {' '.join(cmd)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=36000
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"耗时: {elapsed_time:.2f}秒")
        
        if result.returncode == 0:
            print(f"✓ 对比测试成功完成")
            print(f"CSV文件: {csv_name}")
            return True, elapsed_time
        else:
            print(f"✗ 对比测试失败，返回码: {result.returncode}")
            print(f"错误输出:\n{result.stderr}")
            return False, elapsed_time
            
    except subprocess.TimeoutExpired:
        print(f"✗ 对比测试超时（超过3600秒）")
        return False, 3600
    except Exception as e:
        print(f"✗ 对比测试异常: {str(e)}")
        return False, 0

def main():
    print("="*80)
    print("Brain网络热启动策略对比测试")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_file = "../originInput/brain/brain_161_332_topo.txt"
    output_base_dir = "../LP_output/brain/warmstart_strategies"
    
    test_configs = [
        # shortest算法
        {
            "name": "shortest3",
            "path_file": "../originInput/brain/batch_combined_fixed/brain_k_shortest_k3/brain_batch_1.txt",
            "tra_file": "../originInput/brain/batch_combined_fixed/brain_k_shortest_k3/brain_batch_1.txt",
            "num_tms": 1000
        },
        {
            "name": "shortest4",
            "path_file": "../originInput/brain/batch_combined_fixed/brain_k_shortest_k4/brain_batch_1.txt",
            "tra_file": "../originInput/brain/batch_combined_fixed/brain_k_shortest_k4/brain_batch_1.txt",
            "num_tms": 1000
        },
        {
            "name": "shortest5",
            "path_file": "../originInput/brain/batch_combined_fixed/brain_k_shortest_k5/brain_batch_1.txt",
            "tra_file": "../originInput/brain/batch_combined_fixed/brain_k_shortest_k5/brain_batch_1.txt",
            "num_tms": 1000
        },
        {
            "name": "shortest6",
            "path_file": "../originInput/brain/batch_combined_fixed/brain_k_shortest_k6/brain_batch_1.txt",
            "tra_file": "../originInput/brain/batch_combined_fixed/brain_k_shortest_k6/brain_batch_1.txt",
            "num_tms": 1000
        },
        # # disjoint算法
        # {
        #     "name": "disjoint3",
        #     "path_file": "../originInput/brain/batch_combined_fixed/brain_k_disjoint_k3/brain_batch_1.txt",
        #     "tra_file": "../originInput/brain/batch_combined_fixed/brain_k_disjoint_k3/brain_batch_1.txt",
        #     "num_tms": 1000
        # },
        # {
        #     "name": "disjoint4",
        #     "path_file": "../originInput/brain/batch_combined_fixed/brain_k_disjoint_k4/brain_batch_1.txt",
        #     "tra_file": "../originInput/brain/batch_combined_fixed/brain_k_disjoint_k4/brain_batch_1.txt",
        #     "num_tms": 1000
        # },
        # {
        #     "name": "disjoint5",
        #     "path_file": "../originInput/brain/batch_combined_fixed/brain_k_disjoint_k5/brain_batch_1.txt",
        #     "tra_file": "../originInput/brain/batch_combined_fixed/brain_k_disjoint_k5/brain_batch_1.txt",
        #     "num_tms": 1000
        # },
        # {
        #     "name": "disjoint6",
        #     "path_file": "../originInput/brain/batch_combined_fixed/brain_k_disjoint_k6/brain_batch_1.txt",
        #     "tra_file": "../originInput/brain/batch_combined_fixed/brain_k_disjoint_k6/brain_batch_1.txt",
        #     "num_tms": 1000
        # },
    ]
    
    strategies = ['cold', 'minhop', 'loadbalance', 'greedyqos', 'previous']
    
    all_results = {}
    
    for test_config in test_configs:
        test_name = test_config["name"]
        path_file = test_config["path_file"]
        tra_file = test_config["tra_file"]
        num_tms = test_config["num_tms"]
        
        print(f"\n{'#'*80}")
        print(f"测试集: {test_name}")
        print(f"路径文件: {path_file}")
        print(f"流量矩阵文件: {tra_file}")
        print(f"流量矩阵数量: {num_tms}")
        print(f"{'#'*80}")
        
        output_dir = os.path.join(output_base_dir, test_name)
        os.makedirs(output_dir, exist_ok=True)
        
        csv_name = f"strategy_comparison_{test_name}.csv"
        
        success, elapsed_time = run_comparison_test(
            graph_file, path_file, tra_file, output_dir, 
            num_tms, test_name, csv_name
        )
        
        all_results[test_name] = {
            'success': success,
            'elapsed_time': elapsed_time,
            'csv_file': csv_name
        }
    
    print(f"\n{'='*80}")
    print("测试总结")
    print(f"{'='*80}")
    
    for test_name, result in all_results.items():
        status = "✓ 成功" if result['success'] else "✗ 失败"
        print(f"\n{test_name}:")
        print(f"  状态: {status}")
        print(f"  耗时: {result['elapsed_time']:.2f}s")
        print(f"  CSV文件: {result['csv_file']}")
    
    print(f"\n{'='*80}")
    print(f"测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
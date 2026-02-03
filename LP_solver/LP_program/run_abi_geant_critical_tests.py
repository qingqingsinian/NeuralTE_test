import os
import subprocess
import time

def run_test_with_critical_paths(dataset_name, graph_file, path_file, tra_file, output_dir):
    """运行测试并生成关键约束路径"""
    print(f"\n{'='*80}")
    print(f"Running test for {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"Graph file: {graph_file}")
    print(f"Path file: {path_file}")
    print(f"Traffic file: {tra_file}")
    print(f"Output dir: {output_dir}")
    
    cmd = [
        'python', 'altPathSolver_detail.py',
        f'--graphFile={graph_file}',
        f'--pathFile={path_file}',
        f'--tmFile={tra_file}',
        f'--perfFile={output_dir}\\test_output.txt'
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, timeout=1800)
        elapsed_time = time.time() - start_time
        
        print(f"\nStatus: {'SUCCESS' if result.returncode == 0 else 'FAILED'}")
        print(f"Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
        
        return {
            'dataset': dataset_name,
            'status': 'SUCCESS' if result.returncode == 0 else 'FAILED',
            'time': elapsed_time,
            'output_dir': output_dir
        }
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"\nStatus: TIMEOUT")
        print(f"Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
        
        return {
            'dataset': dataset_name,
            'status': 'TIMEOUT',
            'time': elapsed_time,
            'output_dir': output_dir
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nStatus: ERROR - {str(e)}")
        print(f"Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
        
        return {
            'dataset': dataset_name,
            'status': f'ERROR: {str(e)}',
            'time': elapsed_time,
            'output_dir': output_dir
        }

def main():
    base_dir = "c:\\TE GNN\\test\\distributed-RL-TE-LP-solver\\LP_solver\\LP_program"
    output_base = "c:\\TE GNN\\test\\distributed-RL-TE-LP-solver\\LP_solver\\LP_output"
    
    datasets = [
        {
            'name': 'abi',
            'graph_file': f'{base_dir}\\..\\originInput\\Abi\\Abi.txt',
            'path_file': f'{base_dir}\\..\\originInput\\Abi\\batch_combined_fixed_new\\abi_k_shortest_k3\\abi_batch_1.txt',
            'tra_file': f'{base_dir}\\..\\originInput\\Abi\\batch_combined_fixed_new\\abi_k_shortest_k3\\abi_batch_1.txt',
            'output_dir': f'{output_base}\\Abi\\test'
        },
        {
            'name': 'geant',
            'graph_file': f'{base_dir}\\..\\originInput\\GEA\\GEA.txt',
            'path_file': f'{base_dir}\\..\\originInput\\GEA\\batch_combined_fixed_new\\geant_k_shortest_k3\\geant_batch_1.txt',
            'tra_file': f'{base_dir}\\..\\originInput\\GEA\\batch_combined_fixed_new\\geant_k_shortest_k3\\geant_batch_1.txt',
            'output_dir': f'{output_base}\\GEA\\test'
        }
    ]
    
    all_results = []
    
    for dataset in datasets:
        result = run_test_with_critical_paths(
            dataset['name'],
            dataset['graph_file'],
            dataset['path_file'],
            dataset['tra_file'],
            dataset['output_dir']
        )
        all_results.append(result)
    
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    for result in all_results:
        print(f"\n{result['dataset'].upper()}:")
        print(f"  Status: {result['status']}")
        print(f"  Time: {result['time']:.2f}s ({result['time']/60:.2f} minutes)")
        print(f"  Output dir: {result['output_dir']}")
    
    total_time = sum(r['time'] for r in all_results)
    print(f"\n{'='*80}")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

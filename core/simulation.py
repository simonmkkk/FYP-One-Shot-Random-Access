# 模拟核心函数
import time
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def simulate_single_access_parallel(M, N, num_samples=10000, n_jobs=1):
    """
    並行執行多次單次接入模擬（僅一個AC）
    
    Args:
        M: 設備總數
        N: RAO總數
        num_samples: 模擬樣本數
        n_jobs: 並行作業數
    
    Returns:
        tuple: (平均成功RAO數, 平均碰撞RAO數, 平均空閒RAO數)
    """
    def _single_sample(M, N):
        """單次模擬的核心邏輯"""
        choices = np.random.randint(0, N, M)
        rao_usage = np.bincount(choices, minlength=N)
        success_raos = np.sum(rao_usage == 1)
        collision_raos = np.sum(rao_usage >= 2)
        idle_raos = np.sum(rao_usage == 0)
        return success_raos, collision_raos, idle_raos
    
    if n_jobs == 1:
        # 單線程執行
        results = []
        for _ in tqdm(range(num_samples), desc=f"模擬 M={M}, N={N}", leave=False):
            results.append(_single_sample(M, N))
    else:
        # 並行執行
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_sample)(M, N)
            for _ in tqdm(range(num_samples), desc=f"模擬 M={M}, N={N}", leave=False)
        )
    
    results_array = np.array(results)
    
    # 計算平均值
    mean_success = np.mean(results_array[:, 0])
    mean_collision = np.mean(results_array[:, 1])
    mean_idle = np.mean(results_array[:, 2])
    
    return mean_success, mean_collision, mean_idle

def simulate_single_sample(M, N, I_max):
    """
    模拟一次完整的群组寻呼过程（从第1个AC到第I_max个AC）
    """
    # 初始化：所有M个设备都在第一个AC尝试接入
    remaining_devices = M
    success_delay_sum = 0
    success_count = 0
    total_collision_count = 0
    total_rao_count = 0
    
    # 遍历所有接入周期
    for ac_index in range(1, I_max+1):
        # 即使没有剩餘設備，也要統計RAO數量（與理論一致）
        current_raos = N
        total_rao_count += current_raos
        
        if remaining_devices == 0:
            # 没有剩餘設備，但繼續統計RAO
            continue
            
        # 模擬當前AC的隨機接入（內聯邏輯以保持性能）
        choices = np.random.randint(0, current_raos, remaining_devices)
        rao_usage = np.bincount(choices, minlength=current_raos)
        success_raos = np.sum(rao_usage == 1)
        collision_raos = np.sum(rao_usage >= 2)
        
        # 更新成功設備統計（成功設備在當前AC完成接入）
        success_count += success_raos
        success_delay_sum += success_raos * ac_index  # 延遲=AC索引
        
        # 更新碰撞統計
        total_collision_count += collision_raos
        
        # 剩餘設備是那些遭遇碰撞的設備，它們將在下一個AC重試
        remaining_devices = remaining_devices - success_raos
    
    # 計算本次模擬的性能指標
    access_success_prob = success_count / M if M > 0 else 0
    
    # 修改：只計算成功設備的平均延遲（與論文一致）
    if success_count > 0:
        mean_access_delay = success_delay_sum / success_count
    else:
        # 如果沒有任何成功設備，返回特殊值表示無效樣本
        mean_access_delay = -1
    
    # 碰撞概率：論文定義為所有I_max個AC的碰撞RAO與總RAO之比
    total_theoretical_rao = I_max * N
    collision_prob = total_collision_count / total_theoretical_rao if total_theoretical_rao > 0 else 0
    
    return access_success_prob, mean_access_delay, collision_prob


def run_parallel_simulation(M, N, I_max, num_samples, num_workers, batch_size=None):
    """
    并行运行多次模拟（直接处理单个样本）
    """
    print(f"开始并行模拟 ({num_samples} 样本, {num_workers} 进程)...")
    start_time = time.time()
    
    # 直接并行处理每个样本
    results = Parallel(n_jobs=num_workers)(
        delayed(simulate_single_sample)(M, N, I_max)
        for _ in tqdm(range(num_samples), desc="模拟进度", unit="样本")
    )
    
    # 将结果转换为numpy数组
    results_array = np.array(results)
    
    elapsed_time = time.time() - start_time
    print(f"模拟完成! 耗时: {elapsed_time:.2f}秒, 平均速度: {num_samples/elapsed_time:.0f} 样本/秒")
    
    return results_array
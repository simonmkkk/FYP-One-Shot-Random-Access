# 模拟核心函数 (重構版)
import time
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


# ============================================================================
# 第一層：核心單次模擬函數（最基本的原子操作）
# ============================================================================

def simulate_one_shot_access_single_sample(M, N):
    """
    【核心原子函數】模擬一次 One-Shot Random Access（單個 AC）
    
    這是最基本的模擬單元，執行一次隨機接入過程：
    1. M 個設備隨機選擇 N 個 RAO 中的一個
    2. 統計成功、碰撞、空閒的 RAO 數量
    
    Args:
        M (int): 嘗試接入的設備數量
        N (int): 可用的 RAO（隨機接入機會）數量
    
    Returns:
        tuple: (success_raos, collision_raos, idle_raos)
            - success_raos (int): 成功的 RAO 數量（恰好 1 個設備選擇）
            - collision_raos (int): 碰撞的 RAO 數量（≥2 個設備選擇）
            - idle_raos (int): 空閒的 RAO 數量（0 個設備選擇）
    
    對應論文:
        - Section II-A: One-Shot Random Access 定義
        - 用於驗證公式 (1)-(5) 的模擬基礎
    
    範例:
        >>> simulate_one_shot_access_single_sample(M=10, N=5)
        (3, 2, 0)  # 3個成功, 2個碰撞, 0個空閒
    """
    # 步驟1: 每個設備隨機選擇一個 RAO（0 到 N-1）
    # choices[i] 表示第 i 個設備選擇的 RAO 編號
    choices = np.random.randint(0, N, M)
    
    # 步驟2: 統計每個 RAO 被選擇的次數
    # rao_usage[j] 表示第 j 個 RAO 被多少個設備選中
    # minlength=N 確保即使某些 RAO 沒被選中也會在數組中
    rao_usage = np.bincount(choices, minlength=N)
    
    # 步驟3: 根據論文定義判斷成功/碰撞/空閒
    success_raos = np.sum(rao_usage == 1)    # 恰好 1 個設備 → 成功
    collision_raos = np.sum(rao_usage >= 2)  # ≥2 個設備 → 碰撞
    idle_raos = np.sum(rao_usage == 0)       # 0 個設備 → 空閒
    
    return success_raos, collision_raos, idle_raos


# ============================================================================
# 第二層：Figure 1 & 2 專用函數（單個 AC 的多樣本並行模擬）
# ============================================================================

def simulate_one_shot_access_multi_samples(M, N, num_samples, num_workers):
    
    """
    【Figure 1 & 2 專用】並行執行多次單個 AC 的模擬，計算統計平均值
    
    用途：
        - 驗證論文公式 (1)-(3) 的精確值
        - 驗證論文公式 (4)-(5) 的近似公式
        - 生成 Figure 1 和 Figure 2 的模擬數據點
    
    Args:
        M (int): 設備總數
        N (int): RAO 總數
        num_samples (int): 模擬樣本數（預設 10,000）
        num_workers (int): 並行工作進程數（預設 1 = 單線程）
    
    Returns:
        tuple: (mean_success, mean_collision, mean_idle)
            - mean_success (float): 平均成功 RAO 數量（對應 NS,1）
            - mean_collision (float): 平均碰撞 RAO 數量（對應 NC,1）
            - mean_idle (float): 平均空閒 RAO 數量
    
    對應論文:
        - Figure 1: NS,1/N 和 NC,1/N 與 M/N 的關係（N=3 和 N=14）
        - Figure 2: 近似誤差分析
        - 用於驗證公式 (2), (3), (4), (5)
    
    範例:
        >>> mean_s, mean_c, mean_i = simulate_one_shot_access_multi_samples(
        ...     M=100, N=50, num_samples=100000, num_workers=8
        ... )
        >>> print(f"平均成功 RAO: {mean_s:.2f}")
        平均成功 RAO: 27.06
    """
    print(f"[Figure 1&2] 開始單個 AC 多樣本模擬...")
    print(f"  參數: M={M}, N={N}, 樣本數={num_samples}, 工作進程={num_workers}")
    start_time = time.time()
    
    # 使用 joblib 並行處理
    if num_workers == 1:
        # 單線程模式（用於調試或小規模測試）
        results = [
            simulate_one_shot_access_single_sample(M, N)
            for _ in tqdm(range(num_samples), desc="單線程模擬", unit="樣本")
        ]
    else:
        # 多線程模式（生產環境）
        results = Parallel(n_jobs=num_workers)(
            delayed(simulate_one_shot_access_single_sample)(M, N)
            for _ in tqdm(range(num_samples), desc="並行模擬", unit="樣本")
        )
    
    # 將結果轉換為 NumPy 數組並計算平均值
    results_array = np.array(results)  # Shape: [num_samples, 3]
    mean_success = np.mean(results_array[:, 0])
    mean_collision = np.mean(results_array[:, 1])
    mean_idle = np.mean(results_array[:, 2])
    
    elapsed_time = time.time() - start_time
    print(f"  完成! 耗時: {elapsed_time:.2f}秒")
    print(f"  結果: NS,1={mean_success:.4f}, NC,1={mean_collision:.4f}, 空閒={mean_idle:.4f}")
    
    return mean_success, mean_collision, mean_idle


# ============================================================================
# 第三層：Main.py 專用函數（完整群組尋呼的多樣本並行模擬）
# ============================================================================

def simulate_group_paging_multi_samples(M, N, I_max, num_samples, num_workers):
    """
    【Main.py 專用】並行執行完整群組尋呼過程的多樣本模擬
    
    用途：
        - 模擬從第 1 個 AC 到第 I_max 個 AC 的完整過程
        - 計算論文的三個核心性能指標：PS, Ta, PC
        - 生成 Figure 3, 4, 5 的模擬數據
    
    工作流程：
        1. 並行執行 num_samples 次完整的群組尋呼模擬
        2. 每次模擬包含 I_max 個 AC 的遞推過程
        3. 統計每次模擬的 PS, Ta, PC
        4. 返回所有樣本的結果矩陣
    
    Args:
        M (int): 初始設備總數（K1 = M）
        N (int): 每個 AC 的 RAO 數量（假設 Ni = N 為常數）
        I_max (int): 最大接入周期數（重傳次數限制）
        num_samples (int): 模擬樣本數（論文使用 10^7，預設 10^6）
        num_workers (int): 並行工作進程數（建議設為 CPU 核心數）
    
    Returns:
        np.ndarray: Shape [num_samples, 3] 的結果矩陣
            - 每一行: [access_success_prob, mean_access_delay, collision_prob]
            - access_success_prob: 接入成功率 PS（公式 8）
            - mean_access_delay: 平均接入延遲 Ta（公式 9）
            - collision_prob: 碰撞概率 PC（公式 10）
    
    對應論文:
        - Figure 3: Access Success Probability vs N
        - Figure 4: Mean Access Delay vs N
        - Figure 5: Collision Probability vs N
        - Section III: Numerical Results (10^7 samples)
    
    範例:
        >>> results = simulate_group_paging_multi_samples(
        ...     M=100, N=40, I_max=10, num_samples=1000000, num_workers=16
        ... )
        >>> mean_PS = np.mean(results[:, 0])
        >>> print(f"平均接入成功率: {mean_PS:.6f}")
        平均接入成功率: 0.965432
    """
    print("=" * 70)
    print("【Main 模擬】完整群組尋呼多樣本並行模擬")
    print("=" * 70)
    print(f"  參數配置:")
    print(f"    - 設備數 M = {M}")
    print(f"    - RAO 數 N = {N}")
    print(f"    - 最大 AC 數 I_max = {I_max}")
    print(f"    - 模擬樣本數 = {num_samples:,}")
    print(f"    - 並行工作進程 = {num_workers}")
    print(f"  預估負載: M/N = {M/N:.2f}")
    print("=" * 70)
    
    start_time = time.time()
    
    # 並行執行多次完整模擬
    # 每個樣本獨立執行一次完整的群組尋呼過程（I_max 個 AC）
    results = Parallel(n_jobs=num_workers)(
        delayed(simulate_group_paging_single_sample)(M, N, I_max)
        for _ in tqdm(range(num_samples), desc="模擬進度", unit="樣本")
    )
    
    # 將結果列表轉換為 NumPy 數組
    results_array = np.array(results)
    
    elapsed_time = time.time() - start_time
    samples_per_sec = num_samples / elapsed_time
    
    print("=" * 70)
    print(f"  模擬完成!")
    print(f"    - 總耗時: {elapsed_time:.2f} 秒")
    print(f"    - 平均速度: {samples_per_sec:.0f} 樣本/秒")
    print(f"    - 結果矩陣形狀: {results_array.shape}")
    print("=" * 70)
    
    return results_array


def simulate_group_paging_single_sample(M, N, I_max):
    """
    【內部函數】模擬一次完整的群組尋呼過程（從第 1 個 AC 到第 I_max 個 AC）
    
    這是 simulate_group_paging_multi_samples 的工作單元，
    被並行調用 num_samples 次。
    
    工作流程（對應論文 Section II-B）：
        初始狀態: K1 = M (所有設備在第 1 個 AC 開始競爭)
        
        對於每個 AC (i = 1 to I_max):
            1. Ki 個設備執行 one-shot random access
            2. 統計成功和碰撞的 RAO 數量
            3. 更新性能指標累加器
            4. 計算下一個 AC 的競爭設備數: Ki+1 = Ki - NS,i （公式 7）
        
        最後計算:
            - PS = (總成功設備數) / M （公式 8）
            - Ta = (成功設備的延遲總和) / (總成功設備數) （公式 9）
            - PC = (總碰撞 RAO 數) / (I_max × N) （公式 10）
    
    Args:
        M (int): 初始設備數
        N (int): 每個 AC 的 RAO 數
        I_max (int): 最大 AC 數
    
    Returns:
        tuple: (access_success_prob, mean_access_delay, collision_prob)
    """
    # ========== 初始化統計變量 ==========
    remaining_devices = M           # 當前競爭設備數（對應論文的 Ki）
    success_count = 0               # 累計成功設備總數
    success_delay_sum = 0           # 成功設備的延遲總和（用於計算 Ta）
    total_collision_count = 0       # 累計碰撞 RAO 總數
    
    # ========== 遍歷所有接入周期 ==========
    for ac_index in range(1, I_max + 1):
        # 如果沒有剩餘設備，直接跳過（但繼續迭代以保持 I_max 完整）
        if remaining_devices == 0:
            continue
        
        # ===== 步驟1: 執行當前 AC 的 one-shot random access =====
        # 調用核心函數模擬 remaining_devices 個設備競爭 N 個 RAO
        success_raos, collision_raos, _ = simulate_one_shot_access_single_sample(
            remaining_devices, N
        )
        
        # ===== 步驟2: 更新成功設備統計 =====
        # 成功的 RAO 數量 = 成功接入的設備數量（一對一對應）
        success_count += success_raos
        
        # 累加延遲: 在第 i 個 AC 成功的設備，延遲為 i
        # 對應論文公式 (9) 的分子: Σ i × NS,i
        success_delay_sum += success_raos * ac_index
        
        # ===== 步驟3: 更新碰撞統計 =====
        total_collision_count += collision_raos
        
        # ===== 步驟4: 更新下一個 AC 的競爭設備數 =====
        # 對應論文公式 (7): Ki+1 = Ki - NS,i
        # 成功設備退出競爭，碰撞設備繼續重試
        remaining_devices = remaining_devices - success_raos
    
    # ========== 計算性能指標 ==========
    
    # 公式 (8): 接入成功率 PS = Σ NS,i / M
    access_success_prob = success_count / M if M > 0 else 0.0
    
    # 公式 (9): 平均接入延遲 Ta = Σ(i × NS,i) / Σ NS,i
    # 注意: 只計算成功設備的平均延遲
    if success_count > 0:
        mean_access_delay = success_delay_sum / success_count
    else:
        # 如果沒有任何設備成功，返回特殊值 -1
        mean_access_delay = -1.0
    
    # 公式 (10): 碰撞概率 PC = Σ NC,i / Σ Ni
    # 分母是所有 AC 的總 RAO 數: I_max × N
    total_rao_count = I_max * N
    collision_prob = total_collision_count / total_rao_count if total_rao_count > 0 else 0.0
    
    return access_success_prob, mean_access_delay, collision_prob




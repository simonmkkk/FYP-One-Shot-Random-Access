# ============================================================================
# 【formulas.py - 數學公式模塊】
# ============================================================================
# 論文中的所有數學公式實現
# 基於論文: "Modeling and Estimation of Single Random Access for Finite-User Multi-Channel Slotted ALOHA Systems"
#
# 📊 模塊架構 (6層結構)：
# ├── 【輔助工具函數】
# │   └── generate_partitions()           # 整數分割生成器
# │   └── _compute_configuration_ways()   # 共用配置計算邏輯
# │
# ├── 【精確公式 (1-3)】- 單次隨機接入分析
# │   ├── paper_formula_1_pk_probability()        # 公式(1): k個碰撞RAO的概率
# │   ├── paper_formula_2_collision_raos_exact()  # 公式(2): 期望碰撞RAO數
# │   └── paper_formula_3_success_raos_exact()    # 公式(3): 期望成功RAO數
# │
# ├── 【近似公式 (4-5)】- 快速計算
# │   ├── paper_formula_4_success_approx()        # 公式(4): NS,1 ≈ M·e^(-M/N)
# │   └── paper_formula_5_collision_approx()      # 公式(5): NC,1 ≈ N·(1 - e^(-M/N)(1 + M/N))
# │
# ├── 【迭代公式 (6-7)】- 多個AC循環
# │   ├── paper_formula_6_success_per_cycle()      # 公式(6): 第i個AC的成功設備
# │   └── paper_formula_7_next_contending_devices() # 公式(7): 下一個AC的競爭設備
# │
# └── 【性能指標公式 (8-10) 與工具】
#     ├── paper_formula_8_access_success_probability()  # 公式(8): 接入成功率
#     ├── paper_formula_9_mean_access_delay()           # 公式(9): 平均接入延遲
#     ├── paper_formula_10_collision_probability()      # 公式(10): 碰撞概率
#     ├── calculate_idle_raos()                         # 計算空閒RAO數
#     ├── relative_error_percentage()                   # 相對誤差計算
#     └── confidence_interval_95()                      # 95%置信區間
#
# 🔗 計算複雜度對比：
# ┌─────────┬──────────┬──────────────────┬─────────────────────┐
# │ 公式    │ 類型     │ 時間複雜度       │ 使用場景            │
# ├─────────┼──────────┼──────────────────┼─────────────────────┤
# │ 1-3     │ 精確     │ O(M²·N²)        │ Figure驗證（小規模）│
# │ 4-5     │ 近似     │ O(1)            │ 參數掃描（大規模）  │
# │ 6-7     │ 迭代     │ O(I_max)        │ 多AC計算            │
# │ 8-10    │ 聚合     │ O(I_max)        │ 最終指標            │
# └─────────┴──────────┴──────────────────┴─────────────────────┘
#
# 📝 參數命名約定：
#   M     # 設備總數
#   N, N1 # RAO總數（N1用於公式1-5，N用於公式6-10）
#   K_i   # 第i個AC的競爭設備數
#   NS,i  # 第i個AC的成功設備數
#   NC,i  # 第i個AC的碰撞RAO數
#
# 🔄 模塊間依賴：
#   formulas.py (當前)
#     ↓ 被使用
#   theoretical.py        # 調用公式6-10計算理論值
#     ↓ 被使用
#   main.py              # 主程序
#
# ============================================================================

import numpy as np
from math import factorial, comb
from functools import lru_cache

# ============================================================================
# 【第1層】輔助工具函數
# ============================================================================

def generate_partitions(n: int, k: int, min_val: int = 2):
    """
    【輔助工具】生成所有满足条件的整數分割
    
    用途：在公式(1)-(3)的精確計算中，用於列舉所有可能的碰撞配置方式
    
    功能：生成滿足 i1+i2+...+ik = n, 每個ij >= min_val 的所有分割
    
    參數：
        n (int): 要分割的總數（通常是碰撞RAO中的總用戶數）
        k (int): 分割成k份（通常是碰撞RAO數）
        min_val (int): 每份的最小值，預設為2（因為碰撞至少2個用戶）
    
    返回：
        generator: 生成每種分割方式，如 [2, 3, 5] 表示分成3堆，分別有2、3、5個元素
    
    複雜度：O(n^k)，但使用生成器避免一次性生成所有結果
    
    範例：
        >>> list(generate_partitions(6, 2, 2))  # 將6分成2份，每份≥2
        [[2, 4], [3, 3], [4, 2]]
    
    應用：在 _compute_configuration_ways() 中被調用
    """
    if k == 0:
        if n == 0:
            yield []
        return
    if k == 1:
        if n >= min_val:
            yield [n]
        return
    
    for first in range(min_val, n - min_val * (k - 1) + 1):
        for rest in generate_partitions(n - first, k - 1, min_val):
            yield [first] + rest

def _compute_configuration_ways(M: int, N1: int, k: int, total_in_collision: int, remaining_users: int) -> int:
    """
    【輔助工具】計算給定碰撞配置的方式數（被公式1-3共享調用）
    
    核心邏輯：
        給定某個碰撞配置（k個碰撞RAO，總共有total_in_collision個用戶在其中），
        計算有多少種不同的方式可以達成這個配置
    
    計算包括：
        1. 將 total_in_collision 個用戶分配到 k 個碰撞RAO（每個≥2個用戶）
        2. 將 remaining_users 個用戶分配到 N1-k 個非碰撞RAO（每個≤1個用戶）
        3. 選擇哪 k 個RAO作為碰撞RAO
    
    參數：
        M (int): 設備總數
        N1 (int): RAO總數
        k (int): 碰撞RAO數量
        total_in_collision (int): 碰撞RAO中的總用戶數
        remaining_users (int): 非碰撞RAO中的用戶數（remaining_users + total_in_collision = M）
    
    返回：
        int: 該配置方式的總數
    
    計算步驟：
        ├─ 若 k=0：特殊情況，所有M個用戶分配到N1個RAO（每個最多1個）
        │         方式數 = C(N1, M) × M!
        └─ 若 k>0：
            ├─ 枚舉所有分割方式：將total_in_collision分成k份（每份≥2）
            ├─ 對每種分割：
            │  ├─ 計算分配到碰撞RAO的方式數
            │  ├─ 計算分配到非碰撞RAO的方式數
            │  ├─ 計算選擇哪k個RAO作為碰撞RAO的方式數
            │  └─ 三者相乘
            └─ 對所有分割方式求和
    
    應用：在 paper_formula_1_pk_probability() 和 paper_formula_3_success_raos_exact() 中被調用
    """
    ways_this_config = 0
    
    if k == 0:
        # 沒有碰撞RAO的特殊情況
        if total_in_collision == 0 and remaining_users <= N1:
            ways_non_collision = comb(N1, remaining_users) * factorial(remaining_users)
            ways_this_config = ways_non_collision
    else:
        # 列舉所有可能的分區方式
        partitions = generate_partitions(total_in_collision, k, 2)
        for partition in partitions:
            # 計算將用戶分配到碰撞RAO的方式數
            ways_collision = 1
            remaining = M
            for count in partition:
                ways_collision *= comb(remaining, count)
                remaining -= count
            
            # 計算將剩餘用戶分配到非碰撞RAO的方式數
            ways_non_collision = comb(N1 - k, remaining_users) * factorial(remaining_users)
            
            # 計算特定分配的方式數
            ways_specific_assignment = ways_collision * ways_non_collision
            
            # 選擇哪k個RAO作為碰撞RAO的方式數
            ways_choose_collision_raos = comb(N1, k)
            
            ways_this_config += ways_specific_assignment * ways_choose_collision_raos
    
    return ways_this_config


# ============================================================================
# 【第2層】精確公式 (1-3) - 單次隨機接入分析
# ============================================================================
# 特點：完全精確，無任何近似
#      使用多重求和精確計算
#      計算複雜度高 O(M²·N²)，但可使用LRU快取優化
# 應用：驗證Figure 1&2、與近似公式對比誤差分析

def paper_formula_1_pk_probability(M: int, N1: int, k: int) -> float:
    """
    【精確公式1】pk(M, N1) - k個碰撞RAO的概率
    
    數學定義：
        pk(M, N1) = (恰好有k個碰撞RAO的配置方式數) / N1^M
    
    含義：
        當M個設備隨機選擇N1個RAO時，恰好有k個RAO被≥2個設備選中的概率
    
    計算流程：
        1. 遍歷所有可能的碰撞RAO中的用戶數分佈（2k到M）
        2. 對每種分佈，調用 _compute_configuration_ways() 計算方式數
        3. 求和後除以N1^M得到概率
    
    參數：
        M (int): 嘗試接入的設備數量
        N1 (int): 可用的RAO數量
        k (int): 碰撞RAO數量（0 ≤ k ≤ min(N1, ⌊M/2⌋)）
    
    返回：
        float: 概率值 [0, 1]
    
    特點：
        ✅ 完全精確，無任何近似
        🔒 使用 @lru_cache 避免重複計算
        ⏱️ 計算較慢（多重循環），但適合小規模參數
    
    應用：
        - 驗證論文公式(2)和(3)
        - Figure 1&2的理論基準
        - 用於校驗近似公式的準確度
    
    論文對應：Section II-B, Equation (1)
    """
    # 使用緩存版本
    return _paper_formula_1_pk_probability_impl(M, N1, k)


@lru_cache(maxsize=10000)
def _paper_formula_1_pk_probability_impl(M: int, N1: int, k: int) -> float:
    """
    【實現層】公式(1)的實際計算函數（帶LRU快取）
    
    使用此封裝的目的：
        - 将缓存逻辑与主函数分离
        - paper_formula_1_pk_probability() 作为用户接口
        - _paper_formula_1_pk_probability_impl() 作为实际计算（可缓存）
    
    快取策略：
        - maxsize=10000：最多缓存10000個計算結果
        - 參數(M, N1, k)相同時，直接返回快取結果
        - 避免參數掃描時重複計算
    """
    if k < 0 or k > min(N1, M // 2):
        return 0.0
    
    total_ways = N1 ** M
    
    # 计算满足条件的方法数
    valid_ways = 0
    
    # 遍历碰撞RAO中的总用户数（从2k到M）
    for total_in_collision in range(2 * k, M + 1):
        # 剩余的用户数分配到非碰撞RAO
        remaining_users = M - total_in_collision
        
        # 非碰撞RAO数量为 N1 - k
        # 每个非碰撞RAO最多1个用户，所以 remaining_users <= N1 - k
        if remaining_users > N1 - k:
            continue
        
        # 使用共用函數計算配置方式數
        valid_ways += _compute_configuration_ways(M, N1, k, total_in_collision, remaining_users)
    
    pk = valid_ways / total_ways if total_ways > 0 else 0.0
    return pk

def paper_formula_2_collision_raos_exact(M: int, N1: int) -> float:
    """
    【精確公式2】NC,1 - 期望碰撞RAO數（單個AC）
    
    數學定義：
        NC,1 = Σ(k=1 to min{N1,⌊M/2⌋}) k × pk(M, N1)
    
    含義：
        期望碰撞RAO數 = 所有可能碰撞RAO數量的加權平均
        其中權重是該配置的概率
    
    計算流程：
        1. 對所有可能的 k 值（1到min(N1, ⌊M/2⌋)）
        2. 調用 paper_formula_1_pk_probability(M, N1, k) 計算概率
        3. 將 k × pk 求和
    
    參數：
        M (int): 設備總數
        N1 (int): RAO總數
    
    返回：
        float: 期望的碰撞RAO數（可能為小數）
    
    特點：
        ✅ 完全精確計算
        📊 結果用於Figure 1對比
        🔄 依賴於公式(1)的計算
    
    應用：
        - Figure 1: 分析模型 vs 近似公式的對比
        - 驗證近似公式(5)的準確度
    
    論文對應：Section II-B, Equation (2)
    """
    if M <= 1 or N1 == 0:
        return 0.0
    
    NC_1 = 0.0
    max_k = min(N1, M // 2)
    
    for k in range(1, max_k + 1):
        pk_val = paper_formula_1_pk_probability(M, N1, k)
        NC_1 += k * pk_val
    
    return NC_1

def paper_formula_3_success_raos_exact(M: int, N1: int) -> float:
    """
    【精確公式3】NS,1 - 期望成功RAO數（單個AC）
    
    數學定義：
        NS,1 = Σ(k=0 to ⌊M/2⌋) E[成功RAO數 | k個碰撞RAO] × pk(M, N1)
    
    含義：
        期望成功RAO數 = 在所有可能的碰撞配置下，
        成功RAO數的加權平均（以該配置的概率為權重）
    
    計算流程：
        1. 對所有可能的k值（0到⌊M/2⌋）
        2. 調用 paper_formula_1_pk_probability() 計算 pk
        3. 對每個k，計算該配置下的期望成功RAO數
           - 遍歷所有可能的碰撞RAO中的用戶分佈
           - 計算每種分佈對應的成功用戶數
           - 加權平均
        4. 對所有k的結果加權求和
    
    參數：
        M (int): 設備總數
        N1 (int): RAO總數
    
    返回：
        float: 期望的成功RAO數
    
    特點：
        ✅ 完全精確計算
        📊 用作Figure 1對比基準
        🔄 依賴於公式(1)和配置計算
        ⏱️ 複雜度最高，計算時間較長
    
    應用：
        - Figure 1: 精確值 vs Poisson近似對比
        - 驗證近似公式(4)的準確度
        - 建立理論基準
    
    論文對應：Section II-B, Equation (3)
    """
    if M == 0 or N1 == 0:
        return 0.0
    
    NS_1 = 0.0
    max_k = min(N1, M // 2)
    
    # 为每个k值计算概率和条件期望
    for k in range(0, max_k + 1):
        pk_val = paper_formula_1_pk_probability(M, N1, k)
        
        if pk_val == 0:
            continue
        
        # 对于每个k，计算期望的成功用户数
        expected_success_given_k = 0.0
        
        # 遍历碰撞RAO中的总用户数
        for total_in_collision in range(2 * k if k > 0 else 0, M + 1):
            remaining_users = M - total_in_collision
            
            if remaining_users > N1 - k:
                continue
            
            # 使用共用函數計算配置方式數
            ways_this_config = _compute_configuration_ways(M, N1, k, total_in_collision, remaining_users)
            
            total_ways = N1 ** M
            prob_this_config = ways_this_config / total_ways if total_ways > 0 else 0
            
            if prob_this_config > 0:
                # 在这个配置下，成功用户数就是remaining_users
                expected_success_given_k += remaining_users * (prob_this_config / pk_val)
        
        NS_1 += expected_success_given_k * pk_val
    
    return NS_1


# ============================================================================
# 【第3層】近似公式 (4-5) - 快速計算版本
# ============================================================================
# 特點：基於Poisson近似，O(1)時間複雜度
#      計算速度極快，適合大規模參數掃描
#      誤差隨M/N增大而增大
# 應用：Figure 1&2 誤差分析、參數掃描
# 與精確值對比：誤差分析見Figure 2

def paper_formula_4_success_approx(M, N1):
    """
    【近似公式4】NS,1 ≈ M·e^(-M/N1) - 成功RAO近似公式
    
    數學表達式：
        NS,1 ≈ M·e^(-M/N1)
    
    推導背景：
        - 基於Poisson過程近似
        - 假設M個設備的選擇近似為Poisson分佈
        - 每個RAO被恰好1個設備選中的概率 ≈ (M/N1)·e^(-M/N1)
        - 期望成功RAO數 = N1 × (M/N1)·e^(-M/N1) = M·e^(-M/N1)
    
    參數：
        M (int): 設備總數
        N1 (int): RAO總數
    
    返回：
        float: 成功RAO的近似數量
    
    特點：
        ⚡ O(1)時間複雜度，計算極快
        📈 適合大規模參數掃描
        📉 誤差隨M/N增大而增大
        🎯 當M/N₁ < 1時誤差<5%
    
    誤差分析：見Figure 2
        - 小M/N₁：誤差小（<3%）
        - 中M/N₁：誤差中等（5-10%）
        - 大M/N₁：誤差較大（>15%）
    
    應用：
        - Figure 1: Poisson近似曲線
        - Figure 2: 誤差分析對比
        - 參數掃描（快速計算）
        - 理論分析（當精確值計算困難時）
    
    論文對應：Section II-C, Equation (4), Poisson近似
    """
    return M * np.exp(-M / N1)

def paper_formula_5_collision_approx(M, N1):
    """
    【近似公式5】NC,1 ≈ N1·(1 - e^(-M/N1)·(1 + M/N1)) - 碰撞RAO近似公式
    
    數學表達式：
        NC,1 ≈ N1·(1 - e^(-M/N1)·(1 + M/N1))
    
    推導背景：
        - 基於Poisson過程近似
        - 每個RAO為成功(1個設備)的概率 ≈ (M/N1)·e^(-M/N1)
        - 每個RAO為空閒(0個設備)的概率 ≈ e^(-M/N1)
        - 每個RAO為碰撞(≥2個設備)的概率 ≈ 1 - e^(-M/N1) - (M/N1)·e^(-M/N1)
        - 期望碰撞RAO數 = N1 × [1 - e^(-M/N1)·(1 + M/N1)]
    
    參數：
        M (int): 設備總數
        N1 (int): RAO總數
    
    返回：
        float: 碰撞RAO的近似數量
    
    特點：
        ⚡ O(1)時間複雜度
        📈 適合大規模參數掃描
        📊 與公式(4)互補（成功+碰撞+空閒=N1）
        🎯 當M/N₁ < 1時誤差<5%
    
    應用：
        - Figure 1: Poisson近似曲線
        - Figure 2: 誤差分析對比
        - 參數掃描（快速計算）
    
    論文對應：Section II-C, Equation (5), Poisson近似修正版
    """
    exp_term = np.exp(-M / N1)
    return N1 * (1 - exp_term * (1 + M/N1))


# ============================================================================
# 【第4層】迭代公式 (6-7) - 多個AC循環
# ============================================================================
# 特點：用於計算多個接入周期(AC)的迭代過程
#      從第1個AC遞推到第I_max個AC
#      體現了群組尋呼的完整過程
# 應用：Figure 3-5、multi-AC性能評估

def paper_formula_6_success_per_cycle(K_i, N_i):
    """
    【迭代公式6】NS,i = Ki·e^(-Ki/Ni) - 第i個AC的成功設備數
    
    數學表達式：
        NS,i = Ki·e^(-Ki/Ni)
    
    含義：
        在第i個AC中，有Ki個設備競爭N_i個RAO時，
        期望有多少個設備成功（在某個RAO上獨占）
    
    參數：
        K_i (float): 第i個AC的競爭設備數（嘗試接入的設備）
        N_i (float): 第i個AC的RAO數（通常固定為N）
    
    返回：
        float: 成功的設備數
    
    公式推導：
        - 基於Poisson近似
        - 每個設備選擇特定RAO的概率 = 1/Ni
        - 某個RAO有恰好1個設備的概率 ≈ (Ki/Ni)·e^(-Ki/Ni)
        - 所有RAO的期望成功RAO數 = Ni × (Ki/Ni)·e^(-Ki/Ni) = Ki·e^(-Ki/Ni)
    
    特點：
        📊 基礎迭代函數
        🔄 被 theoretical.py 調用計算每個AC
        ⚡ O(1)時間複雜度
    
    應用：
        - 計算群組尋呼的每一輪結果
        - theoretical_calculation() 中用於循環計算
        - Figure 3-5 數據計算
    
    論文對應：Section III-A, Equation (6)
    """
    return K_i * np.exp(-K_i / N_i)

def paper_formula_7_next_contending_devices(K_i, N_i):
    """
    【迭代公式7】Ki+1 = Ki·(1 - e^(-Ki/Ni)) - 下一個AC的競爭設備數
    
    數學表達式：
        Ki+1 = Ki·(1 - e^(-Ki/Ni))
    
    含義：
        在第i個AC中，有Ki個設備進行隨機接入。
        有NS,i個設備成功，剩下的 Ki - NS,i 個設備失敗。
        這些失敗的設備會在第i+1個AC重新嘗試。
    
    數學關係：
        Ki+1 = Ki - NS,i = Ki - Ki·e^(-Ki/Ni) = Ki·(1 - e^(-Ki/Ni))
    
    參數：
        K_i (float): 當前AC的競爭設備數
        N_i (float): 當前AC的RAO數
    
    返回：
        float: 下一個AC的競爭設備數
    
    特點：
        🔄 遞推關係，銜接相鄰兩個AC
        📉 Ki+1 < Ki（設備數單調遞減）
        ⏹️ 當Ki很小時，Ki+1近似為0，模擬終止
        ⚡ O(1)時間複雜度
    
    物理意義：
        - 第i個AC中成功的設備離開系統
        - 失敗的設備延遲到第i+1個AC重試
        - 直到所有設備成功或達到最大AC數I_max
    
    應用：
        - theoretical.py 中用於迭代計算
        - 多AC模擬（Figure 3-5）
        - 群組尋呼完整過程模擬
    
    論文對應：Section III-A, Equation (7)
    """
    return K_i * (1 - np.exp(-K_i / N_i))


# ============================================================================
# 【第5層】性能指標公式 (8-10)
# ============================================================================
# 特點：最終性能評估指標，基於多AC的累計結果
#      綜合評估系統性能
# 應用：Figure 3-5、系統性能評估

def paper_formula_8_access_success_probability(N_s_list, M):
    """
    【性能指標公式8】PS = Σ(i=1 to Imax) NS,i / M - 接入成功概率
    
    數學表達式：
        PS = (Σ所有AC的成功設備數) / 總設備數M
    
    含義：
        在I_max個AC後，有多少比例的設備最終成功接入網絡
    
    參數：
        N_s_list (list): 各AC的成功設備數列表
                        N_s_list[0] 對應第1個AC，N_s_list[i-1] 對應第i個AC
        M (int): 總設備數
    
    返回：
        float: 接入成功概率 [0, 1]
    
    計算步驟：
        1. 對所有AC的成功設備數求和
        2. 除以總設備數M
    
    特點：
        📊 最重要的性能指標
        📈 值域 [0, 1]，越大越好
        🎯 Figure 3 的主要評估指標
    
    應用：
        - Figure 3: 成功率 vs 參數關係
        - 系統性能評估
        - 與模擬結果對比
    
    論文對應：Section III-A, Equation (8)
    """
    total_success = sum(N_s_list)
    return total_success / M if M > 0 else 0

def paper_formula_9_mean_access_delay(N_s_list):
    """
    【性能指標公式9】Ta = Σ(i=1 to Imax) i·NS,i / Σ(i=1 to Imax) NS,i - 平均接入延遲
    
    數學表達式：
        Ta = (Σ i × NS,i) / (Σ NS,i)
    
    含義：
        在所有成功的設備中，平均在第幾個AC才成功接入
        （加權平均延遲，以AC為單位）
    
    參數：
        N_s_list (list): 各AC的成功設備數列表
    
    返回：
        float: 平均接入延遲（AC單位）
    
    計算步驟：
        1. 對每個AC: i × NS,i （第i個AC的設備數乘以延遲AC數）
        2. 求和得分子（加權總延遲）
        3. 除以總成功設備數（分母）
    
    特點：
        ⏱️ 衡量系統響應速度
        📊 值越小越好（最小=1.0）
        🎯 Figure 4 的主要評估指標
    
    範例：
        若N_s_list = [80, 15, 4, 1]
        則 Ta = (1×80 + 2×15 + 3×4 + 4×1) / (80+15+4+1)
              = (80 + 30 + 12 + 4) / 100
              = 126 / 100 = 1.26 AC
    
    應用：
        - Figure 4: 平均延遲 vs 參數關係
        - 系統效率評估
        - QoS性能指標
    
    論文對應：Section III-A, Equation (9)
    """
    total_success = sum(N_s_list)
    if total_success <= 0:
        return 0
    
    weighted_sum = sum((i + 1) * N_s for i, N_s in enumerate(N_s_list))
    return weighted_sum / total_success

def paper_formula_10_collision_probability(N_c_list, I_max, N):
    """
    【性能指標公式10】PC = Σ(i=1 to Imax) NC,i / Σ(i=1 to Imax) Ni - 碰撞概率
    
    數學表達式：
        PC = (Σ所有AC的碰撞RAO數) / (I_max × N)
    
    含義：
        在所有可用的RAO中，有多少比例遭遇碰撞
        （碰撞資源的浪費比例）
    
    參數：
        N_c_list (list): 各AC的碰撞RAO數列表
        I_max (int): 最大AC數
        N (int): 每個AC的RAO數
    
    返回：
        float: 碰撞概率 [0, 1]
    
    計算步驟：
        1. 對所有AC的碰撞RAO數求和
        2. 除以總RAO數（I_max × N）
    
    特點：
        📊 衡量系統資源浪費
        📉 值越小越好（越少碰撞越好）
        🎯 Figure 5 的主要評估指標
    
    應用：
        - Figure 5: 碰撞概率 vs 參數關係
        - 系統效率評估
        - 資源利用率分析
    
    論文對應：Section III-A, Equation (10)
    """
    total_collision = sum(N_c_list)
    total_rao = I_max * N
    return total_collision / total_rao if total_rao > 0 else 0


# ============================================================================
# 【第6層】工具函數 - 統計與誤差分析
# ============================================================================
# 特點：輔助函數，用於數據處理、誤差計算、置信區間估計
# 應用：Figure 2 誤差分析、統計驗證

def calculate_idle_raos(device_count, N):
    """
    【工具函數】計算空閒RAO數量（期望值）
    
    數學表達式：
        idle_raos = N × ((N-1)/N)^M
    
    含義：
        當M個設備隨機選擇N個RAO時，
        有多少個RAO沒有被任何設備選中（期望值）
    
    推導：
        - 單個設備選擇某個RAO的概率 = 1/N
        - 單個設備不選擇某個RAO的概率 = (N-1)/N
        - M個設備都不選擇某個RAO的概率 = ((N-1)/N)^M
        - 期望有 N × ((N-1)/N)^M 個RAO為空閒
    
    參數：
        device_count (int): 設備數量（M）
        N (int): RAO總數
    
    返回：
        float: 空閒RAO的期望數量
    
    特點：
        🔢 使用log-space計算避免數值下溢
        ⚡ 當M或N很大時，直接計算會導致underflow
        📊 用於驗證 成功+碰撞+空閒=N 的關係
    
    實現細節：
        ((N-1)/N)^M = exp(M × log((N-1)/N))
        避免直接計算 ((N-1)/N)^M 導致精度損失
    
    應用：
        - 驗證RAO分類的完整性
        - 數據檢驗
        - Figure 1 輔助驗證
    """
    # 使用 log-space 計算避免數值下溢
    # ((N-1)/N)^M = exp(M * log((N-1)/N))
    if N <= 0:
        return 0.0
    if device_count == 0:
        return N
    
    prob_empty = np.exp(device_count * np.log((N-1)/N))
    return N * prob_empty

def relative_error_percentage(theoretical, actual):
    """
    【工具函數】計算相對誤差百分比
    
    數學表達式：
        error% = |actual - theoretical| / |theoretical| × 100%
    
    含義：
        衡量近似值與精確值的偏差程度（百分比形式）
    
    參數：
        theoretical (float): 理論/精確值
        actual (float): 實際/近似值
    
    返回：
        float: 相對誤差百分比 [0, +∞)
    
    特點：
        📊 歸一化誤差，便於比較不同量級的數值
        🔍 用於評估近似公式的準確度
        ⚠️ 當theoretical=0時，使用絕對誤差
    
    應用：
        - Figure 2: 誤差分析
        - 評估公式(4)-(5)的準確度
        - 算法性能驗證
    
    範例：
        theoretical = 27.5
        actual = 26.8
        error = |26.8 - 27.5| / |27.5| × 100 = 2.54%
    """
    if theoretical != 0:
        return abs(actual - theoretical) / abs(theoretical) * 100
    else:
        return abs(actual - theoretical) * 100

def confidence_interval_95(data):
    """
    【工具函數】計算95%置信區間（半寬）
    
    數學表達式：
        CI_95 = 1.96 × σ / √n
    
    含義：
        基於樣本數據估計總體參數的95%置信區間
        結果為半寬，完整區間 = [mean ± CI_95]
    
    參數：
        data (array-like): 樣本數據
    
    返回：
        float: 95%置信區間的半寬
    
    推導：
        - 假設數據近似正態分佈
        - 95%置信水平對應 z-score = 1.96
        - 標準誤 = σ / √n
        - 置信區間半寬 = 1.96 × 標準誤
    
    特點：
        📊 用於估計結果的不確定性
        🔍 樣本量越大，置信區間越小
        ⚠️ 基於正態分佈假設
    
    應用：
        - 模擬結果的不確定性估計
        - 論文圖表中的誤差棒（error bar）
        - 統計顯著性檢驗
    
    範例：
        若樣本 data = [10.2, 10.5, 9.8, 10.1, ...]（100個樣本）
        σ = 0.5, √n = 10
        CI_95 = 1.96 × 0.5 / 10 = 0.098
        表示95%置信度下，平均值的不確定性約為 ±0.098
    """
    return 1.96 * np.std(data) / np.sqrt(len(data))
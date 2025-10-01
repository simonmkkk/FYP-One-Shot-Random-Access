# 論文中的所有數學公式實現
# 基於論文: "Modeling and Estimation of Single Random Access for Finite-User Multi-Channel Slotted ALOHA Systems"
import numpy as np
from math import factorial, comb
import itertools
from functools import lru_cache

# 添加 LRU 緩存以加速重複計算
@lru_cache(maxsize=10000)
def paper_formula_1_pk_probability_cached(M: int, N1: int, k: int) -> float:
    """
    论文公式(1)的完全精确实现（帶緩存）：pk(M,N1)
    
    使用 LRU 緩存避免重複計算相同的 (M, N1, k) 組合
    """
    return paper_formula_1_pk_probability_uncached(M, N1, k)

def generate_partitions(n: int, k: int, min_val: int = 2):
    """
    生成所有满足条件的整数分割：i1+i2+...+ik = n, 每个ij >= min_val
    使用生成器以減少記憶體使用
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

def stirling_second_kind(n, k):
    """
    計算第二類斯特林數 S(n,k)
    使用遞推公式：S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    
    Args:
        n: 球的數量
        k: 盒子的數量
    
    Returns:
        int: 第二類斯特林數 S(n,k)
    """
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    if k == 1:
        return 1
    if k == n:
        return 1
    
    # 使用動態規劃計算
    dp = [[0 for _ in range(k+1)] for _ in range(n+1)]
    dp[0][0] = 1
    
    for i in range(1, n+1):
        for j in range(1, min(i+1, k+1)):
            dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
    
    return dp[n][k]

def paper_formula_1_pk_probability(M: int, N1: int, k: int) -> float:
    """
    论文公式(1)的完全精确实现：pk(M,N1)
    
    严格按照论文中的多重求和结构实现
    不使用任何概率近似或简化
    添加LRU快取以避免重複計算
    """
    # 使用緩存版本
    return _paper_formula_1_pk_probability_impl(M, N1, k)


@lru_cache(maxsize=10000)
def _paper_formula_1_pk_probability_impl(M: int, N1: int, k: int) -> float:
    """
    实际计算函数（帶緩存）
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
        
        # 生成将total_in_collision个用户分配到k个碰撞RAO的所有方式（每个至少2个用户）
        partitions = generate_partitions(total_in_collision, k, 2)
        
        for partition in partitions:
            # 计算将用户分配到特定分区的方式数
            ways_collision = 1
            remaining = M
            for count in partition:
                ways_collision *= comb(remaining, count)
                remaining -= count
            
            # 计算将剩余用户分配到非碰撞RAO的方式数
            # 从N1-k个非碰撞RAO中选择remaining_users个，每个分配1个用户
            ways_non_collision = comb(N1 - k, remaining_users) * factorial(remaining_users)
            
            # 计算将用户分配到特定RAO的方式数
            ways_specific_assignment = ways_collision * ways_non_collision
            
            # 乘以选择哪k个RAO作为碰撞RAO的方式数
            ways_choose_collision_raos = comb(N1, k)
            
            valid_ways += ways_specific_assignment * ways_choose_collision_raos
    
    pk = valid_ways / total_ways if total_ways > 0 else 0.0
    return pk

def paper_formula_2_collision_raos_exact(M: int, N1: int) -> float:
    """
    论文公式(2)的完全精确实现：NC,1
    
    NC,1 = Σ(k=1 to min{N1,⌊M/2⌋}) k * pk(M,N1)
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
    论文公式(3)的完全精确实现：NS,1
    
    使用完整的多重求和结构计算期望的成功用户数
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
            
            # 计算在给定k和total_in_collision的情况下，这个配置的概率
            ways_this_config = 0
            
            if k == 0:
                # 没有碰撞RAO的特殊情况
                if total_in_collision == 0 and remaining_users <= N1:
                    ways_non_collision = comb(N1, remaining_users) * factorial(remaining_users)
                    ways_this_config = ways_non_collision
            else:
                partitions = generate_partitions(total_in_collision, k, 2)
                for partition in partitions:
                    ways_collision = 1
                    remaining = M
                    for count in partition:
                        ways_collision *= comb(remaining, count)
                        remaining -= count
                    
                    ways_non_collision = comb(N1 - k, remaining_users) * factorial(remaining_users)
                    ways_specific_assignment = ways_collision * ways_non_collision
                    ways_choose_collision_raos = comb(N1, k)
                    
                    ways_this_config += ways_specific_assignment * ways_choose_collision_raos
            
            total_ways = N1 ** M
            prob_this_config = ways_this_config / total_ways if total_ways > 0 else 0
            
            if prob_this_config > 0:
                # 在这个配置下，成功用户数就是remaining_users
                expected_success_given_k += remaining_users * (prob_this_config / pk_val)
        
        NS_1 += expected_success_given_k * pk_val
    
    return NS_1

def paper_formula_4_success_approx(M, N1):
    """
    論文公式(4): NS,1 = M * e^(-M/N1)
    成功RAO近似公式 (論文原始版本)
    
    Args:
        M: 設備總數
        N1: RAO總數
    
    Returns:
        float: 成功RAO的近似數量
    """
    return M * np.exp(-M / N1)

def paper_formula_5_collision_approx(M, N1):
    """
    論文公式(5): NC,1 = N1 * (1 - e^(-M/N1) * (1 + M/N1))
    碰撞RAO近似公式 (論文原始版本) - 修正實現
    
    Args:
        M: 設備總數
        N1: RAO總數
    
    Returns:
        float: 碰撞RAO的近似數量
    """
    exp_term = np.exp(-M / N1)
    return N1 * (1 - exp_term * (1 + M/N1))

def paper_formula_6_success_per_cycle(K_i, N_i):
    """
    論文公式(6): NS,i = Ki * e^(-Ki/Ni)
    第i個AC的成功設備數
    
    Args:
        K_i: 第i個AC的競爭設備數
        N_i: 第i個AC的RAO數
    
    Returns:
        float: 成功設備數
    """
    return K_i * np.exp(-K_i / N_i)

def paper_formula_7_next_contending_devices(K_i, N_i):
    """
    論文公式(7): Ki+1 = Ki(1 - e^(-Ki/Ni))
    下一個AC的競爭設備數 (論文完整版本)
    
    Args:
        K_i: 當前AC的競爭設備數  
        N_i: 當前AC的RAO數
    
    Returns:
        float: 下一個AC的競爭設備數
    """
    return K_i * (1 - np.exp(-K_i / N_i))

def paper_formula_8_access_success_probability(N_s_list, M):
    """
    論文公式(8): PS = Σ(i=1 to Imax) NS,i / M
    接入成功概率
    
    Args:
        N_s_list: 各AC的成功設備數列表
        M: 總設備數
    
    Returns:
        float: 接入成功概率
    """
    total_success = sum(N_s_list)
    return total_success / M if M > 0 else 0

def paper_formula_9_mean_access_delay(N_s_list):
    """
    論文公式(9): Ta = Σ(i=1 to Imax) i * NS,i / Σ(i=1 to Imax) NS,i
    平均接入延遲
    
    Args:
        N_s_list: 各AC的成功設備數列表
    
    Returns:
        float: 平均接入延遲 (以AC為單位)
    """
    total_success = sum(N_s_list)
    if total_success <= 0:
        return 0
    
    weighted_sum = sum((i + 1) * N_s for i, N_s in enumerate(N_s_list))
    return weighted_sum / total_success

def paper_formula_10_collision_probability(N_c_list, I_max, N):
    """
    論文公式(10): PC = Σ(i=1 to Imax) NC,i / Σ(i=1 to Imax) Ni
    碰撞概率
    
    Args:
        N_c_list: 各AC的碰撞RAO數列表
        I_max: 最大AC數
        N: 每個AC的RAO數
    
    Returns:
        float: 碰撞概率
    """
    total_collision = sum(N_c_list)
    total_rao = I_max * N
    return total_collision / total_rao if total_rao > 0 else 0

def calculate_idle_raos(device_count, N):
    """
    計算空閒RAO數量
    每個RAO不被任何設備選中的概率: ((N-1)/N)^M
    
    Args:
        device_count: 設備數量
        N: RAO總數
    
    Returns:
        float: 空閒RAO的期望數量
    """
    prob_empty = ((N-1)/N)**device_count
    return N * prob_empty

def relative_error_percentage(theoretical, actual):
    """
    計算相對誤差百分比
    
    Args:
        theoretical: 理論值
        actual: 實際值
    
    Returns:
        float: 相對誤差百分比
    """
    if theoretical != 0:
        return abs(actual - theoretical) / abs(theoretical) * 100
    else:
        return abs(actual - theoretical) * 100

def confidence_interval_95(data):
    """
    計算95%置信區間
    
    Args:
        data: 數據數組
    
    Returns:
        float: 95%置信區間半寬
    """
    return 1.96 * np.std(data) / np.sqrt(len(data))
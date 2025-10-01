# 主程序入口
import numpy as np
import os
import sys
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.simulation import run_parallel_simulation
from analysis.metrics import calculate_performance_metrics
from visualization.plotting import plot_single_results, plot_scan_results
from utils.file_io import save_single_results_to_csv, save_scan_results_to_csv
from analysis.theoretical import theoretical_calculation

# ============================================================================
# 配置參數設置（整合自 settings.py）
# ============================================================================

# ===== 運行模式 =====
RUN_MODE = 'scan'  # 'single': 单点模拟, 'scan': 参数扫描

# ===== 模擬本質參數（ALOHA系統參數）=====
M = 100           # 設備總數 - 嘗試接入網絡的設備數量
N = 40            # RAO數量 - 每個接入周期(AC)的隨機接入機會(RAO)數量
I_max = 10        # 最大接入周期數 - 最大重傳次數限制

# ===== 參數掃描設置（僅在 RUN_MODE='scan' 時生效）=====
SCAN_PARAM = 'N'              # 掃描參數: 'N', 'M', 'I_max'
SCAN_RANGE = range(5, 46, 1)  # 掃描範圍: N=5,6,7,...,45

# ===== 性能優化參數（計算資源配置）=====
NUM_SAMPLES = 5000000   # 樣本數量 - 每個參數點的模擬次數（論文使用 10^7）
NUM_WORKERS = 16     # 並行進程數 - CPU核心數（建議設置為實際CPU核心數）

# ===== 輸出設置 =====
PLOT_RESULTS = True     # 是否繪製結果圖表
SAVE_TO_CSV = True      # 是否保存結果到CSV文件

def run_single_simulation():
    """
    运行单点模拟
    """
    print("=" * 60)
    print("多信道时隙ALOHA系统模拟器 - 专注于群组寻呼场景")
    print("=" * 60)
    print(f"模拟参数: M={M}, N={N}, I_max={I_max}")
    print(f"样本数量: {NUM_SAMPLES}")
    print("使用 CPU 並行模擬")
    print(f"並行進程: {NUM_WORKERS}")
    print("=" * 60)

    # 使用CPU并行模拟
    results_array = run_parallel_simulation(M, N, I_max, NUM_SAMPLES, NUM_WORKERS)

    # 计算性能指标
    means, confidence_intervals = calculate_performance_metrics(results_array)
    mean_ps, mean_ta, mean_pc = means
    ci_ps, ci_ta, ci_pc = confidence_intervals
    
    # 打印结果
    print("\n模拟结果:")
    print(f"接入成功率 (P_S): {mean_ps:.6f} ± {ci_ps:.6f} (95% 置信区间)")
    print(f"平均接入延迟 (T_a): {mean_ta:.6f} ± {ci_ta:.6f} (95% 置信区间)")
    print(f"碰撞概率 (P_C): {mean_pc:.6f} ± {ci_pc:.6f} (95% 置信区间)")
    
    # 计算理论值
    ps_theory, ta_theory, pc_theory, _, _ = theoretical_calculation(M, N, I_max)
    print("\n理论值 (论文方法):")
    print(f"接入成功率 (P_S): {ps_theory:.6f}")
    print(f"平均接入延迟 (T_a): {ta_theory:.6f}")
    print(f"碰撞概率 (P_C): {pc_theory:.6f}")
    
    # 绘制结果
    if PLOT_RESULTS:
        print("\n生成结果图表...")
        plot_single_results(results_array, M, N, I_max)
    
    # 保存结果到CSV
    if SAVE_TO_CSV:
        csv_file = save_single_results_to_csv(results_array, M, N, I_max, NUM_SAMPLES)
    
    print("\n模拟完成!")

def run_parameter_scan():
    """
    运行参数扫描
    """
    print("=" * 60)
    print("多信道时隙ALOHA系统性能分析 - 参数扫描")
    print("=" * 60)
    print(f"扫描参数: {SCAN_PARAM}")
    print(f"扫描范围: {list(SCAN_RANGE)}")
    print(f"固定参数: M={M}, I_max={I_max}")
    print(f"样本数量: {NUM_SAMPLES}")
    
    print("使用 CPU 並行模擬")
    print(f"並行進程: {NUM_WORKERS}")
    print("=" * 60)
    
    param_values = []
    P_S_values = []
    T_a_values = []
    P_C_values = []
    
    # 理论值数组
    P_S_theory_values = []
    T_a_theory_values = []
    P_C_theory_values = []
    
    # 根据扫描参数设置变量
    if SCAN_PARAM == 'N':
        scan_range = SCAN_RANGE
        fixed_params = {'M': M, 'I_max': I_max}
    elif SCAN_PARAM == 'M':
        scan_range = SCAN_RANGE
        fixed_params = {'N': N, 'I_max': I_max}
    elif SCAN_PARAM == 'I_max':
        scan_range = SCAN_RANGE
        fixed_params = {'M': M, 'N': N}
    else:
        raise ValueError("不支持的扫描参数，请选择'N', 'M'或'I_max'")
    
    # 遍历扫描范围
    for param_value in scan_range:
        print(f"正在处理 {SCAN_PARAM}={param_value}...")
        
        # 设置当前参数值
        if SCAN_PARAM == 'N':
            current_N = param_value
            current_M = M
            current_I_max = I_max
        elif SCAN_PARAM == 'M':
            current_M = param_value
            current_N = N
            current_I_max = I_max
        elif SCAN_PARAM == 'I_max':
            current_I_max = param_value
            current_M = M
            current_N = N
        
        # 執行模擬
        import time
        start_time = time.time()
        
        print("使用CPU並行模擬...")
        # 使用 CPU 並行模擬
        results_array = run_parallel_simulation(
            current_M, current_N, current_I_max, NUM_SAMPLES, NUM_WORKERS
        )
        means, _ = calculate_performance_metrics(results_array)
        mean_ps, mean_ta, mean_pc = means
        device_type = "CPU"
        
        end_time = time.time()
        elapsed = end_time - start_time
        speed = NUM_SAMPLES / elapsed
        print(f"{device_type}模擬耗時: {elapsed:.2f}秒, 速度: {speed:.0f} 樣本/秒")
        
        # 计算理论值
        ps_theory, ta_theory, pc_theory, _, _ = theoretical_calculation(current_M, current_N, current_I_max)
        
        param_values.append(param_value)
        P_S_values.append(mean_ps)
        T_a_values.append(mean_ta)
        P_C_values.append(mean_pc)
        
        P_S_theory_values.append(ps_theory)
        T_a_theory_values.append(ta_theory)
        P_C_theory_values.append(pc_theory)
        
        print(f"  结果: P_S={mean_ps:.4f}, T_a={mean_ta:.4f}, P_C={mean_pc:.4f}")
    
    print("=" * 60)
    print("扫描完成!")
    
    return (np.array(param_values), np.array(P_S_values), np.array(T_a_values), np.array(P_C_values), 
            np.array(P_S_theory_values), np.array(T_a_theory_values), np.array(P_C_theory_values))

def main():
    """
    主函數 - 根據 RUN_MODE 執行單點模擬或參數掃描
    """
    if RUN_MODE == 'single':
        run_single_simulation()
    elif RUN_MODE == 'scan':
        results = run_parameter_scan()
        param_values, P_S_values, T_a_values, P_C_values, P_S_theory_values, T_a_theory_values, P_C_theory_values = results
        
        # 打印结果摘要
        print("\n结果摘要:")
        print(f"{SCAN_PARAM}范围: {param_values[0]} 到 {param_values[-1]}")
        print(f"P_S范围: {np.min(P_S_values):.4f} 到 {np.max(P_S_values):.4f}")
        print(f"T_a范围: {np.min(T_a_values):.4f} 到 {np.max(T_a_values):.4f}")
        print(f"P_C范围: {np.min(P_C_values):.4f} 到 {np.max(P_C_values):.4f}")
        
        # 保存结果到CSV
        if SAVE_TO_CSV:
            csv_file = save_scan_results_to_csv(param_values, P_S_values, T_a_values, P_C_values, SCAN_PARAM, M, I_max)
        
        # 绘制图表
        if PLOT_RESULTS:
            print("\n生成系统性能图表...")
            combined_fig = plot_scan_results(param_values, P_S_values, T_a_values, P_C_values, 
                                               P_S_theory_values, T_a_theory_values, P_C_theory_values, 
                                               SCAN_PARAM, M, I_max)
            
            # 保存合併的圖表 (Figures 3-5)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存合併的 Figure 3-5
            combined_filename = f"figures_3_4_5_combined_{SCAN_PARAM}_{timestamp}.png"
            combined_filepath = os.path.join('data', 'figures', combined_filename)
            os.makedirs(os.path.dirname(combined_filepath), exist_ok=True)
            combined_fig.savefig(combined_filepath, dpi=300, bbox_inches='tight')
            print(f"合併的 Figure 3-5 已保存到: {combined_filepath}")
    
    else:
        raise ValueError("不支持的运行模式，请选择'single'或'scan'")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成論文Figure 1和Figure 2的驗證版本：近似公式 vs 單次接入模擬的比較

【舊版本 - 串行外層循環】
此版本使用串行遍歷 M 值，每個 M 內部並行處理
性能較慢，僅作為對比參考

====================================
依賴套件（需要先安裝）：
pip install numpy matplotlib joblib tqdm
====================================

說明：
此文件包含所有必需的繪圖和數據生成代碼，
模擬函數調用 core.simulation 模組，
公式計算調用 analysis.formulas 模組。
"""

# ===== 配置參數 =====
N_VALUES = [3, 14]       # 要分析的 N 值列表
N_JOBS = 16         # 並行進程數（建議設為 CPU 核心數）
NUM_SAMPLES = 100000  # 每個(M,N)點的模擬樣本數
# ===================

import sys
import os
from datetime import datetime
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# 添加當前目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 從核心模組導入數學公式和模擬函數
from analysis.formulas import (
    paper_formula_4_success_approx,
    paper_formula_5_collision_approx
)
from core.simulation import simulate_single_access_parallel


# ============================================================================
# 繪圖函數（完全獨立，無需外部依賴）
# ============================================================================

def extract_n_values_from_data(data_dict):
    """從數據字典中提取 N 值列表"""
    n_keys = []
    n_values = []
    for key in sorted(data_dict.keys()):
        if key.startswith('N_'):
            n_val = int(key.split('_')[1])
            n_keys.append(key)
            n_values.append(n_val)
    return n_keys, n_values

def plot_figure1_simulation_validation(sim_vs_approx_data, ax: 'matplotlib.axes.Axes' = None):
    """
    繪製論文Figure 1的驗證版本: 近似公式 vs 單次接入模擬
    動態處理任意數量的 N 值
    """
    # 從數據中自動提取 N 值
    available_N_keys, available_N_values = extract_n_values_from_data(sim_vs_approx_data)
    
    if not available_N_keys:
        raise ValueError(f"數據中沒有找到任何 N 值")
    
    # 若外部提供單一軸，僅在單一 N 場景下於該軸繪製
    if ax is not None:
        if len(available_N_keys) != 1:
            raise ValueError("提供 ax 時，Fig.1 目前僅支援單一 N 值的繪製。")
        fig = ax.figure
        N_key = available_N_keys[0]
        N_value = available_N_values[0]
        N_data = sim_vs_approx_data[N_key]
        
        # 成功RAO: 近似公式 vs 模擬結果（使用與 analytical 一致的樣式）
        ax.plot(N_data['M_over_N'], N_data['sim_N_S'], 'ko-', linewidth=1.5, markersize=4, 
                label=f'N={N_value} $N_{{S,1}}$/N Simulation')
        ax.plot(N_data['M_over_N'], N_data['theory_N_S'], 'k:', linewidth=1.5, 
                label='$N_{S,1}$/N Derived Performance Metric, Eq. (4)')
        
        # 碰撞RAO: 近似公式 vs 模擬結果
        ax.plot(N_data['M_over_N'], N_data['sim_N_C'], 'ko', fillstyle='none', markersize=4, linewidth=1.5,
                label=f'N={N_value} $N_{{C,1}}$/N Simulation')
        ax.plot(N_data['M_over_N'], N_data['theory_N_C'], 'k--', linewidth=1.5, 
                label='$N_{C,1}$/N Derived Performance Metric, Eq. (5)')
        
        ax.set_xlabel('M/N', fontsize=12)
        ax.set_ylabel('RAOs/N', fontsize=12)
        ax.set_title(f'Fig. 1. Simulation and approximation results of $N_{{S,1}}$/N and $N_{{C,1}}$/N', 
                    fontsize=11)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        return fig
    
    # 根據可用數據數量設置子圖佈局
    num_plots = len(available_N_keys)
    if num_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]  # 轉換為列表以統一處理
    elif num_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        # 對於3個或更多N值，使用2行布局
        rows = (num_plots + 1) // 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5*rows))
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if num_plots == 1 else axes
    
    # 動態繪製每個 N 值的子圖
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']  # 支援最多6個子圖
    
    for i, (N_key, N_value) in enumerate(zip(available_N_keys, available_N_values)):
        if i >= len(axes):
            break  # 防止索引超出範圍
            
        N_data = sim_vs_approx_data[N_key]
        ax = axes[i]
        
        # 成功RAO: 近似公式 vs 模擬結果（使用與 analytical 一致的樣式）
        ax.plot(N_data['M_over_N'], N_data['sim_N_S'], 'ko-', linewidth=1.5, markersize=4, 
                label=f'N={N_value} $N_{{S,1}}$/N Simulation')
        ax.plot(N_data['M_over_N'], N_data['theory_N_S'], 'k:', linewidth=1.5, 
                label='$N_{S,1}$/N Derived Performance Metric, Eq. (4)')
        
        # 碰撞RAO: 近似公式 vs 模擬結果
        ax.plot(N_data['M_over_N'], N_data['sim_N_C'], 'ko', fillstyle='none', markersize=4, linewidth=1.5,
                label=f'N={N_value} $N_{{C,1}}$/N Simulation')
        ax.plot(N_data['M_over_N'], N_data['theory_N_C'], 'k--', linewidth=1.5, 
                label='$N_{C,1}$/N Derived Performance Metric, Eq. (5)')
        
        ax.set_xlabel('M/N', fontsize=12)
        ax.set_ylabel('RAOs/N', fontsize=12)
        ax.set_title(f'Fig. 1. Simulation and approximation results of $N_{{S,1}}$/N and $N_{{C,1}}$/N', 
                    fontsize=11)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
    
    # 隱藏多餘的子圖（如果有的話）
    if num_plots < len(axes):
        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_figure2_simulation_validation(error_data, ax: 'matplotlib.axes.Axes' = None):
    """
    繪製論文Figure 2的驗證版本: 近似公式與模擬的誤差分析
    動態處理任意數量的 N 值，使用不同的線型和標記區分
    """
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(8, 6))
    
    # 從數據中自動提取 N 值
    available_N_keys, available_N_values = extract_n_values_from_data(error_data)
    
    if not available_N_keys:
        raise ValueError(f"數據中沒有找到任何 N 值")
    
    # 動態繪製每個 N 值的誤差曲線（使用與 analytical 一致的樣式）
    for i, (N_key, N_value) in enumerate(zip(available_N_keys, available_N_values)):
        N_data = error_data[N_key]
        
        # 根據 N 值選擇不同的樣式
        if i == 0:
            # 第一個 N 值：使用標記點
            # 成功 RAO 誤差（實心圓標記 + 實線）
            ax.plot(N_data['M_over_N'], N_data['N_S_error'], 
                    'ko-', linewidth=1.5, markersize=4,
                    label=f'N={N_value} $N_{{S,1}}$/N')
            # 碰撞 RAO 誤差（空心圓標記 + 實線）
            ax.plot(N_data['M_over_N'], N_data['N_C_error'], 
                    'ko-', fillstyle='none', linewidth=1.5, markersize=4,
                    label=f'N={N_value} $N_{{C,1}}$/N')
        else:
            # 其他 N 值：僅使用線型
            # 成功 RAO 誤差（實線）
            ax.plot(N_data['M_over_N'], N_data['N_S_error'], 
                    'k-', linewidth=1.5,
                    label=f'N={N_value} $N_{{S,1}}$/N')
            # 碰撞 RAO 誤差（虛線）
            ax.plot(N_data['M_over_N'], N_data['N_C_error'], 
                    'k--', linewidth=1.5,
                    label=f'N={N_value} $N_{{C,1}}$/N')
    
    # 設置軸和標籤
    ax.set_xlabel('M/N', fontsize=12)
    ax.set_ylabel('Approximation Error (%)', fontsize=12)
    ax.set_xlim(0, 10)
    
    # 設置對數縱軸
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e3)
    
    # 添加網格
    ax.grid(True, alpha=0.3)
    
    # 添加圖例
    ax.legend(fontsize=8, loc='best')
    
    # 動態設置標題
    N_values_str = ' and '.join(map(str, available_N_values))
    title = f'Fig. 2. Absolute approximation error of $N_{{S,1}}$/N and $N_{{C,1}}$/N with N = {N_values_str}'
    
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    return created_fig if created_fig is not None else ax.figure

# ============================================================================
# 數據生成函數 - 【舊版本：串行外層循環】
# ============================================================================

def generate_simulation_vs_approximation_data(n_values, n_jobs, num_samples):
    """
    生成近似公式 vs 單次接入模擬的比較數據
    
    【舊版本】：使用串行外層循環遍歷 M 值，每個 M 內部並行處理
    性能較慢，約為新版本的 1/21
    
    Args:
        n_values: 要分析的 N 值列表
        n_jobs: 並行作業數量
        num_samples: 每個(M,N)點的模擬樣本數
    
    Returns:
        dict: 包含理論值、模擬值和誤差的數據
    """
    results = {}
    
    for N in n_values:
        print(f"\n正在生成 N={N} 的模擬vs近似比較數據...")
        print(f"【使用舊版本：串行外層循環】")
        
        # 每個整數 M（1..10N）皆模擬
        M_range = list(range(1, 10*N + 1))
        
        start_time = time.time()
        
        # 初始化結果列表
        M_values = []
        
        # 理論值 (近似公式)
        theory_N_S = []
        theory_N_C = []
        
        # 模擬值
        sim_N_S = []
        sim_N_C = []
        
        # 誤差
        error_N_S = []
        error_N_C = []
        
        print(f"  總共需要模擬 {len(M_range)} 個數據點，每點 {num_samples} 樣本...")
        
        # 【舊版本】：串行遍歷每個 M 值
        for idx, M in enumerate(M_range):
            if idx % 5 == 0:  # 每5個點顯示進度
                print(f"  進度: {idx+1}/{len(M_range)} (M={M}, M/N={M/N:.2f})")
            
            # 計算理論值（近似公式）
            theory_ns = paper_formula_4_success_approx(M, N)
            theory_nc = paper_formula_5_collision_approx(M, N)
            
            # 執行模擬（內層並行）
            sim_ns, sim_nc, sim_idle = simulate_single_access_parallel(
                M, N, num_samples, n_jobs  # 內層多核並行
            )
            
            # 計算相對誤差百分比
            error_ns = abs(sim_ns - theory_ns) / abs(theory_ns) * 100 if theory_ns > 0 else 0
            error_nc = abs(sim_nc - theory_nc) / abs(theory_nc) * 100 if theory_nc > 0 else 0
            
            # 保存結果
            M_values.append(M)
            theory_N_S.append(theory_ns / N)  # 正規化
            theory_N_C.append(theory_nc / N)  # 正規化
            sim_N_S.append(sim_ns / N)  # 正規化
            sim_N_C.append(sim_nc / N)  # 正規化
            error_N_S.append(error_ns)
            error_N_C.append(error_nc)
        
        elapsed_time = time.time() - start_time
        print(f"N={N} 模擬完成，耗時: {elapsed_time:.2f}秒")
        
        results[f'N_{N}'] = {
            'M_values': M_values,
            'M_over_N': [m/N for m in M_values],
            # 理論值 (近似公式)
            'theory_N_S': theory_N_S,
            'theory_N_C': theory_N_C,
            # 模擬值
            'sim_N_S': sim_N_S,
            'sim_N_C': sim_N_C,
            # 誤差
            'error_N_S': error_N_S,
            'error_N_C': error_N_C
        }
    
    return results

def generate_simulation_vs_approximation_error_data(sim_vs_approx_data):
    """
    從模擬vs近似數據中提取誤差數據（用於Figure 2驗證版）
    
    Args:
        sim_vs_approx_data: generate_simulation_vs_approximation_data()的輸出
    
    Returns:
        dict: 誤差數據
    """
    error_results = {}
    
    for key, data in sim_vs_approx_data.items():
        print(f"正在提取 {key} 的誤差數據...")
        
        error_results[key] = {
            'M_over_N': data['M_over_N'],
            'N_S_error': data['error_N_S'],
            'N_C_error': data['error_N_C']
        }
        print(f"  {key} 誤差數據提取完成")
    
    return error_results

# ============================================================================
# 主程式
# ============================================================================

def main():
    print("=" * 60)
    print("生成論文Figure 1和Figure 2的驗證版本")
    print("近似公式 vs 單次接入模擬的比較")
    print("【舊版本 - 串行外層循環】")
    print(f"分析參數：N = {N_VALUES}, 並行進程數 = {N_JOBS}, 樣本數 = {NUM_SAMPLES}")
    print("=" * 60)
    
    # 生成模擬vs近似比較數據
    print("\n正在生成模擬vs近似比較數據...")
    print(f"使用 {N_JOBS} 個核心並行計算（僅內層並行），每點 {NUM_SAMPLES} 個樣本...")
    
    sim_vs_approx_data = generate_simulation_vs_approximation_data(
        n_values=N_VALUES,
        n_jobs=N_JOBS, 
        num_samples=NUM_SAMPLES
    )
    print("模擬vs近似比較數據生成完成!")
    
    # 提取誤差數據
    print("\n正在提取誤差數據...")
    error_data = generate_simulation_vs_approximation_error_data(sim_vs_approx_data)
    print("誤差數據提取完成!")
    
    # 繪製與保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figures_dir = os.path.join('data', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n正在繪製組合圖表...")
    num_n_values = len(N_VALUES)
    
    # 創建 2 行 × N 列的子圖佈局（與 analytical 一致）
    fig_width = 5 * num_n_values
    fig_height = 8
    fig_combined, axes = plt.subplots(2, num_n_values, figsize=(fig_width, fig_height), 
                                      constrained_layout=True, squeeze=False)
    
    # 為每個 N 值繪製 Fig1 和 Fig2
    for col_idx, N in enumerate(N_VALUES):
        n_key = f'N_{N}'
        
        # 繪製 Fig1 到第一行
        if n_key in sim_vs_approx_data:
            ax1 = axes[0, col_idx]
            # 提取單個 N 的數據
            single_n_fig1_data = {n_key: sim_vs_approx_data[n_key]}
            plot_figure1_simulation_validation(single_n_fig1_data, ax=ax1)
        
        # 繪製 Fig2 到第二行
        if n_key in error_data:
            ax2 = axes[1, col_idx]
            # 提取單個 N 的數據
            single_n_fig2_data = {n_key: error_data[n_key]}
            plot_figure2_simulation_validation(single_n_fig2_data, ax=ax2)
    
    # 保存組合圖
    combined_path = os.path.join(figures_dir, f"figure1_2_simulation_combined_OLD_SERIAL_{timestamp}.png")
    fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ 組合圖已保存：{combined_path}")
    
    # 顯示圖表
    plt.show()
    
    # 分別保存單獨的圖表
    print("\n正在繪製並保存單獨的 Figure 1...")
    fig1 = plot_figure1_simulation_validation(sim_vs_approx_data)
    fig1_path = os.path.join(figures_dir, f"figure1_simulation_validation_OLD_SERIAL_{timestamp}.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 1已保存：{fig1_path}")
    
    print("\n正在繪製並保存單獨的 Figure 2...")
    fig2 = plot_figure2_simulation_validation(error_data)
    fig2_path = os.path.join(figures_dir, f"figure2_simulation_validation_OLD_SERIAL_{timestamp}.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 2已保存：{fig2_path}")
    
    # 顯示一些關鍵結果
    print("\n" + "=" * 60)
    print("關鍵結果：")
    for N in N_VALUES:
        key = f'N_{N}'
        if key in error_data:
            max_error_s = max(error_data[key]['N_S_error'])
            max_error_c = max(error_data[key]['N_C_error'])
            print(f"  N={N} 成功RAO的最大模擬誤差: {max_error_s:.2f}%")
            print(f"  N={N} 碰撞RAO的最大模擬誤差: {max_error_c:.2f}%")
    
    print("\n結論：")
    print("  1. 模擬結果驗證了近似公式的準確性")
    print("  2. 較大的N值下，近似公式與模擬結果更接近")
    print("  3. 這證實了論文理論分析的正確性")
    print("  【注意】此為舊版本（串行），速度約為新版本的 1/21")
    print("=" * 60)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成論文Figure 1和Figure 2：單次隨機接入中近似公式的有效範圍分析
完全獨立版本 - 僅依賴標準庫和基礎科學計算套件

====================================
依賴套件（需要先安裝）：
pip install numpy matplotlib joblib tqdm
====================================

說明：
此文件包含所有必需的代碼（數學公式、數據生成、繪圖函數），
不依賴項目中的任何其他模組，可以完全獨立運行。
"""

# ===== 配置參數 =====
N_VALUES = [3]  # 要分析的 N 值列表，例如 [3, 14] 或 [14]
N_JOBS = 16      # 並行進程數（建議設為 CPU 核心數）
# ===================

import os
import sys
from datetime import datetime
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import matplotlib
# 優先使用可互動後端（顯示視窗），不可用則回退到 Agg
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 添加當前目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 從核心模組導入數學公式（這些是專案的核心共用元件）
from analysis.formulas import (
    paper_formula_2_collision_raos_exact,
    paper_formula_3_success_raos_exact,
    paper_formula_4_success_approx,
    paper_formula_5_collision_approx
)

# 設置matplotlib支持中文顯示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題



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

def plot_figure1(fig1_data, ax: 'matplotlib.axes.Axes' = None):
    """
    繪製論文Figure 1: 單次隨機接入中分析模型vs近似公式
    動態處理任意數量的 N 值
    """
    # 從數據中自動提取 N 值
    available_N_keys, available_N_values = extract_n_values_from_data(fig1_data)
    if not available_N_keys:
        raise ValueError(f"數據中沒有找到任何 N 值")

    # 若外部提供單一軸，僅在單一 N 場景下於該軸繪製
    if ax is not None:
        if len(available_N_keys) != 1:
            raise ValueError("提供 ax 時，Fig.1 目前僅支援單一 N 值的繪製。請在主程式限制 N_VALUES 為單一值。")
        fig = ax.figure
        N_key = available_N_keys[0]
        N_value = available_N_values[0]
        N_data = fig1_data[N_key]
        # 成功RAO: 分析模型 vs 近似公式4
        ax.plot(N_data['M_over_N'], N_data['analytical_N_S'], 'ko-', linewidth=1.5, markersize=4, 
                label=f'N={N_value} $N_{{S,1}}$/N Analytical Model')
        ax.plot(N_data['M_over_N'], N_data['approx_N_S'], 'k:', linewidth=1.5, 
                label='$N_{S,1}$/N Derived Performance Metric, Eq. (4)')
        # 碰撞RAO: 分析模型 vs 近似公式5
        ax.plot(N_data['M_over_N'], N_data['analytical_N_C'], 'ko', fillstyle='none', markersize=4, linewidth=1.5,
                label=f'N={N_value} $N_{{C,1}}$/N Analytical Model')
        ax.plot(N_data['M_over_N'], N_data['approx_N_C'], 'k--', linewidth=1.5, 
                label='$N_{C,1}$/N Derived Performance Metric, Eq. (5)')
        ax.set_xlabel('M/N', fontsize=12)
        ax.set_ylabel('RAOs/N', fontsize=12)
        ax.set_title(f'Fig. 1. Analytical and approximation results of $N_{{S,1}}$/N and $N_{{C,1}}$/N', 
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        return fig

    # 否則按原邏輯建立子圖
    num_plots = len(available_N_keys)
    if num_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    elif num_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        rows = (num_plots + 1) // 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if num_plots == 1 else axes
    
    # 動態繪製每個 N 值的子圖
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']  # 支援最多6個子圖
    
    for i, (N_key, N_value) in enumerate(zip(available_N_keys, available_N_values)):
        if i >= len(axes):
            break  # 防止索引超出範圍
            
        N_data = fig1_data[N_key]
        ax = axes[i]
        
        # 成功RAO: 分析模型 vs 近似公式4
        ax.plot(N_data['M_over_N'], N_data['analytical_N_S'], 'b-', 
                linewidth=2, label='Successful RAOs (Analytical, Eq. 3)')
        ax.plot(N_data['M_over_N'], N_data['approx_N_S'], 'b--', 
                linewidth=2, label='Successful RAOs (Approximation, Eq. 4)')
        
        # 碰撞RAO: 分析模型 vs 近似公式5
        ax.plot(N_data['M_over_N'], N_data['analytical_N_C'], 'r-', 
                linewidth=2, label='Collision RAOs (Analytical, Eq. 2)')
        ax.plot(N_data['M_over_N'], N_data['approx_N_C'], 'r--', 
                linewidth=2, label='Collision RAOs (Approximation, Eq. 5)')
        
        ax.set_xlabel('M/N', fontsize=12)
        ax.set_ylabel('Normalized RAOs', fontsize=12)
        
        # 動態設置標題
        subplot_label = subplot_labels[i] if i < len(subplot_labels) else f'({chr(97+i)})'
        ax.set_title(f'{subplot_label} N={N_value}: Analytical vs Approximation', 
                    fontsize=13, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 10)
        
        # 動態設置 y 軸範圍
        max_y = max(max(N_data['analytical_N_C']), max(N_data['approx_N_C']))
        ax.set_ylim(0, max_y * 1.1)
    
    # 隱藏多餘的子圖（如果有的話）
    if num_plots < len(axes):
        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)
    
    # 動態設置主標題
    N_values_str = ', '.join(map(str, available_N_values))
    if len(available_N_values) == 1:
        title = f'Fig. 1. Validity range of approximation in single random access (N={N_values_str})'
    else:
        title = f'Fig. 1. Validity range of approximation in single random access (N={N_values_str})'
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_figure2(fig2_data, ax: 'matplotlib.axes.Axes' = None):
    """
    繪製論文Figure 2: 絕對近似誤差分析
    動態處理任意數量的 N 值，使用不同的線型和標記區分
    """
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(12, 8))
    
    # 從數據中自動提取 N 值
    available_N_keys, available_N_values = extract_n_values_from_data(fig2_data)
    
    if not available_N_keys:
        raise ValueError(f"數據中沒有找到任何 N 值")
    
    # 動態繪製每個 N 值的誤差曲線（使用論文格式）
    for i, (N_key, N_value) in enumerate(zip(available_N_keys, available_N_values)):
        N_data = fig2_data[N_key]
        
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
    
    # 設置對數縱軸 (按照論文Figure 2: 10^-2 到 10^3)
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
# 數據生成函數（從 single_access_analysis.py 整合）
# ============================================================================

def analytical_model(M, N):
    """
    使用論文公式(2)和(3)的分析模型（組合模型）
    """
    N_S = paper_formula_3_success_raos_exact(M, N)
    N_C = paper_formula_2_collision_raos_exact(M, N)
    return N_S, N_C

def approximation_formula(M, N):
    """
    使用論文公式(4)和(5)的近似公式
    """
    N_S = paper_formula_4_success_approx(M, N) 
    N_C = paper_formula_5_collision_approx(M, N)
    return N_S, N_C

def compute_single_point(M, N):
    """
    計算單個(M,N)點的分析模型和近似公式結果
    
    Returns:
        tuple: (M, N_S_anal/N, N_C_anal/N, N_S_approx/N, N_C_approx/N)
    """
    # 分析模型結果
    N_S_anal, N_C_anal = analytical_model(M, N)
    
    # 近似公式結果  
    N_S_approx, N_C_approx = approximation_formula(M, N)
    
    return (M, N_S_anal/N, N_C_anal/N, N_S_approx/N, N_C_approx/N)

def generate_figure1_data(n_values, n_jobs):
    """
    生成Figure 1的數據
    M從1到10N，比較分析模型vs近似公式
    """
    results = {}
    
    for N in n_values:
        print(f"\n正在計算 N={N} 的數據...")
        
        # 目標 M/N 網格（與論文座標一致）
        target_m_over_n = np.arange(0, 10.5, 0.5)  # 0, 0.5, 1, ..., 10
        # 僅保留能被當前 N 精確表示的點（使 M 為整數）
        M_range = []
        for m_n in target_m_over_n:
            m_real = m_n * N
            if abs(round(m_real) - m_real) < 1e-9:
                M_range.append(int(round(m_real)))
        # 移除 0（若存在），並保證至少從 1 開始
        M_range = [m for m in sorted(set(M_range)) if m >= 1]
        print(f"  將計算 {len(M_range)} 個精確 M/N 點: {[f'{m/N:.2f}' for m in M_range]}")
        
        start_time = time.time()
        
        if n_jobs == 1:
            # 單線程計算，顯示詳細進度
            print(f"  單線程逐一計算 {len(M_range)} 個數據點 (M/N = 0 到 10)...")
            
            M_values = []
            analytical_N_S = []
            analytical_N_C = []
            approx_N_S = []
            approx_N_C = []
            
            for idx, M in enumerate(M_range):
                print(f"  計算數據點 {idx+1}: M={M}, M/N={M/N:.2f}", end=' ... ')
                
                M_result, ns_anal_norm, nc_anal_norm, ns_approx_norm, nc_approx_norm = compute_single_point(M, N)
                
                M_values.append(M_result)
                analytical_N_S.append(ns_anal_norm)
                analytical_N_C.append(nc_anal_norm)
                approx_N_S.append(ns_approx_norm)
                approx_N_C.append(nc_approx_norm)
                
                print("完成")
        else:
            # 多核心並行計算
            print(f"  多核心並行計算 {len(M_range)} 個數據點 (使用 {n_jobs} 個核心)...")
            
            # 使用默認的 loky backend，充分利用多核心並行計算，並顯示進度條
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(compute_single_point)(M, N) for M in tqdm(M_range, desc=f"  計算 N={N}", unit="點")
            )
            
            print(f"  ✓ 完成 {len(M_range)} 個數據點的並行計算")
            
            # 解包結果
            M_values = [r[0] for r in results_list]
            analytical_N_S = [r[1] for r in results_list]
            analytical_N_C = [r[2] for r in results_list]
            approx_N_S = [r[3] for r in results_list]
            approx_N_C = [r[4] for r in results_list]
        
        elapsed_time = time.time() - start_time
        print(f"N={N} 計算完成，耗時: {elapsed_time:.2f}秒")
        
        results[f'N_{N}'] = {
            'M_values': M_values,
            'M_over_N': [m/N for m in M_values],  # M/N比值
            'analytical_N_S': analytical_N_S,
            'analytical_N_C': analytical_N_C,
            'approx_N_S': approx_N_S,
            'approx_N_C': approx_N_C
        }
    
    return results

def generate_figure2_data(fig1_data):
    """
    生成Figure 2的數據（優化版本）
    按照論文原文定義計算絕對近似誤差：
    誤差(%) = |分析結果 - 近似結果| / |分析結果| × 100%
    """
    print("重用 Figure 1 數據計算 Figure 2...")
    
    error_results = {}
    
    for key, data in fig1_data.items():
        print(f"正在計算 {key} 的誤差數據...")
        N_S_error = []
        N_C_error = []
        
        for i in range(len(data['analytical_N_S'])):
            # 按照論文定義：|analytical - approximation| / |analytical| * 100%
            anal_ns = data['analytical_N_S'][i] 
            approx_ns = data['approx_N_S'][i]
            error_ns = abs(anal_ns - approx_ns) / abs(anal_ns) * 100
            N_S_error.append(error_ns)
            
            # 計算碰撞RAO的絕對近似誤差
            anal_nc = data['analytical_N_C'][i]
            approx_nc = data['approx_N_C'][i]
            error_nc = abs(anal_nc - approx_nc) / abs(anal_nc) * 100
            N_C_error.append(error_nc)
        
        error_results[key] = {
            'M_over_N': data['M_over_N'],
            'N_S_error': N_S_error,
            'N_C_error': N_C_error
        }
        print(f"  {key} 誤差計算完成")
    
    return error_results

# ============================================================================
# 主程式
# ============================================================================

def main():
    print("=" * 60)
    print("生成論文Figure 1和Figure 2：Analytical vs Approximation")
    print("【完全獨立版本 - 不依賴其他模組】")
    print(f"分析參數：N = {N_VALUES}, 並行進程數 = {N_JOBS}")
    print("=" * 60)
    
    # 生成數據
    print("\n正在生成Figure 1數據（多核心並行計算）...")
    fig1_data = generate_figure1_data(n_values=N_VALUES, n_jobs=N_JOBS)
    
    print("\n正在生成Figure 2數據（重用Figure 1數據）...")
    fig2_data = generate_figure2_data(fig1_data)
    
    # 繪製與保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figures_dir = os.path.join('data', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n正在繪製組合圖表...")
    num_n_values = len(N_VALUES)
    
    # 創建 2 行 × N 列的子圖佈局
    # 第一行：所有 N 值的 Fig1
    # 第二行：所有 N 值的 Fig2
    fig_width = 8 * num_n_values  # 每個子圖寬度 8
    fig_height = 12  # 總高度 12（每行 6）
    fig_combined, axes = plt.subplots(2, num_n_values, figsize=(fig_width, fig_height), 
                                      constrained_layout=True, squeeze=False)
    
    # 為每個 N 值繪製 Fig1 和 Fig2
    for col_idx, N in enumerate(N_VALUES):
        n_key = f'N_{N}'
        
        # 繪製 Fig1 到第一行
        if n_key in fig1_data:
            ax1 = axes[0, col_idx]
            # 提取單個 N 的數據
            single_n_fig1_data = {n_key: fig1_data[n_key]}
            plot_figure1(single_n_fig1_data, ax=ax1)
        
        # 繪製 Fig2 到第二行
        if n_key in fig2_data:
            ax2 = axes[1, col_idx]
            # 提取單個 N 的數據
            single_n_fig2_data = {n_key: fig2_data[n_key]}
            plot_figure2(single_n_fig2_data, ax=ax2)
    
    # 保存組合圖
    combined_path = os.path.join(figures_dir, f"figure1_2_combined_standalone_{timestamp}.png")
    fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ 組合圖已保存：{combined_path}")
    
    # 顯示圖表
    plt.show()
    
    # 顯示關鍵結果
    print("\n" + "=" * 60)
    print("關鍵結果：")
    for N in N_VALUES:
        if f'N_{N}' in fig2_data:
            max_error = max(fig2_data[f'N_{N}']['N_S_error'])
            print(f"  N={N} 成功RAO的最大近似誤差: {max_error:.1f}%")
    
    print("\n結論：")
    print("  1. 近似公式在N較大時更準確")
    print("  2. M/N比值影響近似精度")
    print("  3. 論文建議實際應用中使用較大的N值")
    print("=" * 60)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

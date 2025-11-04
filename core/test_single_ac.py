#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試腳本 1: 執行一次最小單元（Single AC 的 One-Shot Random Access）

功能：
- 只執行 1 個接入周期（AC）的隨機接入
- 展示最基本的 ALOHA 系統行為
- 快速測試單次隨機接入的結果

使用方法：
    python test_single_ac.py
"""

import sys
import os
import numpy as np

# 從同目錄的 simulation 模組引用核心函數
from simulation import simulate_one_shot_access_single_sample

# ============================================================================
# 配置參數
# ============================================================================
M = 100  # 設備總數
N = 40   # RAO 數量

print("=" * 70)
print("【測試 1】執行一次最小單元 - Single AC 的 One-Shot Random Access")
print("=" * 70)
print(f"參數配置:")
print(f"  - 設備數 M = {M}")
print(f"  - RAO 數 N = {N}")
print(f"  - 負載比 M/N = {M/N:.2f}")
print("=" * 70)

# 執行一次最小單元模擬
print("\n開始執行...")
success_raos, collision_raos, idle_raos = simulate_one_shot_access_single_sample(M, N)

# 顯示結果
print("\n" + "=" * 70)
print("【模擬結果】單個接入周期（AC）的統計")
print("=" * 70)
print(f"✅ 成功的 RAO 數量:  {success_raos:3d} / {N}  ({success_raos/N*100:.1f}%)")
print(f"❌ 碰撞的 RAO 數量:  {collision_raos:3d} / {N}  ({collision_raos/N*100:.1f}%)")
print(f"⭕ 空閒的 RAO 數量:  {idle_raos:3d} / {N}  ({idle_raos/N*100:.1f}%)")
print("-" * 70)
print(f"📊 總計檢查:        {success_raos + collision_raos + idle_raos} / {N}")
print("=" * 70)

# 設備層面的統計
print("\n【設備層面統計】")
print(f"✅ 成功接入的設備數: {success_raos} 個")
print(f"❌ 接入失敗的設備數: {M - success_raos} 個")
print(f"📈 接入成功率:       {success_raos/M*100:.2f}%")
print("=" * 70)

# 說明
print("\n【結果說明】")
print("1. 成功的 RAO = 恰好 1 個設備選擇的 RAO（該設備接入成功）")
print("2. 碰撞的 RAO = ≥2 個設備選擇的 RAO（所有設備都失敗）")
print("3. 空閒的 RAO = 0 個設備選擇的 RAO（浪費的資源）")
print("4. 失敗的設備需要在下一個 AC 重試（本測試只模擬 1 個 AC）")
print("=" * 70)

print("\n✨ 測試完成！")
print("💡 提示: 多次運行此腳本，每次結果會略有不同（因為是隨機選擇）\n")

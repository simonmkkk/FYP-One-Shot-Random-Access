# 有限用戶多通道時隙ALOHA系統建模與估計

## 📚 項目簡介

本項目實現了論文 **"Modeling and Estimation of One-Shot Random Access for Finite-User Multichannel Slotted ALOHA Systems"** (IEEE Communications Letters, 2012) 中的數學模型和模擬系統。

該研究用於評估 3GPP LTE 中機器類型通信 (MTC) 的群組尋呼性能。

## 🏗️ 項目架構

### 整體結構

```
FYP-Code/
│
├── 📄 main.py                           # ⭐ 主程序入口（單點模擬 & 參數掃描）
├── 📄 README.md                         # 項目說明文檔
├── 📄 requirements.txt                  # Python 依賴清單
├── 📄 pyproject.toml                    # 項目元數據配置
│
├── 📁 analysis/                         # 🧮【分析層】數學與理論計算
│   ├── __init__.py
│   ├── formulas.py                      # ⭐【核心數學】論文公式1-10
│   ├── theoretical.py                   # ⭐【理論計算】多AC迭代計算
│   └── metrics.py                       # 【統計工具】性能指標計算
│
├── 📁 core/                             # 🎯【模擬層】隨機接入模擬
│   ├── __init__.py
│   └── simulation.py                    # ⭐【模擬引擎】3層模擬結構
│
├── 📁 visualization/                    # 📊【可視化層】圖表生成
│   ├── __init__.py
│   └── plotting.py                      # ⭐【繪圖函數】Figure 1-5生成
│
├── 📁 utils/                            # 🛠️【工具層】通用函數
│   ├── __init__.py
│   └── file_io.py                       # 【文件操作】CSV讀寫
│
├── 📁 scripts/                          # 📜【腳本層】獨立執行腳本
│   ├── generate_figure1_figure2_analytical.py
│   └── generate_figure1_figure2_simulation.py
│
├── 📁 docs/                             # 📚【文檔層】論文資料
│   ├── Sim7原論文.md
│   └── FYP-Paper.pdf
│
└── 📁 data/                             # �【數據層】結果存儲
    ├── figures/                         # 生成的圖表
    │   └── paper_reproductions/         # 論文重現結果
    │
    └── results/                         # 模擬結果CSV
        └── latest/                      # 最新結果
```

### 系統架構流程

```
【配置層】
    ↓
  main.py
    │
    ├─→ 【理論計算路徑】          【模擬計算路徑】
    │       ↓                           ↓
    │   theoretical.py             simulation.py
    │       ↓                           ↓
    │   理論值                      模擬結果
    │       \                         /
    │        \                       /
    │         → metrics.py ←
    │              ↓
    │         統計指標
    │              ↓
    │         plotting.py
    │              ↓
    │         圖表輸出
    │              ↓
    └─→    file_io.py
              ↓
           CSV存儲
```

### 核心模組詳解

#### **`analysis/`** - 分析與計算層

- **`formulas.py`** - 論文公式實現
  - 📐 實現論文中的10個關鍵公式
  - 🧮 分為6層結構：輔助工具、精確公式(1-3)、近似公式(4-5)、迭代公式(6-7)、性能指標(8-10)、工具函數
  - 🔒 使用LRU快取優化，避免重複計算
  - ⚡ 精確計算O(M²N²)，近似計算O(1)

- **`theoretical.py`** - 理論計算
  - 📐 使用公式6-10計算理論性能指標
  - 🔄 實現多AC的迭代遞推過程
  - 📊 提供理論基準值供模擬對比

- **`metrics.py`** - 統計工具
  - 📊 計算模擬結果的平均值和95%置信區間
  - 🔍 特殊處理Ta指標（只計算有效樣本）
  - 📈 提供不確定性估計

#### **`core/`** - 模擬層

- **`simulation.py`** - 隨機模擬引擎
  - 🎲 第1層：原子操作 - 單次隨機接入
  - 📊 第2層：多樣本統計 - Figure 1&2用
  - 🔄 第3層：群組尋呼 - Figure 3-5用（I_max個AC迭代）
  - ⚙️ 支持多進程並行加速（joblib）

#### **`visualization/`** - 輸出層

- **`plotting.py`** - 圖表生成
  - 📈 生成論文Figure 1-5
  - 🎨 統一的matplotlib配置（中文支持）
  - 📊 自動添加誤差棒和統計信息

#### **`utils/`** - 工具層

- **`file_io.py`** - 文件操作
  - 💾 模擬結果導出為CSV
  - 📝 保存模擬參數和統計結果
  - 📂 自動目錄管理和時間戳

### 數據流向示例

#### 模擬 → 統計 → 可視化

```
【simulation.py】100次模擬
  ├─ 結果: (100, 3) 矩陣
  │  ├─ [:, 0] PS值 (接入成功率)
  │  ├─ [:, 1] Ta值 (平均延遲)
  │  └─ [:, 2] PC值 (碰撞概率)
         ↓
【metrics.py】統計計算
  ├─ mean_ps=0.85, ci_ps=0.024
  ├─ mean_ta=1.24, ci_ta=0.089
  └─ mean_pc=0.15, ci_pc=0.031
         ↓
【plotting.py】繪製圖表
  └─ data/figures/*.png
```

#### 理論計算

```
【輸入】M=100, N=40, I_max=10
         ↓
【formulas.py】
  ├─ 公式6: NS,i = Ki·e^(-Ki/N)
  ├─ 公式7: Ki+1 = Ki·(1-e^(-Ki/N))
  └─ 公式5: NC,i = N·(1-e^(-Ki/N)(1+Ki/N))
         ↓
【theoretical.py】循環迭代
  └─ 計算每個AC的NS,i, NC,i, Ki+1
         ↓
【formulas.py】聚合計算
  ├─ 公式8: PS = ΣNS,i / M
  ├─ 公式9: Ta = Σi·NS,i / ΣNS,i
  └─ 公式10: PC = ΣNC,i / (I_max·N)
         ↓
【輸出】PS=0.87, Ta=1.19, PC=0.12
```

## 🚀 快速開始

### 環境要求

- Python >= 3.11
- 依賴套件詳見 `requirements.txt`

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 運行模擬

#### 1. 單點模擬

```bash
python main.py
```

默認配置：M=100, N=40, I_max=10, NUM_SAMPLES=100

#### 2. 參數掃描（生成 Figure 3-5）

修改 `main.py` 中的配置：

```python
RUN_MODE = 'scan'        # 切換到掃描模式
SCAN_PARAM = 'N'         # 掃描參數：'N', 'M', 或 'I_max'
SCAN_RANGE = range(5, 46, 1)  # 掃描範圍
```

#### 3. 生成論文 Figure 1 & 2

```bash
# 分析模型版本（精確值 vs 近似公式）
python scripts/generate_figure1_figure2_analytical.py

# 模擬驗證版本（模擬值 vs 近似公式）
python scripts/generate_figure1_figure2_simulation.py
```

### 性能配置

在 `main.py` 中可調整以下參數以優化性能：

```python
M = 100           # 設備總數（增大→更多計算）
N = 40            # RAO數量（增大→更稀疏接入）
I_max = 10        # AC周期（增大→更多迭代）
NUM_SAMPLES = 100 # 樣本數（增大→更準確，更慢）
NUM_WORKERS = 16  # 並行進程（視CPU核心而定）
```

**提示：** 若模擬過慢，可降低 NUM_SAMPLES 或增加 NUM_WORKERS（根據CPU核心數）

## 📊 論文公式實現

本項目完整實現了論文中的所有關鍵公式：

| 層級 | 公式 | 描述 | 類型 | 複雜度 |
|------|------|------|------|--------|
| 1 | 輔助 | 整數分割生成 | 工具 | O(n^k) |
| 2 | (1) | 碰撞概率分佈 pk(M,N) | 精確 | O(M²N²) |
| 2 | (2) | 碰撞RAO數量 NC,1 | 精確 | O(M²N²) |
| 2 | (3) | 成功RAO數量 NS,1 | 精確 | O(M²N²) |
| 3 | (4) | 成功RAO近似公式 | 近似 | O(1) |
| 3 | (5) | 碰撞RAO近似公式 | 近似 | O(1) |
| 4 | (6) | 每個AC的成功設備數 | 迭代 | O(I_max) |
| 4 | (7) | 下一個AC的競爭設備數 | 迭代 | O(I_max) |
| 5 | (8) | 接入成功概率 PS | 聚合 | O(I_max) |
| 5 | (9) | 平均接入延遲 Ta | 聚合 | O(I_max) |
| 5 | (10) | 碰撞概率 PC | 聚合 | O(I_max) |

**特點：**

- ✅ 精確公式(1-3)使用LRU快取優化，避免重複計算
- ⚡ 近似公式(4-5)提供O(1)快速計算，適合大規模參數掃描
- 🔄 迭代公式(6-7)實現多AC遞推計算
- 📊 性能指標公式(8-10)提供最終評估

### 公式說明

#### 精確公式 (1-3)

```
pk(M, N1) = (k個碰撞RAO的配置方式數) / N1^M
NC,1 = Σ(k=1 to ⌊M/2⌋) k × pk(M,N1)
NS,1 = Σ(k=0 to ⌊M/2⌋) E[成功RAO | k碰撞] × pk(M,N1)
```

#### 近似公式 (4-5)

```
NS,1 ≈ M·e^(-M/N)
NC,1 ≈ N·(1 - e^(-M/N)·(1 + M/N))
```

#### 迭代公式 (6-7)

```
NS,i = Ki·e^(-Ki/Ni)                    # 第i個AC的成功設備
Ki+1 = Ki·(1 - e^(-Ki/Ni))              # 下一個AC的競爭設備
```

#### 性能指標 (8-10)

```
PS = ΣNS,i / M                          # 接入成功概率
Ta = Σi·NS,i / ΣNS,i                    # 平均接入延遲
PC = ΣNC,i / (I_max·N)                  # 碰撞概率
```

## � 依賴關係與數據流

### 模組依賴圖

```
main.py (中心控制)
  ├─ analysis/
  │   ├─ formulas.py (10個論文公式的實現)
  │   ├─ theoretical.py (多AC迭代計算)
  │   └─ metrics.py (統計聚合)
  ├─ core/
  │   └─ simulation.py (蒙特卡洛模擬)
  ├─ visualization/
  │   └─ plotting.py (圖表生成)
  └─ utils/
      └─ file_io.py (CSV數據持久化)
```

### 執行流程

#### 流程 1：單次模擬 (生成 Figure 1-2)

```
main.py: run_single_simulation()
  ↓
simulation.py: simulate_one_shot_access_multi_samples()
  ├─ 調用: simulate_one_shot_access_single_sample() × samples
  └─ 輸出: [success_prob, collision_prob]
  ↓
metrics.py: calculate_performance_metrics()
  ├─ 輸入: 所有樣本結果
  └─ 輸出: PS, Ta, PC (平均值±置信區間)
  ↓
plotting.py: plot_figure1(), plot_figure2()
  ├─ 輸入: PS, Ta, PC 數值
  └─ 輸出: Figure_1.png等
  ↓
file_io.py: save_single_results_to_csv()
  └─ 輸出: simulation_results_*.csv
```

#### 流程 2：理論計算 (生成 Figure 3-5)

```
theoretical.py: theoretical_calculation()
  ├─ 循環: AC i=1 to I_max
  │   ├─ 調用: formulas.paper_formula_6/7_*()
  │   └─ 計算: NS,i, Ki+1
  ├─ 調用: formulas.paper_formula_8/9/10_*()
  └─ 輸出: PS_theory, Ta_theory, PC_theory
  ↓
plotting.py: plot_figure3/4/5()
  └─ 輸出: Figure_3.png等
```

#### 流程 3：參數掃描

```
main.py: run_parameter_scan()
  ├─ 循環: 所有 (M, N, I_max) 組合
  │   ├─ 調用: simulation或theoretical
  │   └─ 記錄: (參數, PS, Ta, PC)
  └─ 輸出: scan_results_*.csv
```

## �📈 生成的圖表

本項目可重現論文中的所有圖表：

- **Figure 1**: 分析模型 vs 近似公式（NS,1/N 和 NC,1/N）
- **Figure 2**: 近似公式的絕對誤差分析
- **Figure 3**: 接入成功概率 vs N
- **Figure 4**: 平均接入延遲 vs N  
- **Figure 5**: 碰撞概率 vs N

圖表保存位置：`data/figures/paper_reproductions/`

## 🏗️ 詳細架構說明

### 核心模組

#### analysis/formulas.py (821 行)

**責任：** 實現論文中所有的數學公式

- **6層結構：**
  - Layer 1: 工具函數 (`generate_partitions()`, `_compute_configuration_ways()`)
  - Layer 2-3: 精確公式 (1-3)，複雜度 O(M²N²)，使用LRU快取優化
  - Layer 3-4: 近似公式 (4-5)，複雜度 O(1)
  - Layer 4-5: 迭代公式 (6-7)，複雜度 O(I_max)
  - Layer 5: 聚合公式 (8-10)，計算最終性能指標
  - Layer 6: 分析工具函數 (`relative_error_percentage()`, `confidence_interval_95()`)

#### core/simulation.py (293 行)

**責任：** 蒙特卡洛模擬ALOHA隨機接入過程

- **3層模擬結構：**
  - Layer 1: `simulate_one_shot_access_single_sample()` - 單個樣本原子操作
  - Layer 2: `simulate_one_shot_access_multi_samples()` - 生成Figure 1-2的多樣本統計
  - Layer 3: `simulate_group_paging_multi_samples()` - 多AC迭代模擬，生成Figure 3-5

#### analysis/theoretical.py

**責任：** 實現理論計算方法（公式6-10的迭代應用）

- 循環K₁→K_{I_max}計算每個AC的設備分配
- 調用formulas.py的公開接口函數
- 輸出聚合性能指標

#### analysis/metrics.py (28 行)

**責任：** 統計聚合層，計算性能指標的平均值和置信區間

- 特殊T_a處理：只篩選T_a≥0的樣本
- 輸出：PS, Ta, PC (平均值 ± 95%置信區間)

#### visualization/plotting.py (566 行)

**責任：** 圖表生成，重現論文中的Figure 1-5

- 使用matplotlib生成出版級別的圖表
- 支持靈活的N值範圍配置

#### utils/file_io.py

**責任：** CSV數據持久化

- `save_single_results_to_csv()` - 保存單次模擬結果
- `save_scan_results_to_csv()` - 保存參數掃描結果

#### main.py (222 行)

**責任：** 中心控制器和配置管理

- `run_single_simulation()` - 執行單次模擬
- `run_parameter_scan()` - 執行參數掃描
- 統一的參數配置管理

### 性能優化

| 優化策略 | 實現 | 效果 |
|---------|------|------|
| LRU快取 | `@lru_cache(maxsize=128)` 在`_paper_formula_*_impl()` | 避免精確公式重複計算，加速50-100x |
| 近似公式 | Poisson近似O(1)計算 | 替代O(M²N²)精確計算 |
| 並行計算 | `joblib.Parallel` 並行樣本生成 | 線性加速(取決於CPU核心數) |
| 增量計算 | 迭代公式避免重複計算聚合 | 參數掃描時減少計算量 |

## ⚙️ 性能配置

在 `main.py` 中可調整：

```python
NUM_SAMPLES = 100      # 樣本數（論文使用 10^7）
NUM_WORKERS = 16       # 並行進程數（建議設為 CPU 核心數）
```

## 🔧 故障排除

### 常見問題

| 問題 | 原因 | 解決方案 |
|------|------|---------|
| 運行速度慢 | NUM_SAMPLES過大或NUM_WORKERS設置不當 | 減少樣本數或增加並行進程 |
| 內存不足 | 大量並行進程 | 減少NUM_WORKERS值 |
| T_a計算異常 | 未過濾負值 | metrics.py已自動過濾T_a<0 |
| 精確公式結果不穩定 | M/N比例不合理 | 確保 M << N 或使用近似公式 |
| 圖表生成失敗 | 缺少matplotlib | 執行 `pip install matplotlib` |

### 驗證安裝

```bash
python -c "import numpy, matplotlib, joblib; print('All dependencies installed')"
```

### 調試模式

在 `main.py` 中設置 `DEBUG=True`，查看詳細計算過程：

```python
DEBUG = True  # 啟用詳細日誌輸出
```

## 📚 學習路徑

**初學者：** 理解項目結構

1. 閱讀README的"項目概述"和"系統架構"
2. 查看 `main.py` 的高級控制流程
3. 檢查 `data/results/latest/` 中的示例輸出

**進階：** 理解公式實現

1. 學習 `analysis/formulas.py` 的6層結構
2. 比較精確公式(1-3)和近似公式(4-5)的差異
3. 理解LRU快取對性能的影響

**專家：** 擴展功能

1. 修改 `analysis/theoretical.py` 實現新的迭代方案
2. 在 `visualization/plotting.py` 添加自定義圖表
3. 為 `analysis/metrics.py` 添加新的統計指標

### 代碼導讀

**第一次運行：** 執行單次模擬查看流程

```bash
python main.py
```

**參數掃描：** 發現最優配置

```bash
python main.py --scan
```

**生成圖表：** 重現論文結果

```bash
python scripts/generate_figure1_figure2_analytical.py  # Figure 1-2
python scripts/generate_figure1_figure2_simulation.py  # 模擬版本
```

## 📖 參考文獻

Wei, C. H., Cheng, R. G., & Tsao, S. L. (2012). Modeling and Estimation of One-Shot Random Access for Finite-User Multichannel Slotted ALOHA Systems. *IEEE Communications Letters*, 16(8), 1196-1199.

## 📝 作者

Simon Mak - FYP Project

## 📄 授權

本項目僅用於學術研究目的。

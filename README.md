# TTMATCH — 桌球下一拍預測

Kaggle 課堂競賽 [Introduction to Data Science Competition — TTMATCH](https://www.kaggle.com/competitions/introduction-to-data-science-competition-ttmatch) 的程式碼與報告。

做法：整理並比較 14 個樹模型與序列模型，在同一套交叉驗證（cross-validation，CV；依比賽分組的 5-fold，即把比賽分成 5 份輪流當驗證集）下淘汰、集成（加權平均多個模型的預測機率）；另外自行設計「比分推移」特徵（見第一節）。

## 成績

計入成績的繳交：2026-10-29 06:59（台灣時間）上傳，由本程式碼執行 `python arena.py` 產生。

| 排行榜 | test 比例 | 分數 | 名次 |
|---|---:|---:|---|
| public | 約 67% | 0.41744 | 截止時第 24 名 |
| **private（最終成績）** | 約 33% | **0.39028** | **第 17 名／57 隊** |

## 任務與流程

給定一個 rally（從發球到得分的一段來回擊球）的前 k 拍（稱為前綴），預測第 k+1 拍：

| 目標 | 類別 | 指標 | 權重 |
|---|---|---|---|
| `actionId` 下一拍球種（下稱 action） | 0–14 | Macro-F1（各類別 F1 的平均，每類一視同仁） | 0.4 |
| `pointId` 下一拍落點（下稱 point） | 0–9（0 = 沒落在桌面，rally 結束） | Macro-F1 | 0.4 |
| `serverGetPoint` 發球方是否得分（下稱 winner） | 機率 | AUC（ROC 曲線下面積，衡量能否分辨勝負） | 0.2 |

```
train.csv / test.csv
  ├─ 1. 前處理：每個前綴 → 一筆「預測下一拍」的樣本，建立 50 個特徵 + 比分推移特徵
  ├─ 2. 訓練 14 個方法（run_method.py）：樹模型、序列模型，皆用 5-fold 產生 OOF（每筆樣本由沒看過它的 fold 預測的機率）與 test 預測
  ├─ 3. 評估與集成（arena.py）：依 test 前綴長度抽樣評估 → 調 Macro-F1 決策規則 → 貪婪集成
  └─ 4. 輸出 outputs/submission.csv
```

名詞：**Time2Vec** = 把拍序位置編成可學習的週期性向量；**FiLM** = 用額外特徵對隱藏層做逐維縮放與平移。

## 一、資料前處理（`ttmatch/data.py`、`ttmatch/features.py`）

### 樣本

每一列是一拍；「看完前 k 拍、預測第 k+1 拍」為一筆樣本。

| 名稱 | 內容 | 用途 |
|---|---|---|
| `ap_base` | train 每個 rally 的每個前綴 | 訓練、CV 評估 |
| `ap_aug` | test 內部轉移（test 中已可見的第 i 拍 → 第 i+1 拍） | 樹模型 action / point 的額外訓練資料 |
| `test_pred` | 每個 test rally 的最後一個可見拍 | 要預測的列 |

下一拍是發球（球種 15–18）的列先去掉，所以球種只預測 15 類。

### 特徵（只用前 k 拍的資訊）

- **類別 23 個**：原始逐拍欄位取最近 3 拍與發球拍，加上性別、局數、**下一拍擊球者的選手 ID**；未發生的拍填 50。
- **數值 27 個（自行計算）**：比分、比分差、是否局點／deuce；累計攻擊／控制／防守次數、累計落點 0／1–3／7–9 次數、平均力道、攻擊比例、最近兩拍是否攻擊；前綴長度、擊球者是否發球方。
- **選手 target encoding（TE）**：`P(下一拍球種 | 下一拍擊球者)` 等，即「每位選手過去的打法統計」。樹模型有三組，**只用在 action**（point、winner 都不加）；序列模型一組。用於 `lgbm`、`lgbm_te20` 與 `gru`、`transformer`、`gru_t2v_film`、`gru_mirror`。

### 比分推移特徵（只用於 winner）

同一局**下一個可見 rally 的開局比分**，透露這段期間雙方各得幾分；若一方沒得分，就能確定本 rally 的勝方。在 train 上以同局實際的下一個 rally 驗證：14,995 個 rally 中，有下一個 rally 且其中一方得分為 0（可判定勝方）的有 13,718 個，推得的勝方與答案全部一致。這項資訊來自 test 一次提供完整資料；不使用時 overall 為 **0.3790**，對應指令：`python arena.py --exclude lgbm_prog cat_prog --tag noprog`。

- 特徵：是否有下一個可見 rally、間隔分數、雙方各得幾分、發球方得分比例。
- **train 模擬 test**：test 相鄰可見 rally 的間隔多為 2–4；train 依同樣的間隔分佈抽「下一個可見 rally」，讓兩邊特徵分佈一致。
- **一致性檢查**：選手相同、`rally_id` 與比分對得上才使用。test 的 match 340–354（262 個 rally）資料異常，winner 改用不含比分推移的集成。

## 二、模型選擇（`run_method.py`）

| 方法 | 類型 | 目標 | 說明 |
|---|---|---|---|
| `lgbm` | LightGBM | 三者 | action 加選手 TE + test 內部轉移擴充 |
| `lgbm_note` | LightGBM | 三者 | 不加 TE（對照）；仍有「選手 ID」類別特徵 |
| `lgbm_te20` | LightGBM | action | target encoding 加強平滑 |
| `lgbm_noplayer` | LightGBM | action、point | 不用任何選手資訊（對照） |
| `xgb` | XGBoost | 三者 | 特徵同 `lgbm_note` |
| `cat` | CatBoost | 三者 | 特徵同 `lgbm_note` |
| `lgbm_prog`、`cat_prog` | LightGBM、CatBoost | winner | 加入比分推移特徵 |
| `gru`、`transformer` | GRU、因果 Transformer | 三者 | 多任務序列模型 |
| `gru_t2v_film` | GRU | 三者 | 加入時間編碼（Time2Vec）與特徵調變（FiLM） |
| `gru_mirror` | GRU | 三者 | 左右鏡像資料增強 |
| `gru_pemb`、`transformer_pemb` | GRU、Transformer | 三者 | 選手 ID 嵌入取代 target encoding |

`lgbm` 與 `lgbm_note` 的差別只在 action 的 TE，point、winner 是同一個模型，所以結果表兩者的 point／winner 數字相同。

選擇理由：樹模型擅長類別與手工特徵，序列模型直接讀擊球序列，錯誤型態不同，集成可互補。

## 三、訓練過程

所有方法共用**依比賽分組的 5-fold**。每個方法輸出 OOF 機率與 5 個 fold 對 test 的平均預測，存成 `outputs/oof/<方法>.npz`。

**樹模型**（`ttmatch/methods/trees.py`）
- 三個目標分開訓練；action、point 的類別權重 ∝ 頻率^−0.5，照顧稀有類別；winner 不加權重，也不加 `ap_aug`。
- action / point 的訓練資料加入 `ap_aug`（權重 ×5），讓模型看到 test 選手的打法。×5 與下方序列模型的 12 epoch、1 seed 都是經驗設定，沒有做系統性搜尋；序列模型受限於本機 CPU 時間。
- LightGBM：lr 0.05、num_leaves 63（winner 47）；XGBoost：lr 0.06、depth 7；CatBoost：lr 0.12、depth 6。皆以驗證 fold 做 early stopping（驗證分數不再進步就停止訓練）。

**序列模型**（`ttmatch/methods/nn.py`）
- 讀入整個 rally，在每個位置預測下一拍的球種、落點與勝負（因果遮罩）。
- 輸入：8 個類別欄位嵌入 + 6 個數值 + 拍序位置；皆 2 層、hidden 160。
- 損失 = 0.4 球種 + 0.4 落點 + 0.2 勝負（同評分權重）；球種、落點的類別權重 ∝ 頻率^−0.5，用全部 `ap_base` 的標籤頻率計算（`nn.py:98–99`）；Adam lr 2e-3、cosine 排程、batch 128。
- 訓練資料包含 test rally 已可見的部分。繳交版本：**1 seed × 12 epoch**，不做 early stopping。

## 四、評估與集成（`arena.py`）

1. **Macro-F1 決策規則**：Macro-F1 對每類一視同仁，argmax（取機率最大的類別）會忽略稀有類別；預測 = argmax_c [α·log p_c + w_c]，調整溫度 α 與類別偏置 w_c。
2. **貪婪集成**：從最佳單一方法開始逐一加入其他方法（權重 0.1–0.6），分數進步 ≥ 0.0005 才保留；挑選時用快速版決策調參，最後才用完整調參評分。
3. 未採用：分前綴長度的決策規則、規則修正（CV 增益 < 0.0005）。

驗證設計：

| 目標 | 做法 |
|---|---|
| 驗證與訓練的比賽互不重疊 | 依比賽分組的 5-fold（test 與 train 的比賽也不重疊） |
| TE 不洩漏標籤 | 訓練列用巢狀 OOF（在訓練 fold 內再切 fold 計算）；驗證列與 test 用訓練 fold 擬合的編碼 |
| 驗證前綴長度分佈對齊 test（多只看得到 1–3 拍） | 依 test 的前綴長度分佈抽樣驗證前綴，抽到超過 rally 長度的 k 截到最後一拍，重複 5 次（k=1 因此偏高，見第七節第 2 點） |
| 決策規則（α、類別偏置）不過配 | 跨半調參：一半比賽調參、另一半評分，再互換 |

## 五、結果

**單一方法（CV）**

| 方法 | action argmax | action 調整後 | point argmax | point 調整後 | winner AUC | overall |
|---|---:|---:|---:|---:|---:|---:|
| `transformer` | 0.3798 | **0.3834** | 0.1801 | 0.1889 | 0.6787 | **0.3647** |
| `lgbm_note` | 0.3674 | 0.3687 | 0.1901 | **0.1968** | 0.6652 | 0.3592 |
| `cat` | 0.3646 | 0.3672 | 0.1767 | 0.1864 | 0.6885 | 0.3591 |
| `xgb` | 0.3670 | 0.3641 | 0.1822 | 0.1889 | 0.6854 | 0.3583 |
| `lgbm` | 0.3567 | 0.3581 | 0.1901 | 0.1968 | 0.6652 | 0.3550 |
| `transformer_pemb` | 0.3638 | 0.3661 | 0.1852 | 0.1851 | 0.6632 | 0.3531 |
| `gru` | 0.3712 | 0.3699 | 0.1801 | 0.1894 | 0.6463 | 0.3530 |
| `gru_t2v_film` | 0.3716 | 0.3663 | 0.1813 | 0.1876 | 0.6497 | 0.3515 |
| `gru_pemb` | 0.3663 | 0.3644 | 0.1881 | 0.1938 | 0.6261 | 0.3485 |
| `gru_mirror` | 0.3584 | 0.3543 | 0.1754 | 0.1835 | 0.6452 | 0.3442 |
| `lgbm_te20` | 0.3556 | 0.3584 | — | — | — | — |
| `lgbm_noplayer` | 0.3124 | 0.3116 | 0.1725 | 0.1797 | — | — |
| `cat_prog` | — | — | — | — | **0.8335** | — |
| `lgbm_prog` | — | — | — | — | 0.8253 | — |

決策規則調整後，action F1 的變化在 −0.005 到 +0.004 之間（12 個方法中 6 個下降）；point F1 多提升 0.006–0.010（`transformer_pemb` 持平）；overall 用調整後分數。

**最終集成（繳交版本）**

| 目標 | 方法（權重） | CV |
|---|---|---:|
| action | transformer 0.38、lgbm_note 0.26、gru_pemb 0.20、xgb 0.16 | 0.3971 |
| point | lgbm 0.48、gru_mirror 0.32、gru_pemb 0.20 | 0.2023 |
| winner | cat_prog 0.45、xgb 0.45、transformer 0.10 | 0.8462 |
| winner（異常比賽） | cat 0.38、xgb 0.26、lgbm 0.20、transformer 0.16 | 0.6959 |
| **overall** | | **0.4090** |

**主要發現**

1. **選手身分很重要**：拿掉選手資訊，action F1 從 0.369 降到 0.312。`lgbm`、`lgbm_note` 都有選手 ID 類別特徵，已有 ID 時再加 TE，action F1 反而下降（0.3687 → 0.3581）。
2. **序列與樹模型互補**：Transformer 是最佳單一 action 模型（0.383），集成後達 0.397。
3. **比分推移特徵**讓 winner AUC 從約 0.69 升到 0.85，overall 約 +0.03（0.3790 → 0.4090）。

## 六、操作說明

### 1. 環境

開發環境：Windows 11、Python 3.13、CPU（無 GPU）。`requirements.txt` 只寫最低版本，實際版本：lightgbm 4.7.0、xgboost 3.4.1、catboost 1.2.10、torch 2.15.0（CPU）、pandas 3.1.0、numpy 2.5.4、scikit-learn 1.9.1。隨機種子固定為 42；XGBoost 與 PyTorch 在不同執行緒數或硬體上可能有微小差異。

Windows（PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

單元測試：

```bash
pip install pytest
python -m pytest tests -q
```

### 2. 資料

依序尋找：環境變數 `TTMATCH_DATA_DIR` → repo 的 `data/`（放 `train.csv`、`test.csv`、`sample_submission.csv`）→ 用 `kagglehub` 自動下載（需 `~/.kaggle/kaggle.json` 並已加入競賽）。

### 3. 訓練（重現繳交版本）

```bash
# 樹模型 8 個
python run_method.py lgbm lgbm_note lgbm_te20 lgbm_noplayer xgb cat lgbm_prog cat_prog

# 序列模型 6 個
python run_method.py transformer gru gru_t2v_film gru_mirror gru_pemb transformer_pemb --epochs 12 --seeds 1
```

| 參數 | 說明 |
|---|---|
| `--list` | 列出所有方法 |
| `--device cuda` | 用 GPU 訓練序列模型 |
| `--epochs`、`--seeds` | 預設 16、2（繳交版本 12、1） |
| `--folds` | fold 數（預設 5） |
| `--fast` | 快速測試流程，輸出 `*_fast.npz` |

10 核 CPU 上全部約 4–5 小時；快速確認流程約 5–10 分鐘：

```bash
python run_method.py lgbm_note cat_prog gru_pemb --fast --folds 2 --seeds 1 --epochs 1
python arena.py --fast
```

### 4. 集成並產生提交檔

```bash
python arena.py
```

輸出 `outputs/submission.csv` 與 `outputs/arena_report.md`（評估與集成報告），約 10–15 分鐘。輸出資料夾可用環境變數 `TTMATCH_OUT_DIR` 改掉（預設 `outputs/`）。選項：`--exclude <方法…>` 排除方法、`--tag <名稱>` 改輸出檔名。test 異常 match（340–354）的 winner 固定使用不含比分推移的集成，沒有參數可關閉。

### 檔案結構

```
arena.py                 評估、集成、產生提交檔
run_method.py            訓練單一方法
ttmatch/
  config.py              路徑與設定
  data.py                讀資料、組樣本
  features.py            特徵工程、比分推移特徵
  cv.py                  CV 切分、前綴長度抽樣
  decision.py            Macro-F1 決策規則
  metrics.py             評分
  postprocess.py         規則修正（未採用）
  methods/trees.py       樹模型
  methods/nn.py          序列模型
notebooks/               Colab 訓練序列模型（指令需改成 --epochs 12 --seeds 1）
tests/                   單元測試
```

## 七、檢討與改進方向

1. **預測分佈要和 test 對照**：樹模型用了 test 內部轉移（`ap_aug`），這些列的下一拍一定存在，使樹模型低估落點 0：繳交版樹模型對 test 的平均 P(落點 0) 約 0.11–0.13，OOF 約 0.15–0.17，只看 CV 分數看不出這個差距。截止後在 [`analysis/post-deadline`](https://github.com/ChiJiun/TTMATCH/tree/analysis/post-deadline) 分支比較 CV 與 test 的預測比例才發現並修正，該分支在截止後版本上估計此項修正約讓 overall +0.02；截止後版本的 private 為 0.401–0.403（Late Submission，不計成績）。
2. **評估抽樣對齊 test 的截斷方式**：繳交版 `sample_one_per_rally`（`ttmatch/cv.py:29`）把抽到超過 rally 長度的 k 截到最後一拍，抽樣後 k=1 佔 36.7%，test 為 27.5%。該分支依 test 的前綴長度分佈估計截斷機制，驗證前綴的抽樣與訓練權重都照這個機制，下一步是在對齊後的抽樣上重做集成選擇與決策調參。
3. **多 seed 與重複 CV**：序列模型目前 1 seed、CV 做一次，方法間 overall 差距多在 0.01 以內。多 seed、重複 CV 與依比賽成對的 bootstrap（重複抽樣估計差距的不確定性；該分支已用於版本比較）能讓這類小差距比得更穩。
4. **依類別分析**：Macro-F1 對稀有類別最敏感。下一步是用現有 OOF 做各類別 F1 與混淆矩陣；該分支的分析顯示 point 的瓶頸在落點 1、3（合計 0.8% 的資料，佔 Macro-F1 的 20%）與落點 7／8／9（F1 0.20–0.28），受落點本身的隨機性限制；下一步是從特徵與預測目標著手（例如把「下一拍是否為 rally 最後一拍（落點 0）」單獨建模），而不是加大模型。
5. **決策規則與集成的巢狀選擇**：以截止後分支的 arena 重新評估繳交版本：成員、權重、決策都只用另一半比賽挑選，得到巢狀估計 overall 0.4063（action 0.3982、point 0.1943），與 CV 0.4090 相差 0.0027。test 有 14.2% 的列沒有比分推移資訊，在 CV 上隨機挑同比例的比賽模擬這種情況，預期 overall 約 0.4055，介於 public 0.41744（約 1,236 列）與 private 0.39028（約 609 列）之間。下一步是把版本挑選也放進巢狀。

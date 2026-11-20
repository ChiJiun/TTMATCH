# TTMATCH — 桌球下一拍預測

Kaggle 課堂競賽 [Introduction to Data Science Competition — TTMATCH](https://www.kaggle.com/competitions/introduction-to-data-science-competition-ttmatch) 的程式碼與報告。

做法：實作 14 個樹模型與序列模型，在同一套 CV 下比較、淘汰、集成；另外自行設計「比分推移」特徵。

## 成績

計入成績的繳交：2026-10-29 06:59（台灣時間）上傳，由本程式碼執行 `python arena.py` 產生。

| 排行榜 | test 比例 | 分數 | 名次 |
|---|---:|---:|---|
| public | 約 67% | 0.41744 | 截止時第 24 名 |
| **private（最終成績）** | 約 33% | **0.39028** | **第 17 名／57 隊** |

## 任務與評分

給定一個 rally 的前 n−1 拍，預測：

| 目標 | 類別 | 指標 | 權重 |
|---|---|---|---|
| `actionId` 下一拍球種 | 0–14 | Macro-F1 | 0.4 |
| `pointId` 下一拍落點 | 0–9（0 = 沒落在桌面，rally 結束） | Macro-F1 | 0.4 |
| `serverGetPoint` 發球方是否得分 | 機率 | AUC | 0.2 |

## 流程總覽

```
train.csv / test.csv
  ├─ 1. 前處理：每個前綴 → 一筆「預測下一拍」的樣本，建立 50 個特徵 + 比分推移特徵
  ├─ 2. 訓練 14 個方法（run_method.py）：樹模型、序列模型，皆用依比賽分組的 5-fold 產生 OOF 與 test 預測
  ├─ 3. 評估與集成（arena.py）：依 test 前綴長度抽樣評估 → 調 Macro-F1 決策規則 → 貪婪集成
  └─ 4. 輸出 outputs/submission.csv
```

## 一、資料前處理（`ttmatch/data.py`、`ttmatch/features.py`）

### 樣本

每一列是一拍；「看完第 1..t 拍、預測第 t+1 拍」為一筆樣本。

| 名稱 | 內容 | 用途 |
|---|---|---|
| `ap_base` | train 每個 rally 的每個前綴 | 訓練、CV 評估 |
| `ap_aug` | test 中已可見的轉移（第 i 拍 → 第 i+1 拍） | 樹模型 action / point 的額外訓練資料 |
| `test_pred` | 每個 test rally 的最後一個可見拍 | 要預測的列 |

下一拍是發球（球種 15–18）的列先去掉，所以球種只預測 15 類。

### 特徵（只用第 1..t 拍的資訊）

- **類別 23 個**：性別、局數、下一拍擊球類型；倒數第 1 拍的球種／落點／正反手／力道／旋轉／擊球類型／站位，倒數第 2 拍的球種／落點／正反手／旋轉／力道，倒數第 3 拍的球種／落點；發球拍的球種、旋轉、力道、落點、正反手；**下一拍擊球者的選手 ID**。未發生的拍填 50。
- **數值 27 個**：前綴長度、擊球者是否發球方；雙方比分、比分差、是否局點／deuce；累計攻擊／控制／防守次數、累計落點 0／1–3／7–9 次數、平均力道、攻擊比例、最近兩拍是否攻擊等。
- **選手 target encoding**：`P(下一拍球種 | 下一拍擊球者)` 等（樹模型三組、序列模型一組），用巢狀 OOF 計算以避免洩漏。用於 `lgbm`、`lgbm_te20` 與 `gru`、`transformer`、`gru_t2v_film`、`gru_mirror`。

### 比分推移特徵（只用於 winner）

同一局**下一個可見 rally 的開局比分**，透露這段期間雙方各得幾分；若一方沒得分，就能確定本 rally 的勝方。在 train 上驗證，可判定的 13,718 個 rally 全部與答案一致。

- 特徵：是否有下一個可見 rally、間隔分數、雙方各得幾分、發球方得分比例。
- **train 模擬 test**：test 相鄰可見 rally 的間隔多為 2–4；train 依同樣的間隔分佈抽「下一個可見 rally」，讓兩邊特徵分佈一致。
- **一致性檢查**：選手相同、`rally_id` 與比分對得上才使用。test 的 match 340–354（262 個 rally）資料異常，不使用此特徵，winner 改用不含比分推移的集成。
- 只用到競賽提供的資料，沒有外部資料。

## 二、模型選擇（`run_method.py`）

| 方法 | 類型 | 目標 | 說明 |
|---|---|---|---|
| `lgbm` | LightGBM | 三者 | 選手 target encoding + test 內部轉移擴充 |
| `lgbm_note` | LightGBM | 三者 | 不用選手 target encoding（對照） |
| `lgbm_te20` | LightGBM | action | target encoding 加強平滑 |
| `lgbm_noplayer` | LightGBM | action、point | 不用任何選手資訊（對照） |
| `xgb` | XGBoost | 三者 | 特徵同 `lgbm_note` |
| `cat` | CatBoost | 三者 | 特徵同 `lgbm_note` |
| `lgbm_prog`、`cat_prog` | LightGBM、CatBoost | winner | 加入比分推移特徵 |
| `gru`、`transformer` | GRU、因果 Transformer | 三者 | 多任務序列模型 |
| `gru_t2v_film` | GRU | 三者 | 加入時間編碼（Time2Vec）與特徵調變（FiLM） |
| `gru_mirror` | GRU | 三者 | 左右鏡像資料增強 |
| `gru_pemb`、`transformer_pemb` | GRU、Transformer | 三者 | 選手 ID 嵌入取代 target encoding |

選擇理由：樹模型擅長類別與手工特徵，序列模型直接讀擊球序列；兩者錯誤型態不同，集成可互補。

## 三、訓練過程

所有方法共用**依比賽分組的 5-fold**（test 與 train 的比賽不重疊）。每個方法輸出 OOF 機率與 5 個 fold 對 test 的平均預測，存成 `outputs/oof/<方法>.npz`。

**樹模型**（`ttmatch/methods/trees.py`）
- 三個目標分開訓練；類別權重 ∝ 頻率^−0.5，照顧稀有類別。
- action / point 的訓練資料加入 `ap_aug`（權重 ×5），讓模型看到 test 選手的打法。
- LightGBM：lr 0.05、num_leaves 63（winner 47）；XGBoost：lr 0.06、depth 7；CatBoost：lr 0.12、depth 6。皆以驗證 fold 做 early stopping。

**序列模型**（`ttmatch/methods/nn.py`）
- 讀入整個 rally，在每個位置預測下一拍的球種、落點與勝負（因果遮罩）。
- 輸入：8 個類別欄位的嵌入 + 6 個數值 + 拍序位置；GRU／Transformer 皆 2 層、hidden 160。
- 損失 = 0.4 球種 + 0.4 落點 + 0.2 勝負（同評分權重）；Adam lr 2e-3、cosine 排程、batch 128。
- 訓練資料包含 test rally 已可見的部分。繳交版本：**1 個 seed × 12 個 epoch**。

## 四、評估與集成（`arena.py`）

1. **依 test 前綴長度評估**：test 多半只看得到 1–3 拍；每個驗證 rally 依 test 的長度分佈抽前綴，重複 5 次。
2. **Macro-F1 決策規則**：預測 = argmax_c [α·log p_c + w_c]，調整溫度 α 與類別偏置 w_c，讓稀有類別也被預測。
3. **跨半調參**：比賽分兩半，一半調參、一半評分再互換，避免高估。
4. **貪婪集成**：從最佳單一方法開始逐一加入其他方法（權重 0.1–0.6），分數進步 ≥ 0.0005 才保留。
5. 未採用：分前綴長度的決策規則、規則修正（CV 增益 < 0.0005）。

## 五、結果

### 單一方法（CV）

| 方法 | action F1 | point F1 | winner AUC | overall |
|---|---:|---:|---:|---:|
| `transformer` | **0.3834** | 0.1889 | 0.6787 | **0.3647** |
| `lgbm_note` | 0.3687 | **0.1968** | 0.6652 | 0.3592 |
| `cat` | 0.3672 | 0.1864 | 0.6885 | 0.3591 |
| `xgb` | 0.3641 | 0.1889 | 0.6854 | 0.3583 |
| `lgbm` | 0.3581 | 0.1968 | 0.6652 | 0.3550 |
| `transformer_pemb` | 0.3661 | 0.1851 | 0.6632 | 0.3531 |
| `gru` | 0.3699 | 0.1894 | 0.6463 | 0.3530 |
| `gru_t2v_film` | 0.3663 | 0.1876 | 0.6497 | 0.3515 |
| `gru_pemb` | 0.3644 | 0.1938 | 0.6261 | 0.3485 |
| `gru_mirror` | 0.3543 | 0.1835 | 0.6452 | 0.3442 |
| `lgbm_te20` | 0.3584 | — | — | — |
| `lgbm_noplayer` | 0.3116 | 0.1797 | — | — |
| `cat_prog` | — | — | **0.8335** | — |
| `lgbm_prog` | — | — | 0.8253 | — |

### 最終集成（繳交版本）

| 目標 | 方法（權重） | CV |
|---|---|---:|
| action | transformer 0.38、lgbm_note 0.26、gru_pemb 0.20、xgb 0.16 | 0.3971 |
| point | lgbm 0.48、gru_mirror 0.32、gru_pemb 0.20 | 0.2023 |
| winner | cat_prog 0.45、xgb 0.45、transformer 0.10 | 0.8462 |
| winner（異常比賽） | cat 0.38、xgb 0.26、lgbm 0.20、transformer 0.16 | 0.6959 |
| **overall** | | **0.4090** |

集成在同一批 OOF 上挑選，CV 略為樂觀。

### 主要發現

1. **選手身分很重要**：拿掉選手資訊，action F1 從 0.369 降到 0.312；直接用選手 ID 比 target encoding 好（0.3687 vs 0.3581）。
2. **序列與樹模型互補**：Transformer 是最佳單一 action 模型（0.383），集成後達 0.397。
3. **比分推移特徵**讓 winner AUC 從約 0.69 升到 0.85，overall 約 +0.03。

## 六、操作說明

### 1. 環境

開發環境：Windows 11、Python 3.13、CPU（無 GPU）。

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS / Linux：source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 資料

依序尋找：環境變數 `TTMATCH_DATA_DIR` → repo 的 `data/`（放 `train.csv`、`test.csv`、`sample_submission.csv`）→ 用 `kagglehub` 自動下載（需 `~/.kaggle/kaggle.json` 並已加入競賽）。

### 3. 訓練（重現繳交版本）

```bash
# 樹模型 8 個
python run_method.py lgbm lgbm_note lgbm_te20 lgbm_noplayer xgb cat lgbm_prog cat_prog

# 序列模型 6 個（繳交版本：12 epoch、1 seed）
python run_method.py transformer gru gru_t2v_film gru_mirror gru_pemb transformer_pemb --epochs 12 --seeds 1
```

| 參數 | 說明 |
|---|---|
| `--list` | 列出所有方法 |
| `--device cuda` | 用 GPU 訓練序列模型 |
| `--epochs`、`--seeds` | 預設 16、2（繳交版本 12、1） |
| `--folds` | fold 數（預設 5） |
| `--fast` | 快速測試流程，輸出 `*_fast.npz` |

10 核 CPU 上全部約 4–5 小時。快速確認流程可跑（約 5–10 分鐘）：

```bash
python run_method.py lgbm_note cat_prog gru_pemb --fast --folds 2 --seeds 1 --epochs 1
python arena.py --fast
```

### 4. 集成並產生提交檔

```bash
python arena.py
```

輸出 `outputs/submission.csv`（提交檔）與 `outputs/arena_report.md`（評估與集成報告），約 10–15 分鐘。選項：`--exclude <方法…>` 排除方法、`--tag <名稱>` 改輸出檔名。

### 5. 單元測試

```bash
pip install pytest
python -m pytest tests -q
```

### 6. 重現性

隨機種子固定為 42。XGBoost 與 PyTorch 在不同執行緒數或硬體上會有微小差異（分數約 ±0.001）。

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

## 七、限制與檢討

- **CV 與排行榜落差**：private 只有約 609 列，抽樣雜訊大。
- **事後發現的偏差**：`ap_aug` 的列下一拍一定存在，又和要預測的列來自同一 rally，使樹模型在 test 上幾乎不預測落點 0；CV 看不出這個問題。估計讓 point F1 少約 0.02–0.05（在後續版本上量測）。
- early stopping 用驗證 fold，OOF 分數略為樂觀（各方法一致，比較仍公平）。
- 截止後的改進放在 [`analysis/post-deadline`](https://github.com/ChiJiun/TTMATCH/tree/analysis/post-deadline) 分支，不計成績。

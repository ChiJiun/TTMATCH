# TTMATCH — 桌球下一拍預測

Kaggle 課堂競賽 [Introduction to Data Science Competition — TTMATCH](https://www.kaggle.com/competitions/introduction-to-data-science-competition-ttmatch) 的程式碼與說明文件。

本題資料源自 **AI CUP 2026 春季賽「基於時序資料之桌球戰術與結果預測競賽」**（423 隊）。這份解法的做法是：把該賽前段班公開的方法改寫成同一套程式，放在**同一套資料切分與評分**下比較，淘汰表現差的方法，再把留下來的方法集成；另外加入一個自己設計的特徵（比分推移）。

## 成績

計入成績的繳交只有一筆：2026-10-29 06:59（台灣時間）上傳的 `submission.csv`，由這個 repo 的程式碼執行 `python arena.py` 產生。

| 排行榜 | 使用的 test 比例 | 分數 | 名次 |
|---|---:|---:|---|
| public | 約 67% | 0.41744 | 截止時第 24 名 |
| **private（最終成績）** | 約 33% | **0.39028** | **第 17 名／57 隊** |

## 任務與評分

給定一個 rally 已發生的前 n−1 拍，預測：

| 目標 | 類別 | 指標 | 權重 |
|---|---|---|---|
| `actionId` 下一拍的球種 | 0–14（下一拍不會是發球） | Macro-F1 | 0.4 |
| `pointId` 下一拍的落點 | 0–9（0 代表沒有落在桌面上，也就是失誤、這一拍結束 rally） | Macro-F1 | 0.4 |
| `serverGetPoint` 發球方是否贏得這個 rally | 機率 | AUC | 0.2 |

Final score = 0.4 × action Macro-F1 + 0.4 × point Macro-F1 + 0.2 × winner AUC。

## 方法總覽

```
train.csv / test.csv
   │
   ├─ 1. 前處理：每個 rally 的每個前綴 → 一筆「預測下一拍」的樣本；建立 50 個特徵與比分推移特徵
   │
   ├─ 2. 訓練 14 個候選方法（run_method.py）
   │      樹模型：LightGBM、XGBoost、CatBoost 及其變體
   │      序列模型：GRU、因果 Transformer 及其變體
   │      每個方法都用「依比賽分組的 5-fold」產生 OOF 機率，並對 test 預測（5 個 fold 平均）
   │
   ├─ 3. 統一評估與集成（arena.py）
   │      依 test 的前綴長度分佈抽樣評估 → 調 Macro-F1 的決策規則 → 每個目標各自做貪婪集成
   │
   └─ 4. 產生 outputs/submission.csv
```

### 參考的方法與出處

| 方法 | 來源（AI CUP 2026 私榜名次） | 本 repo 位置 |
|---|---|---|
| LightGBM + 選手 target encoding + test 內部轉移擴充 | [0000workjason](https://github.com/0000workjason/aicup2026-tabletennis-prediction)（第 3 名） | `ttmatch/methods/trees.py` |
| 因果多任務 GRU / Transformer | 同上 | `ttmatch/methods/nn.py` |
| Macro-F1 決策規則（溫度 + 類別偏置） | 同上；第 5 名的 prior-correction 是它的特例 | `ttmatch/decision.py` |
| 依 match 分組的 CV、依 test 前綴長度評估 | 同上 | `ttmatch/cv.py` |
| GRU + Time2Vec + FiLM | [SugarSquirrel](https://github.com/SugarSquirrel/aicup2026-table-tennis)（第 5 名） | `ttmatch/methods/nn.py` |
| 左右鏡像資料增強、規則修正 | [jabir95tsai](https://github.com/jabir95tsai/pingpong_aicup_2026)（第 22 名） | `ttmatch/methods/nn.py`、`ttmatch/postprocess.py` |
| XGBoost / CatBoost 對照 | [lucas6028](https://github.com/lucas6028/AI_CUP_2026_Table_Tennis_Prediction) | `ttmatch/methods/trees.py` |
| **比分推移特徵**、選手 ID 嵌入 | 本 repo 新增 | `ttmatch/features.py`、`ttmatch/methods/nn.py` |

以上程式碼改寫自各隊公開的 repo，改寫的檔案開頭都標註了出處；原作著作權屬原作者。

## 一、資料前處理（`ttmatch/data.py`、`ttmatch/features.py`）

### 樣本的組成

train 與 test 的每一列是一拍。把「觀測完第 1..t 拍、預測第 t+1 拍」當成一筆樣本：

| 名稱 | 內容 | 用途 |
|---|---|---|
| `ap_base` | train 中每個 rally 的每一個前綴（只取有下一拍的列） | 訓練與 CV 評估 |
| `ap_aug` | test 中已經看得到的轉移（第 i 拍 → 第 i+1 拍，i 小於可見長度） | 樹模型的 action / point 額外訓練資料（不含任何要預測的答案） |
| `test_pred` | 每個 test rally 的最後一個可見拍（每個 `rally_uid` 一列） | 要預測的列，順序依 `sample_submission.csv` |

下一拍是發球（球種 15–18）的列會先去掉，因為要預測的那一拍不可能是發球；所以球種只預測 0–14 共 15 類。

### 特徵（每一列只用第 1..t 拍可得的資訊）

- **類別特徵 23 個**：性別、局數、下一拍的擊球類型（`next_strikeId`）；倒數第 1 拍的球種／落點／正反手／力道／旋轉／擊球類型（`last_strikeId`）／站位，倒數第 2 拍的球種／落點／正反手／旋轉／力道，倒數第 3 拍的球種／落點；發球拍的球種、旋轉、力道、落點、正反手；**下一拍擊球者的選手 ID**。還沒發生的拍用哨兵類別 50 表示。
- **數值特徵 27 個**：前綴長度、擊球者是否為發球方、是否在預測接發球／第三拍；發球方與接發方比分、比分差、總分、是否局點、是否 deuce；到目前為止攻擊／控制／防守的次數、落點為 0／1–3（短球區）／7–9（長球區）的次數、平均力道、攻擊比例、最近兩拍是否攻擊等。
- **選手 target encoding**（`lgbm`、`lgbm_te20`，以及序列模型 `gru`、`transformer`、`gru_t2v_film`、`gru_mirror` 使用；`lgbm_note`、`xgb`、`cat`、`*_pemb` 等不用）：樹模型用 `P(下一拍球種 | 下一拍擊球者)` 等三組，序列模型只用 `P(下一拍球種 | 下一拍擊球者)` 這一組。訓練列的編碼用巢狀 OOF（內層依比賽再切 5 份），避免編碼裡含有自己的答案；第 3 名原本的寫法有這個洩漏，這裡已修正。

### 本 repo 新增：比分推移特徵（只用於 winner）

test 裡同一局會出現好幾個 rally。**同一局下一個可見 rally 的開局比分**，揭露了這段期間發球方與接發方各得幾分；若其中一方一分都沒拿到，就能確定本 rally 的勝方。在 train 上用「同局下一個存在的 rally」驗證，只要能判定勝方，與 `serverGetPoint` 的一致率是 100%（13,718 / 13,718 個 rally）。

- 特徵：是否有下一個可見 rally、間隔幾分、發球方得幾分、接發方得幾分、發球方得分比例（`PROG_FEATURES`）。
- **train 端要模擬 test 的可見方式**：test 同局相鄰可見 rally 的 `rally_id` 間隔以 2–4 為主（另有 17% 是該局最後一個可見 rally）。train 端依這個間隔分佈幫每個 rally 抽一個間隔，取 `rally_id ≥ r + 間隔` 的第一個 rally 當作「下一個可見 rally」（`score_progression_train`），讓 train 與 test 的特徵分佈一致。
- **一致性檢查**：兩個 rally 必須是同一對選手、`rally_id − 1` 等於兩方比分和、得分非負且加總等於間隔，才會使用。test 的 match 340–354（262 個 rally）每場出現 21–31 位選手、`rally_id` 與比分對不起來，整場判定為異常，不提供比分推移資訊；這些 rally 的 winner 改用不含比分推移的集成。
- 這個特徵只用到 test 檔本身的欄位，不需要任何外部資料。

## 二、模型選擇（`run_method.py`）

共訓練 14 個候選方法，`python run_method.py --list` 可列出：

| 方法 | 類型 | 預測的目標 | 說明 |
|---|---|---|---|
| `lgbm` | LightGBM | action、point、winner | 第 3 名的作法：選手 target encoding（action）+ test 內部轉移擴充 |
| `lgbm_note` | LightGBM | action、point、winner | 同上但不加選手 target encoding（對照） |
| `lgbm_te20` | LightGBM | action | 選手 target encoding 改用較強的平滑 |
| `lgbm_noplayer` | LightGBM | action、point | 不用任何選手 ID（對照，量選手身分的重要性） |
| `xgb` | XGBoost | action、point、winner | 特徵同 `lgbm_note` |
| `cat` | CatBoost | action、point、winner | 特徵同 `lgbm_note`，類別特徵用 CatBoost 原生處理 |
| `lgbm_prog` | LightGBM | winner | 加入比分推移特徵 |
| `cat_prog` | CatBoost | winner | 加入比分推移特徵 |
| `gru` | GRU | 三個目標 | 第 3 名的因果多任務序列模型 |
| `transformer` | 因果 Transformer | 三個目標 | 同上 |
| `gru_t2v_film` | GRU + Time2Vec + FiLM | 三個目標 | 第 5 名的架構 |
| `gru_mirror` | GRU | 三個目標 | 左右鏡像資料增強（落點 1↔3、4↔6、7↔9），第 22 名 |
| `gru_pemb` | GRU | 三個目標 | 用選手 ID 嵌入取代選手 target encoding（本 repo 新增） |
| `transformer_pemb` | 因果 Transformer | 三個目標 | 同上 |

選這兩類模型的理由：樹模型擅長處理大量類別與手工特徵（例如選手 ID、比分情境），序列模型可以直接讀整串擊球序列。兩者的錯誤型態不同，集成時可以互補。

## 三、訓練過程

所有方法共用同一套切分：**依比賽（`match`）分組的 5-fold**（`GroupKFold`）。test 的 79 場比賽與 train 的 216 場完全不重疊，按比賽切才能反映對新比賽的泛化。每個方法輸出驗證 fold 的 OOF 機率，以及 5 個 fold 模型對 test 的平均預測，存成 `outputs/oof/<方法>.npz`。

### 樹模型（`ttmatch/methods/trees.py`）

- action、point 是多類別分類，winner 是二元分類，三個目標分開訓練。
- 類別權重：與類別頻率的 −0.5 次方成正比，讓稀有類別也被學到（Macro-F1 對每一類一視同仁）。
- action / point 的訓練資料是 train fold 的樣本加上 `ap_aug`（test 內部轉移，權重 ×5），讓模型看到 test 選手的打法。
- LightGBM：learning rate 0.05、num_leaves 63（winner 為 47）、feature / bagging fraction 0.8、L2 1.0，最多 1500 輪，驗證 fold 上 50 輪沒有進步就停止。
- XGBoost：learning rate 0.06、max_depth 7、subsample / colsample 0.8，最多 1500 輪，early stopping 50 輪。
- CatBoost：learning rate 0.12、depth 6、最多 700 輪，early stopping 60 輪；限制 CTR 組合以控制訓練時間。

### 序列模型（`ttmatch/methods/nn.py`）

- 一次讀入整個 rally，在每個位置 t 同時預測第 t+1 拍的球種、落點與 rally 勝負（因果遮罩，位置 t 看不到 t 之後的拍）。
- 每一拍的輸入：8 個類別欄位（球種、落點、正反手、力道、旋轉、擊球類型、站位、性別）的嵌入，加上 6 個數值（比分差、總分、擊球者是否發球方、上一拍是否攻擊、攻擊比例、累計攻擊次數）與拍序位置。
- GRU：2 層、hidden 160、dropout 0.3。Transformer：hidden 160、4 個 head、2 層。
- 損失：0.4 × 球種交叉熵 + 0.4 × 落點交叉熵 + 0.2 × 勝負二元交叉熵，與評分權重相同；類別交叉熵帶類別權重。
- Adam（learning rate 2e-3、weight decay 1e-5）、cosine 學習率排程、batch 128；這次繳交的版本每個方法 **1 個 seed × 12 個 epoch**（本機 CPU 訓練）。
- 訓練資料是 train fold 的 rally 加上 test rally 已可見的部分（test 的前綴本身就有「下一拍」可以學）。
- 選手資訊：`gru`、`transformer`、`gru_t2v_film`、`gru_mirror` 用下一拍擊球者的 target encoding（巢狀 OOF）；`_pemb` 版改用選手 ID 嵌入，沒在訓練資料出現過的選手對應到「未知」，訓練時也會隨機遮蔽。

## 四、評估、決策規則與集成（`arena.py`）

1. **依 test 的前綴長度評估**：test 的 rally 被截在不同長度，多半只看得到 1–3 拍。評估時每個驗證 rally 依 test 的前綴長度分佈抽一個前綴，重複 5 次取平均，讓 CV 的組成跟 test 一樣。
2. **Macro-F1 決策規則**：預測類別 = argmax_c [α · log p_c + w_c]。α 是溫度、w_c 是各類別的偏置，用座標上升調整，讓稀有類別也會被預測到，直接提高 Macro-F1。
3. **跨半調參**：把比賽分成兩半，決策規則在一半上調、在另一半上評分，兩半互換取平均，避免在同一批資料上調參又評分而高估。
4. **貪婪集成（優勝劣汰）**：每個目標各自從最佳單一方法開始，依序嘗試加入其他方法（權重 0.1–0.6），只有讓分數至少進步 0.0005 的方法才留下，其餘淘汰。winner 用 AUC 做同樣的選擇。
5. **另外評估、最後沒有採用**：依前綴長度分組的決策規則、第 22 名的規則修正（預測的類別在同一情境從未出現時改成眾數），兩者在 CV 上的增益都小於 0.0005。
6. 產生提交檔：決策規則在 5 次前綴長度抽樣合併（串接）後的樣本上重新調參後套到 test；test 異常比賽（match 340–354）的 winner 改用不含比分推移的集成。

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
| `cat_prog`（比分推移） | — | — | **0.8335** | — |
| `lgbm_prog`（比分推移） | — | — | 0.8253 | — |

### 最終集成（繳交的版本）

| 目標 | 留下的方法（權重） | CV |
|---|---|---:|
| action | transformer 0.38、lgbm_note 0.26、gru_pemb 0.20、xgb 0.16 | 0.3971 |
| point | lgbm 0.48、gru_mirror 0.32、gru_pemb 0.20 | 0.2023 |
| winner | cat_prog 0.45、xgb 0.45、transformer 0.10 | 0.8462 |
| winner（test 異常比賽的 262 列，不含比分推移） | cat 0.38、xgb 0.26、lgbm 0.20、transformer 0.16 | 0.6959 |
| **overall** | | **0.4090** |

這個 CV 分數的集成成員與權重是在同一批 OOF 上挑的，會略為樂觀。另外 test 有 14.2% 的列拿不到比分推移資訊，實際表現會再低一些（事後在 CV 上模擬這個比例，估計 test 約 0.405；這個模擬不在本程式碼裡）。

### 主要發現

1. **選手身分是最重要的訊號之一**：拿掉選手 ID（`lgbm_noplayer`），action F1 從 0.369 掉到 0.312。但選手 target encoding 修正洩漏後，反而不如直接把選手 ID 當類別特徵（`lgbm_note` 0.3687 對 `lgbm` 0.3581）。
2. **序列模型與樹模型互補**：Transformer 是最好的單一 action 模型（0.383），和樹模型集成後 action 到 0.397；point 以 LightGBM 為主。
3. **第 5 名的 Time2Vec + FiLM、第 22 名的鏡像增強**單獨都沒有勝過基本的 GRU，但鏡像版 GRU 在 point 集成中被選入，提供了多樣性。
4. **比分推移特徵**讓 winner AUC 從約 0.69 提升到 0.85，overall 約 +0.03。

## 六、操作說明

### 1. 環境

開發與繳交時的環境：Windows 11、Python 3.13、只用 CPU（10 核），沒有 GPU。

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS / Linux：source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 列出最低版本；序列模型需要 PyTorch，只用 CPU 時可以改裝 CPU 版：`pip install torch --index-url https://download.pytorch.org/whl/cpu`。

### 2. 資料

程式依下面的順序找資料：

1. 環境變數 `TTMATCH_DATA_DIR` 指定的資料夾；
2. repo 裡的 `data/` 資料夾（放 `train.csv`、`test.csv`、`sample_submission.csv`）；
3. 都沒有時，用 `kagglehub` 自動下載（需要把 Kaggle API token 放在 `~/.kaggle/kaggle.json`，並已加入這個競賽）。

### 3. 訓練候選方法

每個方法會輸出 `outputs/oof/<方法>.npz`（OOF 機率與 test 機率）。重現繳交版本的指令：

```bash
# 樹模型（8 個）
python run_method.py lgbm lgbm_note lgbm_te20 lgbm_noplayer xgb cat lgbm_prog cat_prog

# 序列模型（6 個）：繳交版本用 1 個 seed × 12 個 epoch
python run_method.py transformer gru gru_t2v_film gru_mirror gru_pemb transformer_pemb --epochs 12 --seeds 1
```

常用參數：

| 參數 | 說明 |
|---|---|
| `--list` | 列出所有方法 |
| `--device cuda` | 序列模型用 GPU 訓練（預設 `cpu`） |
| `--epochs`、`--seeds` | 序列模型的 epoch 數與 seed 數（預設 16、2；繳交版本是 12、1） |
| `--folds` | CV 的 fold 數（預設 5） |
| `--fast` | 冒煙測試：少量輪數 / epoch，輸出 `*_fast.npz`；再執行 `python arena.py --fast` 會產生 `outputs/submission_fast.csv`。例如 `python run_method.py lgbm_note cat_prog gru_pemb --fast --folds 2 --seeds 1 --epochs 1` 後接 `python arena.py --fast`，在 10 核 CPU 上約 5–10 分鐘 |

在本機 CPU 上的訓練時間（10 核）：樹模型每個約 3–20 分鐘（`lgbm` 約 16 分鐘、`xgb` 約 4 分鐘；`cat` 較慢，約 20–55 分鐘；`*_prog` 只有 winner，幾秒到 3 分鐘）；序列模型每個約 20–40 分鐘。全部約 4–5 小時。沒有 GPU 時，也可以用 [`notebooks/colab_train_nn.ipynb`](notebooks/colab_train_nn.ipynb) 在 Colab 訓練序列模型，再把 `outputs/oof/*.npz` 下載回本機。注意 notebook 裡的指令是 `--epochs 20 --seeds 3`，要重現繳交版本請改成 `--epochs 12 --seeds 1`。

### 4. 評估、集成、產生提交檔

```bash
python arena.py
```

- 讀取 `outputs/oof/` 底下所有的 `*.npz`，評估、集成後輸出：
  - `outputs/submission.csv`：提交檔（1,845 列：`rally_uid, actionId, pointId, serverGetPoint`）
  - `outputs/arena_report.md`：完整報告（單一方法排行榜、集成過程、最終配方）
- 本機約 10–15 分鐘。
- 其他選項：`--exclude <方法…>` 排除指定方法；`--tag <名稱>` 讓輸出改名為 `submission_<名稱>.csv`；`--no-override` 不嘗試規則修正；`--fast` 改讀冒煙測試的 npz。
- 不用比分推移特徵的版本：`python arena.py --exclude lgbm_prog cat_prog --tag noprog`，輸出 `outputs/submission_noprog.csv`（CV overall 0.3790）。

### 5. 單元測試

```bash
pip install pytest
python -m pytest tests -q
```

測試 Macro-F1 與 sklearn 一致、比分推移特徵的計算與一致性檢查、train 端的模擬、鏡像映射等（6 個測試，GitHub Actions 在每個 PR 自動執行）。

### 6. 重現性

- 所有隨機種子固定為 42；LightGBM 與 CatBoost 在同一台機器上可以重現。
- XGBoost 在不同執行緒數下、PyTorch 在不同硬體上，結果會有很小的差異；集成挑到的成員可能因此不同，分數差異在 0.001 左右。
- `outputs/` 不進版控，繳交時的 npz 沒有附在 repo 裡。

### 檔案結構

```
arena.py                 評估、集成、產生提交檔
run_method.py            訓練單一方法，輸出 outputs/oof/<方法>.npz
requirements.txt
ttmatch/
  config.py              路徑、類別空間、評分權重、CV 設定
  data.py                讀取 CSV、組出 ap_base / ap_aug / test_pred
  features.py            特徵工程、比分推移特徵、target encoding
  cv.py                  依 match 分組的 fold、依 test 前綴長度抽樣
  decision.py            Macro-F1 決策規則
  metrics.py             Macro-F1、AUC、overall
  postprocess.py         規則修正（評估後未採用）
  methods/trees.py       LightGBM / XGBoost / CatBoost
  methods/nn.py          GRU / Transformer / Time2Vec + FiLM
notebooks/colab_train_nn.ipynb   在 Colab 訓練序列模型
tests/test_core.py       單元測試
```

## 七、限制與事後檢討

- **CV 與排行榜的落差**：事後估計 test 約 0.405，實際 public 0.41744、private 0.39028。private 只有約 609 列，Macro-F1 的抽樣雜訊不小。
- **事後才發現的問題：樹模型使用 test 內部轉移，會讓 test 上幾乎不預測落點 0**。`ap_aug` 的每一筆都「一定不是終結拍」（下一拍還看得到），而且和要預測的那一列來自同一個 rally，樹模型因此學到「長得像這個 rally 的列不會是終結拍」。CV 看不到這個偏差：驗證 fold 的 rally 沒有自己的前綴在訓練資料裡。point 集成的主要成員 `lgbm` 就是這種模型。繳交版本的提交檔沒有保留下來，影響大小沒有在這個版本上量過；在後續同樣有這個問題的版本上，估計讓 point Macro-F1 少約 0.02–0.05。這是截止後比較「CV 與 test 預測各類別的比例」才發現的。
- 樹模型與序列模型的 early stopping 都用驗證 fold，OOF 分數會略為樂觀（各方法同樣受影響，方法之間的比較仍公平）。
- 比分推移特徵用的是 test 檔中其他 rally 的比分欄位，只用到競賽提供的資料，沒有使用外部資料。
- 截止後的分析與改進放在另一個分支（[`analysis/post-deadline`](https://github.com/ChiJiun/TTMATCH/tree/analysis/post-deadline)），**沒有計入成績**；以 Late Submission 驗證的 private 約 0.401–0.403。

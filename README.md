# TTMATCH — 桌球下一拍預測方法競技場

Kaggle 課堂競賽 [Introduction to Data Science Competition — TTMATCH](https://www.kaggle.com/competitions/introduction-to-data-science-competition-ttmatch) 的解法 repo。

本題資料源自 **AI CUP 2026 春季賽「基於時序資料之桌球戰術與結果預測競賽」**（423 隊）。本 repo 把該賽前段班公開的方法放在**同一套資料切分與評分**下比較，淘汰表現差的方法，再把勝出方法的優點結合起來。

## 任務

給定一個 rally 的前 n−1 拍，預測：

| 目標 | 類別 | 指標 | 權重 |
|---|---|---|---|
| `actionId` 下一拍球種 | 0–14（下一拍不會是發球） | Macro-F1 | 0.4 |
| `pointId` 下一拍落點 | 0–9 | Macro-F1 | 0.4 |
| `serverGetPoint` 發球方是否贏得此 rally | 機率 | AUC | 0.2 |

> Kaggle 頁面需登入才能看評分說明，這裡沿用 AI CUP 原賽的 `0.4·F1 + 0.4·F1 + 0.2·AUC`。

## 參考方法與出處

| 方法（`run_method.py` 名稱） | 來源隊伍（AI CUP 2026 私榜名次） | 本 repo 位置 |
|---|---|---|
| `lgbm`：LightGBM + 選手 target encoding + test 內部轉移擴充 | [0000workjason](https://github.com/0000workjason/aicup2026-tabletennis-prediction)（第 3 名） | `ttmatch/methods/trees.py` |
| `gru`、`transformer`：因果多任務序列模型 | 同上 | `ttmatch/methods/nn.py` |
| Macro-F1 決策規則（溫度 + 類別偏置） | 同上；第 5 名的 prior-correction β 是其特例 | `ttmatch/decision.py` |
| 依 match 分組 CV + test 長度對齊評估 | 同上（第 5、22 名也強調 match 分組） | `ttmatch/cv.py` |
| `gru_t2v_film`：GRU + Time2Vec + FiLM | [SugarSquirrel](https://github.com/SugarSquirrel/aicup2026-table-tennis)（第 5 名） | `ttmatch/methods/nn.py` |
| `gru_mirror`：左右鏡像資料增強 | [jabir95tsai](https://github.com/jabir95tsai/pingpong_aicup_2026)（第 22 名） | `ttmatch/methods/nn.py` |
| 規則修正（情境中從未出現的預測改為眾數） | 同上 | `ttmatch/postprocess.py` |
| `xgb`、`cat`：XGBoost / CatBoost 對照 | [lucas6028](https://github.com/lucas6028/AI_CUP_2026_Table_Tennis_Prediction)（前標） | `ttmatch/methods/trees.py` |
| `lgbm_prog`、`cat_prog`：**比分推移特徵**（本 repo 新增，見下） | — | `ttmatch/features.py` |

以上程式碼改寫自各隊公開 repo，每個檔案開頭都標註了出處；原作著作權屬原作者。

### 本 repo 新增：比分推移特徵

test 裡同一局會出現多個 rally，同局相鄰可見 rally 的 rally_id 間隔以 2–4 為主（2：12%、3：38%、4：31%；另有 17% 是該局最後一個可見 rally，沒有下一個）。**下一個可見 rally 的開局比分**揭露了這段期間發球方與接發方各得幾分：若發球方一分都沒拿到，本 rally 一定是接發方得分，反之亦然。在 train 上以「同局下一個存在的 rally」驗證：只要其中一方得分為 0（可判定勝方），推得的勝方與 `serverGetPoint` 一致率為 100%（13,718 / 13,718 個 rally）。

為了讓 train 和 test 的特徵分佈一致，train 端會依 test 實際的 rally 間隔分佈為每個 rally 抽一個間隔，取 rally_id ≥ r+間隔的第一個存在 rally 當作「下一個可見 rally」（`score_progression_train`）。這個特徵只用到 test 檔案本身提供的欄位，不需要任何外部標註。

兩個 rally 必須通過一致性檢查才會使用：同一對選手、`rally_id − 1 == 兩方比分和`、得分非負且相加等於間隔。test 的 match 340–354（262 個 rally）每場出現 21–31 位選手、rally_id 與比分不符，整場判定為異常，不提供比分推移資訊；arena 對這些 rally 的 winner 改用不含比分推移的集成。

> ⚠️ 這個特徵是從 test 檔中其他 rally 的比分欄位推得勝負。它不用外部資料，但效果極大（winner AUC 約 0.67 → 0.83）。是否符合課堂競賽規則，請自行判斷；不想用時執行 `python arena.py --exclude lgbm_prog cat_prog --tag noprog`，會另外產生 `outputs/submission_noprog.csv`。

## 使用方式

```bash
python -m venv .venv
.venv/Scripts/activate          # macOS/Linux：source .venv/bin/activate
pip install -r requirements.txt  # torch 可改裝 CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cpu
```

資料：把 Kaggle API token 放在 `~/.kaggle/kaggle.json`，程式會用 `kagglehub` 自動下載；或手動把 `train.csv`、`test.csv`、`sample_submission.csv` 放到 `data/`。

```bash
python run_method.py --list                       # 列出所有方法
python run_method.py lgbm xgb cat lgbm_prog       # 訓練方法，輸出 outputs/oof/<name>.npz
python run_method.py gru transformer --device cpu # 序列模型（本機 CPU 可跑，較慢）
python arena.py                                   # 比較、淘汰、集成 → outputs/arena_report.md、outputs/submission.csv
```

GPU 不夠時，用 [`notebooks/colab_train_nn.ipynb`](notebooks/colab_train_nn.ipynb) 在 Colab 訓練序列模型，再把 `outputs/oof/*.npz` 下載回本機。

## 評估方式（所有方法一致）

1. **依 match 分組的 5-fold**：test 的 79 場比賽與 train 的 216 場完全不重疊，按 match 切才能反映真實泛化。
2. **長度對齊**：test 的前綴長度大多是 1–3 拍；評估時每個驗證 rally 依 test 的長度分佈抽一個前綴（重複 5 次）。
3. **誠實的決策調參**：Macro-F1 決策規則在「另一半比賽」上調參、在這一半評估，兩半互換取平均。
4. **優勝劣汰**：每個任務各自做貪婪前向選擇，從最佳單一方法開始，只有能讓誠實分數上升的方法才會加入，其餘淘汰。

## 結果

5-fold、依 match 分組、長度對齊、決策規則跨半調參（完整報告：執行 `python arena.py` 後見 `outputs/arena_report.md`）。序列模型在本機 CPU 只跑了 1 個 seed × 12 epoch；用 Colab notebook 跑更多 seed 應該還能再進步。

### 單一方法

| 方法 | action F1 | point F1 | winner AUC | overall |
|---|---:|---:|---:|---:|
| `transformer`（第 3 名） | **0.3834** | 0.1889 | 0.6787 | **0.3647** |
| `lgbm_note`（第 3 名特徵，不加選手 TE） | 0.3687 | **0.1968** | 0.6652 | 0.3592 |
| `cat`（lucas6028） | 0.3672 | 0.1864 | 0.6885 | 0.3591 |
| `xgb`（lucas6028） | 0.3641 | 0.1889 | 0.6854 | 0.3583 |
| `lgbm`（第 3 名原作法，選手 TE 已修正洩漏） | 0.3581 | 0.1968 | 0.6652 | 0.3550 |
| `transformer_pemb` | 0.3661 | 0.1851 | 0.6632 | 0.3531 |
| `gru`（第 3 名） | 0.3699 | 0.1894 | 0.6463 | 0.3530 |
| `gru_t2v_film`（第 5 名） | 0.3663 | 0.1876 | 0.6497 | 0.3515 |
| `gru_pemb` | 0.3644 | 0.1938 | 0.6261 | 0.3485 |
| `gru_mirror`（第 22 名） | 0.3543 | 0.1835 | 0.6452 | 0.3442 |
| `lgbm_noplayer`（不用任何選手 ID） | 0.3116 | 0.1797 | — | — |
| `cat_prog`（比分推移） | — | — | **0.8335** | — |
| `lgbm_prog`（比分推移） | — | — | 0.8253 | — |

### 集成（優勝劣汰後留下的方法）

| 任務 | 留下的方法（權重） | CV |
|---|---|---:|
| action | transformer 0.38、lgbm_note 0.26、gru_pemb 0.20、xgb 0.16 | 0.3971 |
| point | lgbm 0.48、gru_mirror 0.32、gru_pemb 0.20 | 0.2023 |
| winner | cat_prog 0.45、xgb 0.45、transformer 0.10 | 0.8462 |
| **overall** | | **0.4090** |
| overall（不用比分推移，`--exclude lgbm_prog cat_prog`） | winner 改為 cat / lgbm / xgb / transformer | 0.3790 |

分深度決策與規則修正（第 22 名）在 CV 上都沒有帶來穩定增益（< 0.0005），因此未採用。集成分數含選擇偏差，會比單一方法排行榜略為樂觀。

### 主要發現

1. **選手身分是最重要的訊號之一**：拿掉選手 ID，action F1 從 0.369 掉到 0.312。但第 3 名的選手 target encoding 原寫法有標籤洩漏（訓練列的 TE 含自己的答案），修正成巢狀 OOF 後仍不如直接用原始選手 ID 類別特徵。
2. **序列模型與樹模型互補**：Transformer 是最佳單一 action 模型，但集成後 action 從 0.383 提升到 0.397；point 則以 LightGBM 為主。
3. **第 5 名的 Time2Vec + FiLM、第 22 名的鏡像增強**在這裡單獨都沒有勝過基本 GRU，但鏡像版 GRU 在 point 集成中被選入（增加多樣性）。
4. **比分推移特徵**讓 winner AUC 從 0.70 提升到 0.85，overall 約 +0.03。

### 已知限制

- Kaggle 公開榜尚未上傳驗證；評分公式沿用 AI CUP 原賽 0.4/0.4/0.2，未能在 Kaggle 頁面確認。
- 樹模型與 NN 的 early stopping 用的是驗證 fold，OOF 分數會略為樂觀（各方法同樣受影響，排名比較仍公平）。
- CV 無法衡量 test 端「同場比賽資訊」（ap_aug 的 test 內部轉移、test 選手 TE）帶來的好處，實際 test 表現可能與 CV 有落差。
- 類別權重用整個 ap_base 計算（影響極小）。
- test 的 match 340–354 資料異常（每場 21–31 位選手），這 262 個 rally 的 winner 只能靠擊球序列預測。

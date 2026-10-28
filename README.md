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

test 裡同一局會出現多個 rally（依 rally_id 間隔 1–4 抽樣）。**下一個可見 rally 的開局比分**揭露了這段期間發球方與接發方各得幾分：若發球方一分都沒拿到，本 rally 一定是接發方得分，反之亦然。在 train 上，相鄰 rally 的比分變化與 `serverGetPoint` 一致率為 98.6%。

為了讓 train 和 test 的特徵分佈一致，train 端會依 test 實際的 rally 間隔分佈為每個 rally 抽一個間隔，模擬「下一個可見 rally」（`score_progression_train`）。這個特徵只用到 test 檔案本身提供的欄位，不需要任何外部標註。

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

<!-- RESULTS -->

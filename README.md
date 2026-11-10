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
| `lgbm_prog`、`cat_prog`、`xgb_prog`：**比分推移特徵**（本 repo 新增，見下） | — | `ttmatch/features.py` |
| `*_hist`、`*_hist2`：**同場先前資訊（history）特徵**（本 repo 新增，見下） | — | `ttmatch/history.py` |

以上程式碼改寫自各隊公開 repo，每個檔案開頭都標註了出處；原作著作權屬原作者。

### 本 repo 新增：比分推移特徵

test 裡同一局會出現多個 rally，同局相鄰可見 rally 的 rally_id 間隔以 2–4 為主（2：12%、3：38%、4：31%；另有 17% 是該局最後一個可見 rally，沒有下一個）。**下一個可見 rally 的開局比分**揭露了這段期間發球方與接發方各得幾分：若發球方一分都沒拿到，本 rally 一定是接發方得分，反之亦然。在 train 上以「同局下一個存在的 rally」驗證：只要其中一方得分為 0（可判定勝方），推得的勝方與 `serverGetPoint` 一致率為 100%（13,718 / 13,718 個 rally）。

為了讓 train 和 test 的特徵分佈一致，train 端會依 test 實際的 rally 間隔分佈為每個 rally 抽一個間隔，取 rally_id ≥ r+間隔的第一個存在 rally 當作「下一個可見 rally」（`score_progression_train`）。這個特徵只用到 test 檔案本身提供的欄位，不需要任何外部標註。

兩個 rally 必須通過一致性檢查才會使用：同一對選手、`rally_id − 1 == 兩方比分和`、得分非負且相加等於間隔。test 的 match 340–354（262 個 rally）每場出現 21–31 位選手、rally_id 與比分不符，整場判定為異常，不提供比分推移資訊；arena 對這些 rally 的 winner 改用不含比分推移的集成。

> ⚠️ 這個特徵是從 test 檔中其他 rally 的比分欄位推得勝負。它不用外部資料，但效果極大（winner AUC 約 0.67 → 0.83）。是否符合課堂競賽規則，請自行判斷；不想用時執行 `python arena.py --preset noprog --tag noprog`，會另外產生 `outputs/submission_noprog.csv`。

### 本 repo 新增：同場先前資訊（history）特徵

train 與 test 的比賽不重疊，但 test 每場比賽平均可見 23 個 rally。「下一拍擊球者在這場比賽較早的 rally 裡怎麼打」（各球種、各落點的比例、攻擊率、平均力道，往全體分佈平滑）是選手 ID 之外合法可用的訊息：只用該 rally **之前**的 rally，不碰任何要預測的拍（`ttmatch/history.py`）。`*_hist2` 再加上目前擊球者（對手）的同一組統計。

分佈對齊是關鍵。test 每局只看得到約 34% 的 rally，每個 rally 只看得到前綴 k 拍，而且第 k+1 拍一定存在（那是要預測的拍），所以看不到 rally 的最後一拍，長 rally 也更容易成為可見 rally。train 端先以 34% 機率保留 rally，再把 test 的前綴長度分佈「由長到短」配對到夠長的 rally 上，使 train 的可見前綴長度分佈與 test 幾乎相同（每個可見 rally 平均 3.07 拍，test 也是 3.07）。

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
python arena.py --preset final                    # 比較、淘汰、集成 → outputs/arena_report.md、outputs/submission.csv
python eval_oof.py lgbm_note lgbm_hist            # 只比較幾個方法的單模分數（實驗用，比 arena 快很多）
```

GPU 不夠時，用 [`notebooks/colab_train_nn.ipynb`](notebooks/colab_train_nn.ipynb) 在 Colab 訓練序列模型，再把 `outputs/oof/*.npz` 下載回本機。

## 評估方式（所有方法一致）

1. **依 match 分組的 5-fold**：test 的 79 場比賽與 train 的 216 場完全不重疊，按 match 切才能反映真實泛化。
2. **長度對齊**：test 的前綴長度大多是 1–3 拍；評估時每個驗證 rally 依 test 的長度分佈抽一個前綴（重複 5 次）。
3. **誠實的決策調參**：Macro-F1 決策規則在「另一半比賽」上調參、在這一半評估，兩半互換取平均。
4. **優勝劣汰**：每個任務各自做貪婪前向選擇，從最佳單一方法開始，只有能讓誠實分數上升的方法才會加入，其餘淘汰。
5. **巢狀估計**：集成成員、權重與決策規則全部只用「另一半比賽」挑選，再在這一半評分，兩半互換。這是沒有選擇偏差的估計，用來判斷一個改動是否真的有效（`arena.py` 的 `nested_estimate`）。

## 結果

5-fold、依 match 分組、長度對齊、決策規則跨半調參。重現：`python arena.py --preset final`（報告在 `outputs/arena_report.md`）。序列模型在本機 CPU 只跑了 1 個 seed × 12 epoch；用 Colab notebook 跑更多 seed 應該還能再進步。

### 三種分數怎麼看

| 分數 | 意義 |
|---|---|
| in-sample | 集成成員、權重、決策都在同一份 OOF 上挑，含選擇偏差，偏樂觀 |
| **巢狀估計** | 成員、權重、決策只用「另一半比賽」挑選，在這一半評分，兩半互換；沒有選擇偏差，**用來決定採用哪個版本** |
| **test 預期** | 再模擬 test 有 14.2% 的列（異常 match）拿不到比分推移、只能用 plain 集成 |

### 最終集成（`--preset final`，22 個方法參與，優勝劣汰後留下 11 個，含異常 match 用的 plain 集成）

| 任務 | 留下的方法（權重） | in-sample | 巢狀估計 |
|---|---|---:|---:|
| action | xgb_hist2 0.36、transformer_hist2 0.25、lgbm_note 0.18、lgbm_hist 0.11、gru_pemb_hist 0.10 | 0.4160 | 0.4127 |
| point | xgb_hist2 0.38、lgbm_hist2 0.27、transformer_hist2 0.25、gru_pemb_hist 0.10 | 0.2127 | 0.2086 |
| winner | xgb_prog 0.45、cat 0.45、transformer_hist 0.10 | 0.8458 | 0.8462 |
| **overall** | | 0.4206 | **0.4177** |
| **test 預期** | winner 約 0.8281（±0.0017） | | **約 0.4171** |
| 不用比分推移（`--preset noprog`，winner 0.6959） | | 0.3907 | 0.3874 |

分深度決策與規則修正（第 22 名）在 CV 上都沒有帶來穩定增益（< 0.0005），因此未採用。

### 各版本比較（巢狀估計）

| 版本 | overall 巢狀估計 | test 預期 |
|---|---:|---:|
| PR #3（第一輪方法） | 0.4063 | 0.4055 |
| **+ 同場 history 特徵（目前採用）** | **0.4177** | **0.4171** |
| + action/point 也用比分推移（`lgbm_hist_prog` 等 3 個） | 0.4157 | 0.4161 |

最後一列的單一模型確實變強（`transformer_hist2_prog` 單模 overall 0.4100，是最佳單一模型），但多了候選方法後，貪婪選擇用快速調參分數挑中的成員在完整調參下反而較差（action 0.4122 vs 0.4160），巢狀估計下降，因此不採用。

### 單一方法（節錄）

| 方法 | action F1 | point F1 | winner AUC | overall |
|---|---:|---:|---:|---:|
| `transformer_hist2`（+ 雙方 history） | 0.4030 | 0.2042 | 0.6712 | **0.3771** |
| `transformer_hist`（+ history） | **0.4034** | 0.1971 | 0.6769 | 0.3756 |
| `transformer`（第 3 名） | 0.3834 | 0.1889 | 0.6787 | 0.3647 |
| `gru_hist` | 0.3907 | 0.1943 | 0.6514 | 0.3643 |
| `lgbm_note`（第 3 名特徵，不加選手 TE） | 0.3687 | 0.1968 | 0.6652 | 0.3592 |
| `cat`（lucas6028） | 0.3672 | 0.1864 | **0.6885** | 0.3591 |
| `xgb`（lucas6028） | 0.3641 | 0.1889 | 0.6854 | 0.3583 |
| `gru_pemb_hist` | 0.3808 | 0.2041 | 0.6186 | 0.3577 |
| `gru`（第 3 名） | 0.3699 | 0.1894 | 0.6463 | 0.3530 |
| `gru_t2v_film`（第 5 名） | 0.3663 | 0.1876 | 0.6497 | 0.3515 |
| `gru_mirror`（第 22 名） | 0.3543 | 0.1835 | 0.6452 | 0.3442 |
| `xgb_hist2` / `xgb_hist` / `cat_hist` | 0.3883 / 0.3855 / 0.3865 | 0.2002 / 0.1986 / 0.1934 | — | — |
| `lgbm_hist2` / `lgbm_hist` | 0.3726 / 0.3752 | 0.2005 / 0.1962 | — | — |
| `xgb_prog` / `cat_prog` / `lgbm_prog`（比分推移） | — | — | 0.8346 / 0.8335 / 0.8253 | — |

完整排行榜（含所有對照組）見 `outputs/arena_report.md`；`python eval_oof.py <方法…>` 可單獨比較幾個方法。

### 主要發現

1. **同場先前資訊（history 特徵）是本輪最大的進步**：每個模型家族都受益——Transformer action +0.020、XGBoost action +0.024 / point +0.011、CatBoost action +0.019、GRU（選手嵌入版）point +0.010；集成的巢狀估計 +0.011。關鍵是 train 端的模擬：只用 test 實際可見的 rally 比例與前綴長度分佈，否則 train 的統計量會比 test 豐富得多（第一版沒模擬時，每列「先前可見拍數」train 平均 53.6、test 只有 18.4）。
2. **選手身分是最重要的訊號之一**：拿掉選手 ID，action F1 從 0.369 掉到 0.312。第 3 名的選手 target encoding 原寫法有標籤洩漏，修正成巢狀 OOF 後仍不如直接用原始選手 ID。再加入「目前擊球者 ID」反而變差（action −0.006），第二個高基數選手特徵會過擬合。
3. **左右鏡像增強（第 22 名）在這份資料無效**：套到樹模型 action −0.017、point −0.004，GRU 鏡像版也是最弱的 NN。大多數選手慣用右手，正手 / 反手兩側並不對稱。
4. **seed bagging 與調整 LightGBM 大小 / 正則化都沒有穩定增益**（action −0.003～+0.002；較大模型的 point 反而 −0.006）。
5. **比分推移特徵**讓 winner AUC 從 0.70 提升到 0.85，overall 約 +0.03。它也能幫 point（下一拍擊球者若是這個 rally 的輸家，有 39% 是失誤落點 0；若是勝方只有 0.7%），單一模型 point +0.003～+0.008，但集成層面沒有提升（見上表）。

### 分數上限：為什麼離 0.5 還很遠

要在 winner AUC 0.83 下達到 overall 0.5，action + point 的 Macro-F1 合計需 0.835，目前約 0.63。以目前集成在 OOF 上的表現：

| | top-1 正確率 | top-3 正確率 | Macro-F1 |
|---|---:|---:|---:|
| action | 51% | 86% | 0.42 |
| point | 31% | 64% | 0.22 |

point 是瓶頸：落點 7 / 8 / 9 的 F1 都只有約 0.2；落點 1、3 合計只佔 1.2% 的資料卻佔 point Macro-F1 的 20%，F1 只有 0.13 / 0.05。這反映的是下一拍落點本身的隨機性，不是模型容量問題。原 AI CUP 第 3 名私榜 0.392（評分資料不同，不能直接比較）。再往上追 CV 分數，主要會是在擬合雜訊。

### 已知限制

- Kaggle 公開榜尚未上傳驗證；評分公式沿用 AI CUP 原賽 0.4/0.4/0.2，未能在 Kaggle 頁面確認。
- 樹模型與 NN 的 early stopping 用的是驗證 fold，OOF 分數會略為樂觀（各方法同樣受影響，排名比較仍公平）。
- CV 無法衡量 test 端「同場比賽資訊」（ap_aug 的 test 內部轉移）帶來的好處，實際 test 表現可能與 CV 有落差。
- history 特徵的 train 模擬與 test 仍有小差異：test 的攻擊率略高（0.37 vs 0.34）、落點 9 比例略高（0.26 vs 0.23）。
- 類別權重用整個 ap_base 計算（影響極小）。
- test 的 match 340–354 資料異常（每場 21–31 位選手），這 262 個 rally 拿不到比分推移資訊，winner 只能靠擊球序列與 history 預測；上表的「test 預期」已模擬這一點。
- 巢狀估計只評一半比賽，雜訊比全量評估大；0.002 以內的差異不應視為顯著。

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
| `lgbm`：LightGBM + 選手 target encoding + test 內部轉移擴充（其餘樹模型不用這個擴充，原因見下） | [0000workjason](https://github.com/0000workjason/aicup2026-tabletennis-prediction)（第 3 名） | `ttmatch/methods/trees.py` |
| `gru`、`transformer`：因果多任務序列模型 | 同上 | `ttmatch/methods/nn.py` |
| Macro-F1 決策規則（溫度 + 類別偏置） | 同上；第 5 名的 prior-correction β 是其特例 | `ttmatch/decision.py` |
| 依 match 分組 CV | 同上（第 5、22 名也強調 match 分組） | `ttmatch/cv.py` |
| `gru_t2v_film`：GRU + Time2Vec + FiLM | [SugarSquirrel](https://github.com/SugarSquirrel/aicup2026-table-tennis)（第 5 名） | `ttmatch/methods/nn.py` |
| `gru_mirror`：左右鏡像資料增強 | [jabir95tsai](https://github.com/jabir95tsai/pingpong_aicup_2026)（第 22 名） | `ttmatch/methods/nn.py` |
| 規則修正（情境中從未出現的預測改為眾數） | 同上 | `ttmatch/postprocess.py` |
| `xgb`、`cat`：XGBoost / CatBoost 對照 | [lucas6028](https://github.com/lucas6028/AI_CUP_2026_Table_Tennis_Prediction)（前標） | `ttmatch/methods/trees.py` |
| `lgbm_prog`、`cat_prog`、`xgb_prog`：**比分推移特徵**（本 repo 新增，見下） | — | `ttmatch/features.py` |
| `*_hist`、`*_hist2`：**同場先前資訊（history）特徵**（本 repo 新增，見下） | — | `ttmatch/history.py` |
| **把打散的 test rally 放回原本的比賽**（本 repo 新增，見下） | — | `ttmatch/relocate.py` |
| **截斷機制對齊**：訓練權重與評估抽樣都照 test 的截法（本 repo 新增，見下） | — | `ttmatch/truncation.py` |

以上程式碼改寫自各隊公開 repo，每個檔案開頭都標註了出處；原作著作權屬原作者。

### 本 repo 新增：比分推移特徵

test 裡同一局會出現多個 rally，同局相鄰可見 rally 的 rally_id 間隔以 2–4 為主（把打散的 rally 放回去之後，2：26%、3：33%、4：22%；另有 15% 是該局最後一個可見 rally、4% 是放不回去的打散 rally，沒有下一個）。**下一個可見 rally 的開局比分**揭露了這段期間發球方與接發方各得幾分：若發球方一分都沒拿到，本 rally 一定是接發方得分，反之亦然。在 train 上以「同局下一個存在的 rally」驗證：只要其中一方得分為 0（可判定勝方），推得的勝方與 `serverGetPoint` 一致率為 100%（13,718 / 13,718 個 rally）。

為了讓 train 和 test 的特徵分佈一致，train 端會依 test 實際的 rally 間隔分佈為每個 rally 抽一個間隔，取 rally_id ≥ r+間隔的第一個存在 rally 當作「下一個可見 rally」（`score_progression_train`）。這個特徵只用到 test 檔案本身提供的欄位，不需要任何外部標註。

兩個 rally 必須通過一致性檢查才會使用：同一對選手、`rally_id − 1 == 兩方比分和`、得分非負且相加等於間隔。test 的 match 340–354（262 個 rally）每場出現 21–31 位選手、rally_id 與比分不符；其中 187 個能放回原本的比賽（見下一節），剩下 75 個不提供比分推移資訊，arena 對這 75 個 rally 的 winner 改用不含比分推移的集成。

> ⚠️ 這個特徵是從 test 檔中其他 rally 的比分欄位推得勝負。它不用外部資料，但效果極大（winner AUC 約 0.67 → 0.83）。是否符合課堂競賽規則，請自行判斷；不想用時執行 `python arena.py --preset noprog --tag noprog`，會另外產生 `outputs/submission_noprog.csv`。

### 本 repo 新增：同場先前資訊（history）特徵

train 與 test 的比賽不重疊，但 test 每場比賽平均可見 23 個 rally。「下一拍擊球者在這場比賽較早的 rally 裡怎麼打」（各球種、各落點的比例、攻擊率、平均力道，往全體分佈平滑）是選手 ID 之外合法可用的訊息：只用該 rally **之前**的 rally，不碰任何要預測的拍（`ttmatch/history.py`）。`*_hist2` 再加上目前擊球者（對手）的同一組統計。

分佈對齊是關鍵。test 每局只看得到約 34% 的 rally，每個 rally 只看得到前綴 k 拍，而且第 k+1 拍一定存在（那是要預測的拍），所以看不到 rally 的最後一拍。train 端先以 34% 機率保留 rally（與 rally 長度無關），再把 test 的前綴長度分佈「由長到短」配對到夠長的 rally 上（偏向長 rally 的只有前綴長度 k 的分配，不是哪些 rally 可見），使 train 的可見前綴長度分佈與 test 幾乎相同（每個可見 rally 平均 3.07 拍，test 也是 3.07）。

### 本 repo 新增：把打散的 test rally 放回原本的比賽

test 的 match 340–354 不是真的比賽。這 262 個 rally 的選手組合**全部**出現在其他正常的 test match 裡，選手 ID、性別與開局比分都是真的，只有 `match`、`numberGame`、`rally_id` 三個編號被打亂——它們是正常 test match 裡沒有公開的 rally。

正常資料中 `rally_id − 1 = 開局比分總和`，所以真實的 rally_id 可以直接由比分算出，剩下要找的是哪一場、哪一局。候選的局必須同時滿足：同一對選手與性別、該局沒有公開過這個 rally_id、比分夾在前後已知 rally 之間（雙方得分都不倒退）、發球者符合發球輪替、該局在這個比分下還沒結束。只用這 5 個條件就能唯一定位的 rally 有很強的規律：126 個裡有 124 個剛好在「前一個公開 rally 的後 2 分」。所以再把這個規律當成第 6 個必要條件，6 個條件都滿足的候選只剩一個才放回去（`ttmatch/relocate.py`）。

- 放回 **187 / 262** 個；在 train 上模擬同樣的公開 / 打散樣式，放回去的 rally 有 98.8% 位置正確（666 個中 658 個）。
- 放回去的 rally 自己有了比分推移與同場 history 特徵；它前面的公開 rally 也多了一個更近的比分檢查點（間隔 4 變成 2，間隔 2 時勝負幾乎可以直接讀出來）。
- 拿不到比分推移資訊的 test 列從 14.2% 降到 4.1%。剩下 75 個放不回去的 rally 也幾乎沒有同場 history（平均先前可見拍數 0.3，其他列是 23.2）。

只用到 test 已公開的欄位（選手、比分），不涉及任何答案；性質與比分推移特徵相同，不想用時加 `--no-relocate`（`arena.py`）或用 `--preset noprog`。

### 本 repo 新增：截斷機制對齊

資料上的兩個事實（train 中分別是 100% 與 99.9%）：**rally 的最後一拍 `pointId` 一定是 0**，而且**最後一拍的擊球者輸掉這一分**。所以「要預測的那一拍是不是 rally 的最後一拍」（以下稱終結拍）同時決定了落點 0、球種分佈與勝負。

train 的每個 rally 提供所有前綴（k = 1..L−1），其中終結拍只佔 21.5%。test 每個 rally 只有一個前綴，終結拍佔多少取決於主辦單位怎麼截。假設每個 rally 從一個固定的分佈 g 抽截斷位置 k、抽到超過 L−1 就重抽（稱為 F0），再假設 test 的 rally 長度分佈與 train 相同，g 就能由 test 的前綴長度分佈唯一解出（`ttmatch/truncation.py`）。F0 之下終結拍佔 **43.2%**，是未加權訓練資料的兩倍。

證據不用任何 test 答案：比分推移揭露了部分 test rally 的勝負，而勝負等同 rally 長度的奇偶，所以每個機制都能算出這份比分資料的對數概似（`python tools/truncation_evidence.py`）。

| 機制 | 對數概似 | 整體終結率 | k=1 / 2 / 3 的終結率 | k 邊際與 test 的最大差 |
|---|---:|---:|---|---:|
| 要預測的那一拍從來不是最後一拍 | −1817.4 | 0 | 0 / 0 / 0 | 0.064 |
| 所有前綴等權（未加權的訓練資料） | −1788.3 | 0.215 | 0.125 / 0.197 / 0.287 | 0.069 |
| 上一版的評估抽樣（k 照 test 分佈抽，超過就截到 L−1） | −1786.6 | 0.420 | 0.341 / 0.408 / 0.530 | 0.090 |
| **F0** | **−1778.9** | 0.432 | 0.453 / 0.422 / 0.502 | 0.002 |

F0 比「等權」高 9.3（約 1 萬倍的概似比），比「從來不是最後一拍」高 38；與等權的差異幾乎全部來自 k = 1：長度 2 的 rally 只能截在 k = 1，所以 test 中 k = 1 的終結率是 P(L=2) / P_test(k=1) ≈ 0.45（比分資料單獨估出來是 0.47，95% 區間 0.32–0.61），等權時只有 0.125。**k ≥ 2 的終結率，比分資料分辨不出 0.2–0.45 之間的值**（各 k 的 95% 區間都很寬，k = 2、4、5、6 的點估計比 F0 低），這一段是機制假設推出來的，不是資料直接證實的；「結果」一節有對這個假設的敏感度分析。

用法有兩處：

1. **訓練權重**：每個訓練列的權重是 `g(k) / Σ_{j ≤ L−1} g(j)`，加權後的列分佈就是 F0 下的 test 分佈（有效樣本數約為原本的 56%）。test 內部轉移（一定不是終結拍）併入訓練時，各前綴長度的終結拍比例維持不變（`blend_aug_weights`）。
2. **評估抽樣**：每個驗證 rally 依 g(k) 抽一個前綴（`cut_matched_indices`）。上一版的抽樣把超過的 k 截到 L−1，k = 1 的比例因此偏高（36.7% vs test 的 27.5%）。

### 本 repo 的修正：樹模型不用 test 內部轉移擴充

第 3 名的作法把 test rally 內部的轉移（前綴 k → 第 k+1 拍，k 小於可見長度）也加進訓練，讓模型看到 test 選手的打法。這些列有兩個特性：它們和「真正要預測的那一列」來自同一個 rally（選手、比分、前幾拍都一樣），而且**一定不是終結拍**（下一拍還看得到）。樹模型因此學到「長得像這個 rally 的列不會是終結拍」，對要預測的那一列幾乎不預測落點 0：

| LightGBM + history | 平均 P(落點 0)：CV | 平均 P(落點 0)：test | test 中下一拍擊球者沒在 train 出現過的列 |
|---|---:|---:|---:|
| 使用 test 內部轉移（`lgbm_hist_aug`） | 0.41 | **0.07** | 0.02 |
| 不使用（`lgbm_hist`，現行作法） | 0.39 | 0.40 | 0.44 |

CV 完全看不到這個偏差：驗證 fold 的 rally 沒有自己的前綴在訓練集裡。它是在比較「集成在 CV 與在 test 上預測各類別的比例」時發現的——未修正時 CV 有 41% 預測落點 0，test 只有 1.4%。若 test 的終結拍比例如上一節估計的 43%，這會讓 point Macro-F1 少約 0.05（把 CV 上類別 0 的機率壓到同樣的預測比例來估計），改善回合 1 的提交檔也有同樣的問題（test 只有 4% 預測落點 0，估計 point 少約 0.02）。

序列模型沒有這個現象（test 與 CV 的平均 P(落點 0) 一致），仍然使用這些列。`arena.py` 的報告現在固定列出「CV 與 test 的預測分佈」，差異過大時會警告。

## 使用方式

```bash
python -m venv .venv
.venv/Scripts/activate          # macOS/Linux：source .venv/bin/activate
pip install -r requirements.txt  # torch 可改裝 CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cpu
```

資料：把 Kaggle API token 放在 `~/.kaggle/kaggle.json`，程式會用 `kagglehub` 自動下載；或手動把 `train.csv`、`test.csv`、`sample_submission.csv` 放到 `data/`。

```bash
python run_method.py --list                              # 列出所有方法
python run_method.py lgbm_note xgb lgbm_hist_prog cat_prog  # 訓練方法，輸出 outputs/oof/<name>.npz
python run_method.py gru transformer_hist2 --device cpu  # 序列模型（本機 CPU 可跑，較慢）
python arena.py --preset final                           # 比較、淘汰、集成 → outputs/arena_report.md、outputs/submission.csv
python arena.py --preset noprog --tag noprog             # 不用比分推移特徵的版本 → outputs/submission_noprog.csv
python eval_oof.py lgbm_note lgbm_hist                   # 只比較幾個方法的單模分數（實驗用，比 arena 快很多）
```

同時跑多個訓練行程時，用環境變數 `TTMATCH_THREADS=4` 限制每個樹模型行程的執行緒數（CatBoost 預設會佔滿所有核心）。GPU 不夠時，用 [`notebooks/colab_train_nn.ipynb`](notebooks/colab_train_nn.ipynb) 在 Colab 訓練序列模型，再把 `outputs/oof/*.npz` 下載回本機。

檢查用的工具（都只讀 `outputs/`，不訓練）：

```bash
python tools/truncation_evidence.py                 # 截斷機制的證據（對數概似、各前綴長度的終結率估計）
python tools/terminal_bias.py lgbm_hist_aug lgbm_hist  # 各方法在 CV 與 test 上的平均 P(落點 0)
python tools/sensitivity.py outputs                 # 截斷假設不成立時，集成的分數差多少
python tools/compare_versions.py outputs/r1 outputs # 兩個版本的差距與 95% 區間（依 match 成對 bootstrap）
```

## 評估方式（所有方法一致）

1. **依 match 分組的 5-fold**：test 的 79 場比賽與 train 的 216 場完全不重疊，按 match 切才能反映真實泛化。
2. **截斷機制對齊**：評估時每個驗證 rally 依估計出來的 test 截斷機制抽一個前綴（重複 5 次），前綴長度分佈與終結拍比例都對齊 test（見上面「截斷機制對齊」）。
3. **誠實的決策調參**：Macro-F1 決策規則在「另一半比賽」上調參、在這一半評估，兩半互換取平均。
4. **優勝劣汰**：每個任務各自做貪婪前向選擇，從最佳單一方法開始，只有能讓誠實分數上升的方法才會加入，其餘淘汰。
5. **巢狀估計**：集成成員、權重與決策規則全部只用「另一半比賽」挑選，再在這一半評分，兩半互換。這是沒有選擇偏差的估計，用來判斷一個改動是否真的有效（`arena.py` 的 `nested_estimate`）。
6. **CV 與 test 的預測分佈檢查**：同一組決策規則下，集成在 CV 抽樣與在 test 上「預測為各類別」的比例應該接近；差很多代表 test 端的預測與 CV 不一致，這種問題 CV 分數看不出來（`arena.py` 的 `prediction_shift`）。

## 結果

5-fold、依 match 分組、截斷機制對齊、決策規則跨半調參。重現：`python arena.py --preset final`（報告在 `outputs/arena_report.md`）。

> 評估抽樣在改善回合 2 改了（見「截斷機制對齊」），**以下數字不能和先前版本 README 的數字直接比較**。「各版本比較」把上一輪的模型放在新的評估方式下重新計分。

序列模型在本機 CPU 只跑了 1 個 seed × 12 epoch。不含 history 的 6 個序列模型（`gru`、`transformer`、`gru_pemb`、`transformer_pemb`、`gru_t2v_film`、`gru_mirror`）沿用改善回合 1 訓練的結果，這一輪沒有重訓：它們不用 history 與比分推移特徵，輸入不受這一輪的改動影響。

### 三種分數怎麼看

| 分數 | 意義 |
|---|---|
| in-sample | 集成成員、權重、決策都在同一份 OOF 上挑，含選擇偏差，偏樂觀 |
| **巢狀估計** | 成員、權重、決策只用「另一半比賽」挑選，在這一半評分，兩半互換；集成內部的挑選沒有選擇偏差，**用來決定採用哪個版本** |
| **test 預期** | 再模擬 test 有 4.1% 的列（放不回去的打散 rally）拿不到比分推移、只能用不含比分推移的集成。折減量是用 in-sample 的集成估的（主集成 − 混合），再從巢狀估計扣掉 |

注意：巢狀估計不含「版本挑選」這一層；它只重做集成成員、權重與決策規則的挑選，分深度決策、規則修正沒有在巢狀估計裡重做（兩者最後都沒有採用）。巢狀估計每次只評一半比賽，單看它的雜訊不小；版本之間的比較另外用依 match 的成對 bootstrap 量化（見下）。

### 最終集成（`--preset final`，30 個方法參與，主集成留下 7 個）

| 任務 | 留下的方法（權重） | in-sample | 巢狀估計 |
|---|---|---:|---:|
| action | transformer_hist2 0.34、xgb_hist2_prog 0.32、gru_hist_al 0.20、lgbm_note 0.14 | 0.4310 | 0.4233 |
| point | xgb_hist2_prog 0.54、lgbm_hist_prog 0.36、lgbm_hist_eq 0.10 | 0.2143 | 0.2074 |
| winner | cat_prog_bag3 0.90、gru_hist_al 0.10 | 0.8622 | 0.8617 |
| **overall** | | 0.4305 | **0.4246** |
| **test 預期** | winner 約 0.8565（±0.0008） | 0.4291 | **約 0.4231** |
| 不用比分推移（`--preset noprog`，winner 0.6912） | | 0.3907 | 0.3863 |

放不回去的 75 個打散 rally 另外用不含比分推移的集成（action 6 個、point 5 個、winner 3 個方法，配方見報告）。分深度決策與規則修正（第 22 名）在 CV 上都沒有帶來穩定增益（< 0.0005），未採用。

集成在 CV 與 test 上的預測分佈一致：預測落點 0 的比例 CV 38.9%、test 40.1%；action / point 的總變異距離 0.090 / 0.072，來自 test 的選手組成與 train 不同（test 有 30% 的列，下一拍擊球者沒在 train 出現過）。

### 各版本比較（都用新的評估方式）

| 版本 | overall 巢狀估計 | test 預期 | 預測落點 0 的比例：CV → test | 估計的隱藏損失（overall） |
|---|---:|---:|---|---:|
| 改善回合 1 的模型（PR #4 的集成） | 0.4149 | 0.4110 | 約 30% → 6% | 約 −0.008 |
| 回合 2，樹模型仍用 test 內部轉移 | 0.4260 | 0.4244 | 41% → 1.4% | 約 −0.023 |
| **回合 2（目前採用）** | **0.4246** | **0.4231** | 39% → 40% | — |

不用比分推移的版本（`--preset noprog`）：巢狀估計 0.3821 → 0.3863。

- 「隱藏損失」是 CV 看不到的部分：樹模型誤用 test 內部轉移，在 test 上幾乎不預測落點 0。估計方式是在 CV 上把類別 0 的機率壓低到與 test 相同的預測比例，再算 Macro-F1（回合 1：point −0.019；回合 2 未修正：point −0.052、action −0.004），前提是 test 的終結拍比例如 F0 的估計。扣掉之後，回合 1 與「回合 2 未修正」的 test 預期都只有約 0.40。
- 第 2、3 列的巢狀估計只差 0.0014，在雜訊範圍內：拿掉 test 內部轉移對 CV 分數沒有影響，影響的是 test。
- 回合 1 → 回合 2（目前採用）在 CV 上看得到的進步，用依 match 的成對 bootstrap 量化（`python tools/compare_versions.py outputs/r1 outputs`；兩邊都是 in-sample 挑的集成、決策規則跨半調參）：

| | 回合 1 | 回合 2 | 差 | 95% 區間 |
|---|---:|---:|---:|---|
| action | 0.4260 | 0.4336 | +0.0076 | [+0.0027, +0.0116] |
| point | 0.2012 | 0.2174 | +0.0162 | [+0.0107, +0.0228] |
| winner | 0.8387 | 0.8623 | +0.0236 | [+0.0203, +0.0261] |
| **overall** | 0.4186 | 0.4329 | **+0.0142** | [+0.0109, +0.0176] |

（這張表把 5 次重複抽樣合在一起計分，所以絕對值與 arena 報告的略有不同；300 次重抽中 overall 的差全部大於 0。）

### 截斷假設的敏感度

F0 只有 k = 1 的終結率被資料直接支持。如果 k ≥ 2 的終結率其實比較低，集成的分數會怎麼變（`python tools/sensitivity.py outputs/r1 outputs`；決策規則仍在 F0 抽樣上調參，只換評估樣本）：

| 假設 | 整體終結率 | 回合 1 overall | 回合 2 overall | 差 |
|---|---:|---:|---:|---:|
| F0（正式評估） | 0.432 | 0.4133 | 0.4280 | +0.0147 |
| M1：k = 1 照 F0，k ≥ 2 照所有前綴等權 | 0.300 | 0.4146 | 0.4294 | +0.0148 |
| 所有 k 都照所有前綴等權（回合 1 訓練時的隱含假設） | 0.210 | 0.4244 | 0.4297 | +0.0053 |

三種假設下回合 2 都比較好；即使 test 的終結拍其實像 train 的所有前綴那麼少，也沒有變差。（這張表是 CV 上的數字，不含上面說的隱藏損失。）

### 單一方法（節錄）

| 方法 | action F1 | point F1 | winner AUC | overall |
|---|---:|---:|---:|---:|
| `transformer_hist2_prog_al`（雙方 history + 比分推移，加權） | 0.4152 | 0.1994 | 0.8465 | **0.4151** |
| `transformer_hist2_al`（加權） | 0.4125 | 0.1879 | 0.6865 | 0.3775 |
| `transformer_hist2`（不加權） | **0.4184** | 0.1895 | 0.6565 | 0.3745 |
| `gru_hist_al`（加權） | 0.4038 | 0.1881 | 0.6867 | 0.3741 |
| `gru_pemb_hist_al`（加權） | 0.3867 | 0.1872 | 0.6828 | 0.3661 |
| `transformer`（第 3 名） | 0.4051 | 0.1789 | 0.6605 | 0.3657 |
| `lgbm_note`（第 3 名特徵，不加選手 TE） | 0.3748 | 0.1834 | 0.6758 | 0.3584 |
| `gru`（第 3 名） | 0.3936 | 0.1808 | 0.6332 | 0.3564 |
| `xgb`（lucas6028） | 0.3684 | 0.1767 | 0.6863 | 0.3553 |
| `gru_t2v_film`（第 5 名） | 0.3910 | 0.1786 | 0.6362 | 0.3551 |
| `gru_mirror`（第 22 名） | 0.3690 | 0.1770 | 0.6341 | 0.3452 |
| `xgb_hist2_prog` / `lgbm_hist_prog`（history + 比分推移，加權） | 0.3898 / 0.3751 | **0.2029** / 0.2015 | — | — |
| `xgb_hist2` / `lgbm_hist2` / `lgbm_hist`（加權） | 0.3902 / 0.3761 / 0.3759 | 0.1941 / 0.1883 / 0.1888 | — | — |
| `xgb_hist2_eq` / `lgbm_hist2_eq` / `lgbm_hist_eq`（不加權） | 0.3972 / 0.3813 / 0.3836 | 0.1870 / 0.1806 / 0.1869 | — | — |
| `cat_prog_bag3` / `cat_prog` / `xgb_prog` / `lgbm_prog`（比分推移） | — | — | **0.8617** / 0.8610 / 0.8608 / 0.8418 | — |
| `cat_hist_w` / `xgb_hist_w` / `lgbm_hist_w`（比分推移 + history） | — | — | 0.8604 / 0.8596 / 0.8451 | — |

完整排行榜見 `outputs/arena_report.md`；`python eval_oof.py <方法…>` 可單獨比較幾個方法。CatBoost 的 action / point（`cat`、`cat_hist`）這一輪沒有重訓（單一方法要 1–1.5 小時，而且前兩輪都沒被選進集成），不在候選裡。

### 主要發現

改善回合 2：

1. **樹模型誤用 test 內部轉移，是 CV 看不到的最大問題**（見上面的專節）。修正後 CV 分數不變，但 test 上預測落點 0 的比例從 1.4% 回到 40%，估計 overall 約 +0.02。這是比較「CV 與 test 的預測分佈」才發現的，現在是 arena 報告的固定檢查。
2. **winner 的進步來自兩件事**：把打散的 rally 放回去（拿不到比分推移的 test 列 14.2% → 4.1%，而且多了間隔 2 的比分檢查點），以及訓練權重對齊截斷機制。巢狀估計的 winner AUC 0.8377 → 0.8617，test 預期 0.8195 → 0.8565。不含比分推移的模型也受益於對齊：XGBoost 0.666 → 0.686、LightGBM 0.649 → 0.676、GRU + history 0.639 → 0.687。原因是「發球方贏 ⇔ rally 長度是偶數」，終結率隨前綴長度怎麼變，直接決定勝負機率。
3. **對齊對 action / point 的單一模型沒有一致的效果**：樹模型加權後 action 平均約 −0.004、point 約 +0.002（有效樣本數只剩 56%）；序列模型 GRU +0.007 / +0.006、Transformer −0.006 / −0.002，都在單一模型的雜訊範圍。所以兩種版本都放進候選，讓貪婪選擇與巢狀估計決定：最後 action 用了不加權的 Transformer 與加權的 GRU、樹模型，point 以加權的比分推移樹模型為主。
4. **比分推移特徵這一輪對 point 有用了**：point 的巢狀估計 0.1945 → 0.2074，最佳的單一 point 模型都帶比分推移。上一輪它在集成層面沒有幫助；對齊之後終結拍佔 43%，「下一拍擊球者是這一分的輸家 → 這一拍很可能是終結拍（落點 0）」這個訊號的份量變大。
5. 評估抽樣改成 F0 之後，上一輪模型的 action 分數變高、point 變低（巢狀估計 0.4127 / 0.2086 → 0.4238 / 0.1945），overall 0.4177 → 0.4149：新的抽樣修正了 k = 1 比例偏高的問題，終結拍的比例也略高。

先前回合的發現（數字是當時的評估方式）：

6. **同場先前資訊（history 特徵）**：每個模型家族都受益——Transformer action +0.020、XGBoost action +0.024 / point +0.011、GRU（選手嵌入版）point +0.010。關鍵是 train 端的模擬：只用 test 實際可見的 rally 比例與前綴長度分佈，否則 train 的統計量會比 test 豐富得多。
7. **選手身分是最重要的訊號之一**：拿掉選手 ID，action F1 從 0.369 掉到 0.312。第 3 名的選手 target encoding 原寫法有標籤洩漏，修正成巢狀 OOF 後仍不如直接用原始選手 ID。再加入「目前擊球者 ID」反而變差（action −0.006）。
8. **左右鏡像增強（第 22 名）在這份資料無效**：套到樹模型 action −0.017、point −0.004，GRU 鏡像版也是最弱的 NN。
9. **seed bagging 與調整 LightGBM 大小 / 正則化都沒有穩定增益**。
10. **比分推移特徵**讓 winner AUC 從 0.69 提升到 0.86，overall 約 +0.04。

### 分數上限：為什麼離 0.5 還很遠

要在 winner AUC 0.86 下達到 overall 0.5，action + point 的 Macro-F1 合計需 0.82，目前約 0.63。以目前集成在 OOF 上的表現（截斷機制對齊的抽樣）：

| | top-1 正確率 | top-3 正確率 | Macro-F1 |
|---|---:|---:|---:|
| action | 51% | 85% | 0.43 |
| point | 46% | 77% | 0.21 |

point 是瓶頸。落點 0（終結拍，佔 43.5%）的 F1 是 0.60：要預測的那一拍是不是 rally 的最後一拍，事前只能部分判斷。落點 7 / 8 / 9 的 F1 只有 0.20–0.28；落點 1、3 合計只佔 0.8% 的資料卻佔 point Macro-F1 的 20%，F1 只有 0.10 / 0.02。這反映的是下一拍落點本身的隨機性，不是模型容量問題。原 AI CUP 第 3 名私榜 0.392（評分資料不同，不能直接比較）。再往上追 CV 分數，主要會是在擬合雜訊。

### 已知限制

- Kaggle 公開榜尚未上傳驗證；評分公式沿用 AI CUP 原賽 0.4/0.4/0.2，未能在 Kaggle 頁面確認。
- **截斷機制 F0 在 k ≥ 2 是假設**，不是資料直接證實的（見「截斷機制對齊」與「截斷假設的敏感度」）。另外假設了 test 的 rally 長度分佈與 train 相同。
- 把打散的 rally 放回去的正確率是在 train 上模擬出來的（98.8%），test 上無法直接驗證；還有 75 個放不回去，它們拿不到比分推移、也幾乎沒有同場 history，而 CV 上不含比分推移的集成分數是在有 history 的列上算的，對這 75 列會偏樂觀（佔 test 的 4.1%）。
- 放回打散的 rally 與比分推移特徵一樣，用的是 test 檔本身的欄位（選手、比分），是否符合課堂競賽規則請自行判斷。`--preset noprog` 不用比分推移特徵，但 history 特徵仍用到放回去的 rally；要完全不用，訓練與 arena 都要加 `--no-relocate`。
- 樹模型不再用 test 內部轉移，等於放棄了「test 選手在 test 裡怎麼打」這份資訊（history 特徵保留了同一場較早 rally 的部分）。序列模型仍然使用，檢查過平均 P(落點 0) 在 CV 與 test 一致，但更細的偏差（例如其他類別）沒有逐一檢查。CV 無法衡量這份資訊在 test 上的好處或壞處。
- 「隱藏損失」是估計值：假設 test 的終結拍比例如 F0，而且只模擬了類別 0 的預測比例，沒有模擬其他類別的變化。
- 樹模型與 NN 的 early stopping 用的是驗證 fold（而且驗證列沒有加權），OOF 分數會略為樂觀（各方法同樣受影響，排名比較仍公平）。
- history 特徵的 train 模擬與 test 仍有小差異：test 的攻擊率略高（0.37 vs 0.34）、落點 9 比例略高（0.26 vs 0.23）。
- 類別權重用整個 ap_base 計算（影響極小）。
- 巢狀估計只評一半比賽、而且不含版本挑選這一層；0.002 以內的差異不應視為顯著。成對 bootstrap 的區間量的是抽樣雜訊，兩邊的集成都是 in-sample 挑的。

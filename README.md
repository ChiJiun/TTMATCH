# TTMATCH — 桌球下一拍預測方法競技場

Kaggle 課堂競賽 [Introduction to Data Science Competition — TTMATCH](https://www.kaggle.com/competitions/introduction-to-data-science-competition-ttmatch) 的解法 repo。

> **這個分支（`analysis/post-deadline`）是競賽截止後的分析。** 競賽在 2026-11-16 截止，計入成績的只有 2026-10-29 的那一筆繳交（public 0.41744、private 0.39028），推定是 PR #2 的程式碼產生的（依據見 [SUBMISSIONS.md](SUBMISSIONS.md)），那個版本在 `main`。這裡的改善回合 1–3 都是在那之後做的，沒有計入成績；各版本在 Kaggle 上的實際分數見 [SUBMISSIONS.md](SUBMISSIONS.md)。

本題資料源自 **AI CUP 2026 春季賽「基於時序資料之桌球戰術與結果預測競賽」**（423 隊）。本 repo 把該賽前段班公開的方法放在**同一套資料切分與評分**下比較，淘汰表現差的方法，再把勝出方法的優點結合起來。

## 任務

給定一個 rally 的前 n−1 拍，預測：

| 目標 | 類別 | 指標 | 權重 |
|---|---|---|---|
| `actionId` 下一拍球種 | 0–14（下一拍不會是發球） | Macro-F1 | 0.4 |
| `pointId` 下一拍落點 | 0–9 | Macro-F1 | 0.4 |
| `serverGetPoint` 發球方是否贏得此 rally | 機率 | AUC | 0.2 |

> 評分公式已在 Kaggle 競賽頁確認：`0.4·F1 + 0.4·F1 + 0.2·AUC`，與 AI CUP 原賽相同。

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
| `*_fa`：**終結拍分解**，樹模型安全地使用 test 內部轉移（本 repo 新增，見下） | — | `ttmatch/methods/trees.py` |
| **終結拍更新**：用 winner 集成的勝負機率修正 action / point 集成（本 repo 新增，見下） | — | `arena.py` |

以上程式碼改寫自各隊公開 repo，每個檔案開頭都標註了出處；原作著作權屬原作者。

### 本 repo 新增：比分推移特徵

test 裡同一局會出現多個 rally，同局相鄰可見 rally 的 rally_id 間隔以 2–4 為主（把打散的 rally 放回去之後，2：29%、3：33%、4：20%；另有 14% 是該局最後一個可見 rally、2.5% 是放不回去的打散 rally，沒有下一個）。**下一個可見 rally 的開局比分**揭露了這段期間發球方與接發方各得幾分：若發球方一分都沒拿到，本 rally 一定是接發方得分，反之亦然。在 train 上以「同局下一個存在的 rally」驗證：只要其中一方得分為 0（可判定勝方），推得的勝方與 `serverGetPoint` 一致率為 100%（13,718 / 13,718 個 rally）。

為了讓 train 和 test 的特徵分佈一致，train 端會依 test 實際的 rally 間隔分佈為每個 rally 抽一個間隔，取 rally_id ≥ r+間隔的第一個存在 rally 當作「下一個可見 rally」（`score_progression_train`）。這個特徵只用到 test 檔案本身提供的欄位，不需要任何外部標註。

兩個 rally 必須通過一致性檢查才會使用：同一對選手、`rally_id − 1 == 兩方比分和`、得分非負且相加等於間隔。test 的 match 340–354（262 個 rally）每場出現 21–31 位選手、rally_id 與比分不符；其中 215 個能放回原本的比賽（見下一節），剩下 47 個不提供比分推移資訊，arena 對這 47 個 rally 改用不含比分推移的集成。

> ⚠️ 這個特徵是從 test 檔中其他 rally 的比分欄位推得勝負。它不用外部資料，但效果極大（winner AUC 約 0.67 → 0.83）。是否符合課堂競賽規則，請自行判斷；不想用時執行 `python arena.py --preset noprog --tag noprog`，會另外產生 `outputs/submission_noprog.csv`。

### 本 repo 新增：同場先前資訊（history）特徵

train 與 test 的比賽不重疊，但 test 每場比賽平均可見 23 個 rally。「下一拍擊球者在這場比賽較早的 rally 裡怎麼打」（各球種、各落點的比例、攻擊率、平均力道，往全體分佈平滑）是選手 ID 之外合法可用的訊息：只用該 rally **之前**的 rally，不碰任何要預測的拍（`ttmatch/history.py`）。`*_hist2` 再加上目前擊球者（對手）的同一組統計。

分佈對齊是關鍵。test 每局只看得到約 34% 的 rally，每個 rally 只看得到前綴 k 拍，而且第 k+1 拍一定存在（那是要預測的拍），所以看不到 rally 的最後一拍。train 端先以 34% 機率保留 rally（與 rally 長度無關），再把 test 的前綴長度分佈「由長到短」配對到夠長的 rally 上（偏向長 rally 的只有前綴長度 k 的分配，不是哪些 rally 可見），使 train 的可見前綴長度分佈與 test 幾乎相同（每個可見 rally 平均 3.07 拍，test 也是 3.07）。

### 本 repo 新增：把打散的 test rally 放回原本的比賽

test 的 match 340–354 不是真的比賽。這 262 個 rally 的選手組合**全部**出現在其他正常的 test match 裡，選手 ID、性別與開局比分都是真的，只有 `match`、`numberGame`、`rally_id` 三個編號被打亂——它們是正常 test match 裡沒有公開的 rally。

正常資料中 `rally_id − 1 = 開局比分總和`，所以真實的 rally_id 可以直接由比分算出，剩下要找的是哪一場、哪一局。候選的局必須同時滿足：同一對選手與性別、該局沒有公開過這個 rally_id、比分夾在前後已知 rally 之間（雙方得分都不倒退）、發球者符合發球輪替、該局在這個比分下還沒結束。只用這 5 個條件就能唯一定位的 rally 有很強的位置規律，所以再加兩個必要條件，7 個條件都滿足的候選只剩一個才放回去（`ttmatch/relocate.py`）：

- 條件 6：剛好在「前一個公開 rally 的後 2 分」（只用條件 1–5 唯一定位的 126 個裡有 124 個如此）。
- 條件 7（改善回合 3 新增）：該局下一個公開 rally 剛好在後 2 分、而且這 2 分**雙方各得 1 分**，或者該局之後沒有公開的 rally。只用條件 1–6 唯一定位的 183 個裡有 177 個如此；其中「下一個公開 rally 在後 2 分」的 95 個全部是各得 1 分，沒有任何一個是同一方連得 2 分，而 train 的連續 rally 有 49% 是同一方連得 2 分。候選能不能唯一定位與這 2 分怎麼分無關，所以這是主辦單位挑選打散 rally 的方式，不是定位過程造成的。

結果：

- 放回 **215 / 262** 個（只用條件 1–6 是 187 個；其中 180 個位置相同，另外 7 個不符合條件 7 的不再放回）。
- 在 train 上模擬同樣的公開 / 打散樣式（`python tools/relocation_sim.py`）：條件 1–6 放回 82%、其中 99.8% 位置正確；條件 1–7 放回 92%、99.8% 位置正確（20 次重複，每次約 106 個打散 rally）。模擬是照 test 觀察到的樣式產生打散 rally 的，所以它量的是「樣式如同觀察到的那樣時，候選唯一是否代表位置正確」，不能證明樣式本身；test 實際放回的比例（82%）比模擬低，表示 test 還有模擬沒重現的結構。
- 放回去的 rally 自己有了比分推移與同場 history 特徵；它前面的公開 rally 也多了一個更近的比分檢查點（間隔 4 變成 2）。
- 拿不到比分推移資訊的 test 列從 14.2% 降到 2.5%。剩下 47 個放不回去的 rally 也幾乎沒有同場 history。

這個挑選方式有一個副作用：打散的 rally 與下一個公開 rally 之間永遠是各得 1 分，所以 test 裡「比分能直接確定勝負」的 rally 比 train 端的模擬少（間隔 2 的列：test 37%、train 模擬 49%；所有有比分推移資訊的列：test 21.1%、train 模擬 24.6%）。模型學的是「給定間隔與雙方得分」的條件機率，不受這個組成差異影響；受影響的是 CV 的 winner 分數（CV 的組成偏容易），`tools/sensitivity.py` 有一個情境把 CV 重新加權成 test 的組成來量這個差距（見「截斷假設的敏感度」）。

只用到 test 已公開的欄位（選手、比分），不涉及任何答案；性質與比分推移特徵相同，不想用時加 `--no-relocate`（`arena.py`）或用 `--preset noprog`。

### 本 repo 新增：截斷機制對齊

資料上的兩個事實（train 中分別是 100% 與 99.9%）：**rally 的最後一拍 `pointId` 一定是 0**，而且**最後一拍的擊球者輸掉這一分**。所以「要預測的那一拍是不是 rally 的最後一拍」（以下稱終結拍）同時決定了落點 0、球種分佈與勝負。

train 的每個 rally 提供所有前綴（k = 1..L−1），其中終結拍只佔 21.5%。test 每個 rally 只有一個前綴，終結拍佔多少取決於主辦單位怎麼截。假設每個 rally 從一個固定的分佈 g 抽截斷位置 k、抽到超過 L−1 就重抽（稱為 F0），再假設 test 的 rally 長度分佈與 train 相同，g 就能由 test 的前綴長度分佈解出（`ttmatch/truncation.py`；k < 9 逐格吻合，k ≥ 9 的 test 樣本太少、只調一個共同倍率，邊際分佈與 test 的最大差是 0.002）。F0 之下終結拍佔 **43.2%**，是未加權訓練資料的兩倍。

證據不用任何 test 答案：比分推移揭露了部分 test rally 的勝負，而勝負等同 rally 長度的奇偶，所以每個機制都能算出這份比分資料的對數概似（`python tools/truncation_evidence.py`）。

| 機制 | 對數概似 | 整體終結率 | k=1 / 2 / 3 的終結率 | k 邊際與 test 的最大差 |
|---|---:|---:|---|---:|
| 要預測的那一拍從來不是最後一拍 | −1832.5 | 0 | 0 / 0 / 0 | 0.064 |
| 所有前綴等權（未加權的訓練資料） | −1803.4 | 0.215 | 0.125 / 0.197 / 0.287 | 0.069 |
| 上一版的評估抽樣（k 照 test 分佈抽，超過就截到 L−1） | −1802.3 | 0.420 | 0.341 / 0.408 / 0.530 | 0.090 |
| **F0** | **−1794.2** | 0.432 | 0.453 / 0.422 / 0.502 | 0.002 |

F0 比「等權」高 9.2（約 1 萬倍的概似比），比「從來不是最後一拍」高 38；與等權的差異幾乎全部來自 k = 1：長度 2 的 rally 只能截在 k = 1，所以 test 中 k = 1 的終結率是 P(L=2) / P_test(k=1) ≈ 0.45（比分資料單獨估出來是 0.47，95% 區間 0.33–0.61），等權時只有 0.125。**k ≥ 2 的終結率，比分資料分辨不出 0.2–0.45 之間的值**（各 k 的 95% 區間都很寬，k = 2、4、5、6 的點估計比 F0 低，其中 k = 6 的 95% 區間 [0, 0.30] 不包含 F0 給的 0.36），這一段是機制假設推出來的，不是資料直接證實的；「結果」一節有對這個假設的敏感度分析。

用法有兩處：

1. **訓練權重**：每個訓練列的權重是 `g(k) / Σ_{j ≤ L−1} g(j)`，加權後的列分佈就是 F0 下的 test 分佈（有效樣本數約為原本的 56%）。test 內部轉移（一定不是終結拍）併入訓練時，各前綴長度的終結拍比例維持不變（`blend_aug_weights`）。
2. **評估抽樣**：每個驗證 rally 依 g(k) 抽一個前綴（`cut_matched_indices`）。上一版的抽樣把超過的 k 截到 L−1，k = 1 的比例因此偏高（36.7% vs test 的 27.5%）。

### 本 repo 的修正：樹模型不用 test 內部轉移擴充

第 3 名的作法把 test rally 內部的轉移（前綴 k → 第 k+1 拍，k 小於可見長度）也加進訓練，讓模型看到 test 選手的打法。這些列有兩個特性：它們和「真正要預測的那一列」來自同一個 rally（選手、比分、前幾拍都一樣），而且**一定不是終結拍**（下一拍還看得到）。樹模型因此學到「長得像這個 rally 的列不會是終結拍」，對要預測的那一列幾乎不預測落點 0：

| LightGBM + history | 平均 P(落點 0)：CV | 平均 P(落點 0)：test | test 中下一拍擊球者沒在 train 出現過的列 |
|---|---:|---:|---:|
| 使用 test 內部轉移（`lgbm_hist_aug`） | 0.41 | **0.07** | 0.02 |
| 不使用（`lgbm_hist`，現行作法） | 0.39 | 0.40 | 0.43 |

CV 完全看不到這個偏差：驗證 fold 的 rally 沒有自己的前綴在訓練集裡。它是在比較「集成在 CV 與在 test 上預測各類別的比例」時發現的——未修正時 CV 有 41% 預測落點 0，test 只有 1.4%。若 test 的終結拍比例如上一節估計的 43%，這會讓 point Macro-F1 少約 0.05（把 CV 上類別 0 的機率壓到同樣的預測比例來估計），改善回合 1 的提交檔也有同樣的問題（test 只有 4% 預測落點 0，估計 point 少約 0.02）。

序列模型沒有這個現象（test 與 CV 的平均 P(落點 0) 一致），仍然使用這些列。`arena.py` 的報告固定列出「CV 與 test 的預測分佈」，差異過大時會警告；「預測為落點 0 的比例」另外單獨檢查（CV 與 test 差超過 0.05 就警告）。沒指定 `--preset` 時，這兩個對照用的方法（`lgbm`、`lgbm_hist_aug`）預設不參與集成。

### 本 repo 新增：終結拍分解（樹模型安全地使用 test 內部轉移）

上一節的修正等於放棄了「test 選手在 test 裡怎麼打」這份資訊。改善回合 3 把問題拆開（`ttmatch/methods/trees.py` 的 `train_factored`，方法名稱後綴 `_fa`）：

    P(類別 | x) = P(終結 | x) · P(類別 | 終結, x) + (1 − P(終結 | x)) · P(類別 | 非終結, x)

- `P(終結 | x)`：二元模型，只用 train。
- `P(類別 | 非終結, x)`：多類別模型，用 train 的非終結列 **加上 test 內部轉移**。test 內部轉移全部是非終結拍，放進這個條件模型不會改變任何終結拍的比例。
- `P(類別 | 終結, x)`：落點一定是 0；球種另訓練一個只用 train 終結列的模型。

| LightGBM + history | action F1 | point F1 | 平均 P(落點 0)：CV | test | test 中選手沒在 train 出現過的列 |
|---|---:|---:|---:|---:|---:|
| 直接分類、不用 test 內部轉移（`lgbm_hist`） | 0.3765 | 0.1879 | 0.39 | 0.40 | 0.43 |
| 直接分類、使用（`lgbm_hist_aug`，有偏差的對照） | 0.3883 | 0.1907 | 0.41 | **0.07** | 0.02 |
| 終結拍分解、不用（`lgbm_hist_f`，對照） | 0.3798 | 0.1857 | 0.40 | 0.41 | 0.41 |
| **終結拍分解、使用（`lgbm_hist_fa`）** | **0.3897** | 0.1885 | 0.39 | 0.41 | 0.41 |

（`lgbm_hist_aug` 與 `lgbm_hist_f` 是放回規則更新前訓練的，其餘是更新後；差異在 0.001 以內。）分解版拿回了 test 內部轉移在 action 上的好處（LightGBM 各變體 +0.005～0.013、XGBoost 不含 history 的版本 +0.015、含 history 的版本持平），而 test 上的終結拍比例與 CV 一致。point 沒有一致的變化。CV 的驗證選手不在 test 裡，所以 CV 量到的只是「多了一批非終結拍訓練資料」的效果；它對 test 選手（30% 的列，下一拍擊球者沒在 train 出現過）額外的好處 CV 量不到。

**`_fa` 在 test 上的行為與 CV 不同，最後沒有採用。** 上表檢查的是終結拍的比例，看不到另一種差異：模型有多大把握。`_fa` 的非終結條件模型在 test 上，每一列都有同一個 rally、同一場比賽的內部轉移可學；CV 的驗證 fold 沒有。量「機率分佈的平均最大機率」（落點只看非終結拍的條件分佈）：

| | action：CV | test | 差 | point（非終結）：CV | test | 差 |
|---|---:|---:|---:|---:|---:|---:|
| `lgbm_hist`（樹模型，不用 test 內部轉移） | 0.545 | 0.523 | −0.022 | 0.342 | 0.339 | −0.002 |
| `transformer_hist2`（序列模型，使用） | 0.554 | 0.547 | −0.007 | 0.338 | 0.338 | +0.000 |
| `lgbm_hist_prog_fa` | 0.534 | 0.536 | +0.002 | 0.353 | 0.413 | **+0.059** |
| `xgb_hist2_prog_fa` | 0.517 | 0.551 | **+0.034** | 0.340 | 0.440 | **+0.100** |

不用 test 內部轉移的樹模型與序列模型，兩邊的差都在 ±0.03 以內；`_fa` 的落點在 test 上明顯更有把握。這可能是好事（真的學到 test 選手的打法），也可能是壞事（對同一個 rally 的前幾拍過度擬合），CV 分不出來。`arena.py` 的報告現在固定列出這張表，差的絕對值超過 0.03（`CONF_WARN`）就警告。這個門檻是看過上表才定的，離不用 test 內部轉移的樹模型（−0.022）不遠，只能當作提醒，不是嚴格的檢定。

把 7 個 `_fa` 方法從候選拿掉（`--preset nofa`），巢狀估計是 0.4309，含 `_fa` 是 0.4289；in-sample 的成對 bootstrap 則是含 `_fa` 略高 0.0007（95% 區間 [−0.0014, +0.0028]）。兩種比較都在雜訊範圍內，也就是 **CV 量不到 `_fa` 進集成後的增益**。單看 point，兩種比較的方向相反：in-sample 是含 `_fa` 高 0.0026（95% 區間 [−0.0054, +0.0002]，依 match 重抽 300 次約 97% 偏向含 `_fa`），巢狀估計則是不含 `_fa` 較高（0.2149 對 0.2124）；action 兩種比較都是不含 `_fa` 持平或略高。既然沒有可測的好處、又有 CV 看不到的行為差異，最後採用不含 `_fa` 的版本：CV 分不出高下時，選比較簡單、沒有 CV 看不到的行為差異的那個。（實際的結果是用 `--exclude` 排除這 7 個方法跑的；`--preset nofa` 是之後加上的寫法，候選清單與順序相同。）

**決定的時間點：** 含 `_fa` 的回合 3 版本在 2026-11-17 00:52 已經上傳 Kaggle（public 0.44107、private 0.40235，見 [SUBMISSIONS.md](SUBMISSIONS.md)），不含 `_fa` 的消融與這個決定是在那之後（同日約 06:20）才做的，所以做決定時已經看過含 `_fa` 版的 Kaggle 分數。決定依據的是上面的 CV 比較與把握程度檢查；不含 `_fa` 的版本在決定時還沒上傳，沒有它的 Kaggle 分數可以比。

### 本 repo 新增：終結拍更新（用勝負機率修正 action / point）

終結拍的擊球者一定輸掉這一分，所以勝負資訊可以反過來修正「下一拍是終結拍」的機率。落點模型給的 P(落點 0) 記為 p0，模型隱含的「下一拍擊球者輸掉這一分」的機率是 p0 + (1 − p0)·q_k（q_k 是前綴長度 k、下一拍不是終結拍時，擊球者最後仍輸掉的比例，由 train 估計）。winner 集成（含比分推移）對同一件事有更準的機率，兩者的比值就是修正倍率（`arena.py` 的 `terminal_update`）：

    P(終結) ← p0 · (winner 集成的 P(擊球者輸) / 模型隱含的 P(擊球者輸)) ^ λ

落點的類別 0 換成更新後的機率，其餘類別依原比例分配；球種依終結 / 非終結拍的球種先驗比調整。底可以是不含比分推移的集成（勝負資訊只從 winner 進來一次）或主集成，λ ∈ {0.5, 0.75, 1}；arena 把這些更新版與原集成一起做貪婪選擇，有進步才採用，巢狀估計也在「另一半比賽」重做這個選擇。test 中放不回去的打散 rally 沒有比分推移，不做更新。

## 使用方式

```bash
python -m venv .venv
.venv/Scripts/activate          # macOS/Linux：source .venv/bin/activate
pip install -r requirements.txt  # torch 可改裝 CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cpu
```

資料：把 Kaggle API token 放在 `~/.kaggle/kaggle.json`，程式會用 `kagglehub` 自動下載；或手動把 `train.csv`、`test.csv`、`sample_submission.csv` 放到 `data/`。

```bash
python run_method.py --list                              # 列出所有方法
python run_method.py lgbm_note xgb lgbm_hist_fa cat_prog  # 訓練方法，輸出 outputs/oof/<name>.npz
python run_method.py gru transformer_hist2 --device cpu  # 序列模型（本機 CPU 可跑，較慢）
python arena.py --preset final                           # 比較、淘汰、集成 → outputs/arena_report.md、outputs/submission.csv
python arena.py --preset noprog --tag noprog             # 不用比分推移特徵的版本 → outputs/submission_noprog.csv
python eval_oof.py lgbm_note lgbm_hist                   # 只比較幾個方法的單模分數（實驗用，比 arena 快很多）
```

同時跑多個訓練行程時，用環境變數 `TTMATCH_THREADS=4` 限制每個樹模型行程的執行緒數（CatBoost 預設會佔滿所有核心）。GPU 不夠時，用 [`notebooks/colab_train_nn.ipynb`](notebooks/colab_train_nn.ipynb) 在 Colab 訓練序列模型，再把 `outputs/oof/*.npz` 下載回本機。

檢查用的工具（都不訓練；除了最後一個，都讀 `outputs/` 裡的 npz 與報告）：

```bash
python tools/truncation_evidence.py                 # 截斷機制的證據（對數概似、各前綴長度的終結率估計）
python tools/terminal_bias.py lgbm_hist_aug lgbm_hist  # 各方法在 CV 與 test 上的平均 P(落點 0)
python tools/sensitivity.py outputs                 # 截斷假設不成立、或比分推移組成照 test 時，集成的分數差多少
python tools/compare_versions.py outputs/r2 outputs # 兩個版本的差距與 95% 區間（依 match 成對 bootstrap）
python tools/hidden_loss.py outputs                 # 集成在 test 上預測類別 0 的比例與 CV 不同時，估計損失多少
python tools/relocation_sim.py                      # 在 train 上模擬打散 / 放回，量放回的比例與正確率（不需要 outputs/）
```

## 評估方式（所有方法一致）

1. **依 match 分組的 5-fold**：test 的 79 場比賽與 train 的 216 場完全不重疊，按 match 切才能反映真實泛化。
2. **截斷機制對齊**：評估時每個驗證 rally 依估計出來的 test 截斷機制抽一個前綴（重複 5 次），前綴長度分佈與終結拍比例都對齊 test（見上面「截斷機制對齊」）。
3. **誠實的決策調參**：Macro-F1 決策規則在「另一半比賽」上調參、在這一半評估，兩半互換取平均。
4. **優勝劣汰**：每個任務各自做貪婪前向選擇，從最佳單一方法開始，只有能讓誠實分數上升的方法才會加入，其餘淘汰。
5. **終結拍更新**：用 winner 集成的勝負機率修正 action / point 集成（見上面的專節），有進步才採用。
6. **巢狀估計**：集成成員、權重、終結拍更新與決策規則全部只用「另一半比賽」挑選，再在這一半評分，兩半互換。這是沒有選擇偏差的估計，用來判斷一個改動是否真的有效（`arena.py` 的 `nested_estimate`）。
7. **CV 與 test 的預測分佈檢查**：同一組決策規則下，集成在 CV 抽樣與在 test 上「預測為各類別」的比例應該接近；差很多代表 test 端的預測與 CV 不一致，這種問題 CV 分數看不出來（`arena.py` 的 `prediction_shift`）。

## 結果

5-fold、依 match 分組、截斷機制對齊、決策規則跨半調參。重現：`python arena.py --preset nofa --tag nofa`（報告在 `outputs/arena_report_nofa.md`）；含 `_fa` 的版本是 `--preset final`。

> 評估抽樣在改善回合 2 改了（見「截斷機制對齊」），**以下數字不能和回合 1 以前 README 的數字直接比較**。回合 2 與回合 3 用同一套評估抽樣，可以直接比較。

序列模型都在本機 CPU 訓練。含 history 的 5 個（`transformer_hist2`、`transformer_hist2_al`、`transformer_hist2_prog_al`、`gru_hist_al`、`gru_pemb_hist_al`）這一輪用 2 個 seed × 16 epoch 重訓（回合 2 是 1 個 seed × 12 epoch）。不含 history 的 6 個（`gru`、`transformer`、`gru_pemb`、`transformer_pemb`、`gru_t2v_film`、`gru_mirror`）沿用改善回合 1 訓練的結果（不用 history 與比分推移特徵），沒有隨放回規則的更新重訓。不含 history 的 4 個樹模型（`xgb`、`lgbm_note`、`xgb_eq`、`lgbm_note_eq`）則在放回規則更新後重訓過：放回會改變約 39 個 test 列的 `numberGame`。LightGBM 重訓後的 OOF 與原本逐位元相同；XGBoost 在不同執行緒數下不能逐位元重現，OOF 有雜訊等級的差異（argmax 約 90% 相同）。

### 三種分數怎麼看

| 分數 | 意義 |
|---|---|
| in-sample | 集成成員、權重、終結拍更新、決策都在同一份 OOF 上挑，含選擇偏差，偏樂觀 |
| **巢狀估計** | 成員、權重、終結拍更新、決策只用「另一半比賽」挑選，在這一半評分，兩半互換；集成內部的挑選沒有選擇偏差，**用來決定採用哪個版本** |
| **test 預期** | 再模擬 test 有 2.5% 的列（放不回去的打散 rally）拿不到比分推移、只能用不含比分推移的集成。折減量是用 in-sample 的集成估的（主集成 − 混合），再從巢狀估計扣掉 |

注意：巢狀估計不含「版本挑選」這一層；分深度決策、規則修正沒有在巢狀估計裡重做（兩者最後都沒有採用）。巢狀估計每次只評一半比賽，單看它的雜訊不小；版本之間的比較另外用依 match 的成對 bootstrap 量化（見下）。

### 最終集成（`--preset nofa`，30 個方法參與）

| 任務 | 留下的方法（權重） | 終結拍更新 | in-sample | 巢狀估計 |
|---|---|---|---:|---:|
| action | 不含比分推移的集成：transformer_hist2 0.35、xgb_eq 0.22、xgb_hist2 0.18、lgbm_hist 0.15、gru_pemb_hist_al 0.10 | 以這個集成為底，λ = 1 | 0.4301 | 0.4298 |
| point | lgbm_hist_prog 0.50、xgb_hist2_prog 0.30、transformer_hist2 0.20 | 0.9 ×（以不含比分推移的集成為底，λ = 0.75）+ 0.1 × 原集成 | 0.2168 | 0.2149 |
| winner | cat_prog_bag3 0.70、xgb_prog 0.30 | — | 0.8661 | 0.8653 |
| **overall** | | | 0.4320 | **0.4309** |
| **test 預期** | winner 約 0.8622（±0.0005） | | 0.4309 | **約 0.4298** |
| 含 `_fa`（`--preset final`，37 個方法） | | | 0.4326 | 0.4289 |
| 不用比分推移（`--preset noprog`，winner 0.6926） | | 未採用（沒有進步） | 0.3897 | 0.3871 |

- 不做終結拍更新時的巢狀估計：action 0.4276、point 0.2072、overall 0.4270；終結拍更新貢獻 +0.0039（含 `_fa` 的版本是 +0.0018）。
- action 的主集成（transformer_hist2 0.39、xgb_eq 0.24、transformer_hist2_prog_al 0.20、lgbm_hist 0.17）含比分推移，但終結拍更新的貪婪選擇只留下「以不含比分推移的集成為底」的版本，所以最後的 action 沒有直接用到比分推移的方法，勝負資訊由終結拍更新帶進來。
- 放不回去的 47 個打散 rally 另外用不含比分推移的集成（action 5 個、point 2 個、winner 3 個方法，配方見報告），不做終結拍更新。
- 分深度決策與規則修正（第 22 名）在 CV 上都沒有帶來穩定增益，未採用。
- 集成在 CV 與 test 上的預測分佈一致：預測落點 0 的比例 CV 40.5%、test 40.6%；action / point 的總變異距離 0.100 / 0.076，來自 test 的選手組成與 train 不同（test 有 30% 的列，下一拍擊球者沒在 train 出現過）。平均最大機率在 CV 與 test 的差是 action −0.011、point +0.003，沒有成員被標出來。
- **test 預期沒有扣掉 winner 在 CV 偏樂觀的部分**（見下面敏感度表的最後一列），overall 要再少約 0.001。
- **集成配方對雜訊很敏感**：4 個不含 history 的樹模型重訓後，含 `_fa` 的版本巢狀估計 0.4290 → 0.4289，幾乎沒變，但 action 的成員換了一半（重訓前：transformer_hist2、xgb_hist2_eq、gru_pemb_hist_al、lgbm_hist、xgb_eq；重訓後：transformer_hist2、lgbm_hist、xgb_eq、transformer_hist2_prog_al、xgb_hist2_prog_fa、lgbm_note）。候選方法之間的差距小於雜訊，誰被選進去不代表它比較好。

### 各版本比較（都用新的評估方式）

| 版本 | overall 巢狀估計 | test 預期 | 預測落點 0 的比例：CV → test | 估計的隱藏損失（overall） |
|---|---:|---:|---|---:|
| 改善回合 1 的模型（PR #4 的集成） | 0.4149 | 0.4110 | 約 30% → 6% | 約 −0.008 |
| 回合 2，樹模型仍用 test 內部轉移 | 0.4260 | 0.4244 | 41% → 1.4% | 約 −0.023 |
| 回合 2（PR #6） | 0.4246 | 0.4231 | 39% → 40% | 約 0 |
| 回合 3（PR #7，含 `_fa`） | 0.4290 | 0.4280 | 44% → 43% | 約 0 |
| 回合 3，4 個樹模型重訓後重跑（含 `_fa`） | 0.4289 | 0.4279 | 44% → 43% | 約 0 |
| **回合 3，不含 `_fa`（目前採用）** | **0.4309** | **0.4298** | 40.5% → 40.6% | 約 0 |

不用比分推移的版本（`--preset noprog`）：巢狀估計 0.3821 → 0.3863 → 0.3871。

Kaggle 上的實際分數（回合 2、回合 3 都是截止後才上傳，不計成績；每一筆的版本與配方記錄在 [SUBMISSIONS.md](SUBMISSIONS.md)）：

| 版本 | CV 的 test 預期 | public（67% 的 test） | private（33%） |
|---|---:|---:|---:|
| 準時繳交（PR #2，在 `main`） | 0.4055（當時的評估方式） | 0.41744 | 0.39028 |
| 回合 2（PR #6） | 0.4231 | 0.42919 | 0.40268 |
| 回合 3（PR #7，含 `_fa`） | 0.4280 | 0.44107 | 0.40235 |
| 回合 3，不含 `_fa` | 0.4298 | 尚未上傳 | 尚未上傳 |

回合 2、回合 3 在 private 上比準時繳交高約 0.012；回合 2 → 回合 3 在 public 多 0.012，在 private 沒有差別，與下面 bootstrap 的結論一致。private 只有約 609 列，單筆之間的小差距不宜過度解讀。準時繳交那一列的「CV 的 test 預期」是當時的評估方式（沒有對齊 test 的截斷機制），與其他列不能直接比較。這些分數沒有拿來調參；最後採用不含 `_fa` 的版本，是在看過回合 3 的 Kaggle 分數之後依 CV 決定的（見上面「`_fa` 在 test 上的行為與 CV 不同」）。

- 這張表的每一列，是把該版本的 npz 放在各自的資料夾（`outputs/r1`、`outputs/r2_aug`、`outputs/r2`、`outputs/`），用 `TTMATCH_OUT_DIR=<資料夾> python arena.py --preset <r1 或 final>` 計分。`outputs/` 不進版控，所以前三列無法只靠 repo 重現（要用當時的程式碼重訓）；前三列是用 PR #6 的程式碼算的，放不回去的列當時是 4.1%。「回合 2，樹模型仍用 test 內部轉移」那一列有 34 個候選方法，其餘回合 2 的列是 30 個。
- 「隱藏損失」是 CV 看不到的部分：樹模型誤用 test 內部轉移，在 test 上幾乎不預測落點 0。估計方式是在 CV 上把類別 0 的機率壓低到與 test 相同的預測比例，再算 Macro-F1（`python tools/hidden_loss.py <資料夾>`；回合 1：point −0.019；回合 2 未修正：point −0.052、action −0.004；回合 2 與回合 3 的 action / point 都在 ±0.001 以內），前提是 test 的終結拍比例如 F0 的估計。扣掉之後，回合 1 與「回合 2 未修正」的 test 預期都只有約 0.40。
- 第 2、3 列的巢狀估計只差 0.0014，在雜訊範圍內。這是集成層級的結果；單一樹模型拿掉 test 內部轉移後，7 個裡有 5 個的 action 掉了 0.006–0.014（例如 `xgb` 0.3825 → 0.3684），集成時由序列模型補上。
- 回合 2 → 回合 3 的差距，用依 match 的成對 bootstrap 量化（`python tools/compare_versions.py outputs/r2 outputs/nofa`，`outputs/nofa/blend.npz` 是 `outputs/blend_nofa.npz` 的複本；兩邊都是 in-sample 挑的集成、決策規則跨半調參）：

| | 回合 2 | 回合 3（不含 `_fa`） | 差 | 95% 區間 | 差 > 0 的比例 |
|---|---:|---:|---:|---|---:|
| action | 0.4336 | 0.4331 | −0.0005 | [−0.0052, +0.0041] | 44% |
| point | 0.2174 | 0.2194 | +0.0020 | [−0.0039, +0.0068] | 79% |
| winner | 0.8623 | 0.8661 | +0.0039 | [+0.0026, +0.0049] | 100% |
| **overall** | 0.4329 | 0.4342 | **+0.0014** | [−0.0019, +0.0042] | 82% |

（這張表把 5 次重複抽樣合在一起計分，所以絕對值與 arena 報告的略有不同。）**回合 3 的進步小**：winner 的進步超出抽樣雜訊，action、point 與 overall 的 95% 區間都包含 0。含 `_fa` 的版本同樣的比較是 overall +0.0021 [−0.0006, +0.0051]（PR #7 當時、樹模型重訓前是 +0.0026 [−0.0012, +0.0056]）。巢狀估計的差（+0.0063）比這裡大，但巢狀估計本身的雜訊約 ±0.002。回合 1 → 回合 2 同樣的比較是 overall +0.0142 [+0.0109, +0.0176]。

### 截斷假設與比分推移組成的敏感度

F0 只有 k = 1 的終結率被資料直接支持。如果 k ≥ 2 的終結率其實比較低，或者 CV 的比分推移組成改成 test 的樣子，集成的分數會怎麼變（`python tools/sensitivity.py outputs/r2 outputs/nofa`；決策規則仍在 F0 抽樣上調參，只換評估樣本）：

| 假設 | 整體終結率 | 比分能確定勝負的列 | 回合 2 overall | 回合 3 overall | 差 | 回合 3 的 winner AUC |
|---|---:|---:|---:|---:|---:|---:|
| F0（正式評估） | 0.432 | 0.246 | 0.4280 | 0.4301 | +0.0021 | 0.8650 |
| M1：k = 1 照 F0，k ≥ 2 照所有前綴等權 | 0.300 | 0.246 | 0.4294 | 0.4314 | +0.0020 | 0.8486 |
| 所有 k 都照所有前綴等權 | 0.210 | 0.247 | 0.4297 | 0.4324 | +0.0027 | 0.8405 |
| F0＋比分推移組成照 test | 0.433 | 0.211 | 0.4275 | 0.4290 | +0.0015 | 0.8500 |

四種情境下回合 3 都比較好。最後一列是這一輪新增的：test 裡「比分能直接確定勝負」的列比 train 端的模擬少（21.1% vs 24.6%，見「把打散的 test rally 放回原本的比賽」），把 CV 重新加權成 test 的組成後，**winner AUC 是 0.850 而不是 0.865**；action / point 在這個情境下略高，overall 少約 0.001。所以上面「test 預期」裡的 winner 0.862 偏樂觀，比較接近 0.85。

### 單一方法（節錄）

| 方法 | action F1 | point F1 | winner AUC | overall |
|---|---:|---:|---:|---:|
| `transformer_hist2_prog_al`（雙方 history + 比分推移，加權） | 0.4151 | 0.2025 | 0.8505 | **0.4171** |
| `transformer_hist2_al`（加權） | 0.4126 | 0.1921 | 0.6881 | 0.3795 |
| `gru_hist_al`（加權） | 0.4089 | 0.1871 | 0.6883 | 0.3760 |
| `gru_pemb_hist_al`（加權） | 0.4031 | 0.1930 | 0.6877 | 0.3760 |
| `transformer_hist2`（不加權） | **0.4177** | 0.1904 | 0.6590 | 0.3750 |
| `transformer`（第 3 名） | 0.4051 | 0.1789 | 0.6605 | 0.3657 |
| `lgbm_note`（第 3 名特徵，不加選手 TE） | 0.3748 | 0.1834 | 0.6758 | 0.3584 |
| `xgb`（lucas6028） | 0.3701 | 0.1774 | 0.6874 | 0.3565 |
| `gru`（第 3 名） | 0.3936 | 0.1808 | 0.6332 | 0.3564 |
| `gru_t2v_film`（第 5 名） | 0.3910 | 0.1786 | 0.6362 | 0.3551 |
| `gru_mirror`（第 22 名） | 0.3690 | 0.1770 | 0.6341 | 0.3452 |
| `xgb_hist2_prog` / `lgbm_hist_prog`（history + 比分推移，加權） | 0.3891 / 0.3754 | 0.2042 / 0.1988 | — | — |
| `xgb_hist2_prog_fa` / `lgbm_hist_prog_fa`（同上，終結拍分解 + test 轉移） | 0.3914 / 0.3827 | 0.1987 / **0.2061** | — | — |
| `xgb_hist2` / `lgbm_hist2` / `lgbm_hist`（加權） | 0.3853 / 0.3758 / 0.3765 | 0.1896 / 0.1905 / 0.1879 | — | — |
| `xgb_hist2_fa` / `lgbm_hist2_fa` / `lgbm_hist_fa`（終結拍分解 + test 轉移） | 0.3859 / 0.3843 / 0.3897 | 0.1858 / 0.1893 / 0.1885 | — | — |
| `xgb_fa` / `lgbm_note_fa`（不含 history，終結拍分解 + test 轉移） | 0.3835 / 0.3797 | 0.1804 / 0.1848 | — | — |
| `xgb_hist2_eq` / `lgbm_hist2_eq` / `lgbm_hist_eq`（不加權） | 0.3939 / 0.3806 / 0.3759 | 0.1851 / 0.1821 / 0.1831 | — | — |
| `cat_prog_bag3` / `cat_prog` / `xgb_prog` / `lgbm_prog`（比分推移） | — | — | **0.8655** / 0.8644 / 0.8641 / 0.8586 | — |
| `cat_hist_w` / `xgb_hist_w` / `lgbm_hist_w`（比分推移 + history） | — | — | 0.8646 / 0.8638 / 0.8526 | — |

完整排行榜見 `outputs/arena_report.md`；`python eval_oof.py <方法…>` 可單獨比較幾個方法。CatBoost 的 action / point（`cat`、`cat_hist`）這一輪沒有重訓（單一方法要 1–1.5 小時，而且前兩輪都沒被選進集成），不在候選裡。

### 主要發現

改善回合 3（巢狀估計 0.4246 → 0.4309；進步小，見「各版本比較」的 bootstrap）：

1. **打散的 rally 是照固定樣式挑的**：前一個公開 rally 在前 2 分，而且下一個公開 rally 在後 2 分並且雙方各得 1 分（或它是該局最後一個）。把這個樣式當成放回的第 7 個條件，放回數 187 → 215，拿不到比分推移的 test 列 4.1% → 2.5%；winner 的單模 AUC 也因此上升（`cat_prog_bag3` 0.8617 → 0.8655）。這個樣式同時說明 test 裡比分能直接確定勝負的列比 train 模擬少，CV 的 winner AUC 偏樂觀約 0.015。
2. **終結拍更新**：用 winner 集成的勝負機率修正 action / point 集成，巢狀估計 overall +0.0039（action +0.0022、point +0.0077；含 `_fa` 的版本是 +0.0018）。不用比分推移時沒有進步（勝負機率沒有帶來新的資訊），不採用。
3. **終結拍分解讓樹模型用 test 內部轉移時不再低估終結拍**：單模 action +0.005～0.013，test 上沒有終結拍偏差。但進集成後 CV 量不到增益，而且它在 test 上比 CV 更有把握（見專節），最後沒有採用。
4. **序列模型多 1 個 seed、多 4 個 epoch**：GRU 兩個變體的 action +0.005 / +0.016，Transformer 持平。回合 3 的進步有一部分來自這裡，不是方法上的改動：把這 5 個序列模型換回上一版重跑（含 `_fa` 的方法集），巢狀估計 0.4289 → 0.4280，多訓練的貢獻約 +0.0009。
5. **0.45 的距離**（回合 2 的集成上試算，作弊情境只用來量上限）：就算 winner 完全正確（AUC = 1），overall 也只到 0.458；完全知道「下一拍是不是終結拍」是 0.471；兩者都知道是 0.499。實際可用的勝負資訊（比分推移）只能確定約 20% test 列的勝負，終結拍更新已經用掉了這份資訊的大部分。

改善回合 2：

6. **樹模型誤用 test 內部轉移，是 CV 看不到的最大問題**（見上面的專節）。修正後集成的 CV 分數不變（單一樹模型的 action 多數下降 0.006–0.014），但 test 上預測落點 0 的比例從 1.4% 回到 40%，估計 overall 約 +0.02。這是比較「CV 與 test 的預測分佈」才發現的，現在是 arena 報告的固定檢查。
7. **winner 的進步來自兩件事**：把打散的 rally 放回去（拿不到比分推移的 test 列 14.2% → 4.1%，而且多了間隔 2 的比分檢查點），以及訓練權重對齊截斷機制。巢狀估計的 winner AUC 0.8377 → 0.8617，test 預期 0.8195 → 0.8565。不含比分推移的模型也受益於對齊：XGBoost 0.666 → 0.686、LightGBM 0.649 → 0.676、GRU + history 0.639 → 0.687。原因是「發球方贏 ⇔ rally 長度是偶數」，終結率隨前綴長度怎麼變，直接決定勝負機率。
8. **對齊對 action / point 的單一模型沒有一致的效果**：樹模型加權後 action 平均約 −0.004、point 約 +0.002（有效樣本數只剩 56%）；序列模型 GRU +0.007 / +0.006、Transformer −0.006 / −0.002，都在單一模型的雜訊範圍。所以兩種版本都放進候選，讓貪婪選擇與巢狀估計決定：最後 action 用了不加權的 Transformer 與加權的 GRU、樹模型，point 以加權的比分推移樹模型為主。
9. **比分推移特徵這一輪對 point 有用了**：point 的巢狀估計 0.1945 → 0.2074，最佳的單一 point 模型都帶比分推移。上一輪它在集成層面沒有幫助；對齊之後終結拍佔 43%，「下一拍擊球者是這一分的輸家 → 這一拍很可能是終結拍（落點 0）」這個訊號的份量變大。
10. 評估抽樣改成 F0 之後，上一輪模型的 action 分數變高、point 變低（巢狀估計 0.4127 / 0.2086 → 0.4238 / 0.1945），overall 0.4177 → 0.4149：新的抽樣修正了 k = 1 比例偏高的問題，終結拍的比例也略高。

先前回合的發現（數字是當時的評估方式）：

11. **同場先前資訊（history 特徵）**：每個模型家族都受益——Transformer action +0.020、XGBoost action +0.024 / point +0.011、GRU（選手嵌入版）point +0.010。關鍵是 train 端的模擬：只用 test 實際可見的 rally 比例與前綴長度分佈，否則 train 的統計量會比 test 豐富得多。
12. **選手身分是最重要的訊號之一**：拿掉選手 ID，action F1 從 0.369 掉到 0.312。第 3 名的選手 target encoding 原寫法有標籤洩漏，修正成巢狀 OOF 後仍不如直接用原始選手 ID。再加入「目前擊球者 ID」反而變差（action −0.006）。
13. **左右鏡像增強（第 22 名）在這份資料無效**：套到樹模型 action −0.017、point −0.004，GRU 鏡像版也是最弱的 NN。
14. **seed bagging 與調整 LightGBM 大小 / 正則化都沒有穩定增益**。
15. **比分推移特徵**讓 winner AUC 從 0.69 提升到 0.86，overall 約 +0.04。

### 分數上限：為什麼離 0.5 還很遠

要在 winner AUC 0.86 下達到 overall 0.5，action + point 的 Macro-F1 合計需 0.82，目前約 0.64。以改善回合 2 的集成在 OOF 上的表現（截斷機制對齊的抽樣；回合 3 的數字相近）：

| | top-1 正確率 | top-3 正確率 | Macro-F1 |
|---|---:|---:|---:|
| action | 51% | 85% | 0.43 |
| point | 46% | 77% | 0.21 |

point 是瓶頸。落點 0（終結拍，佔 43.5%）的 F1 是 0.60：要預測的那一拍是不是 rally 的最後一拍，事前只能部分判斷。落點 7 / 8 / 9 的 F1 只有 0.20–0.28；落點 1、3 合計只佔 0.8% 的資料卻佔 point Macro-F1 的 20%，F1 只有 0.10 / 0.02。這反映的是下一拍落點本身的隨機性，不是模型容量問題。原 AI CUP 第 3 名私榜 0.392（評分資料不同，不能直接比較）。再往上追 CV 分數，主要會是在擬合雜訊。

### 已知限制

- 改善回合 1–3 都在競賽截止後才上傳驗證，不計成績（見 [SUBMISSIONS.md](SUBMISSIONS.md)）。public 與 private 分數沒有拿來調參；採用不含 `_fa` 的版本是在看過回合 3 的 Kaggle 分數之後依 CV 決定的。
- **截斷機制 F0 在 k ≥ 2 是假設**，不是資料直接證實的（見「截斷機制對齊」與「截斷假設的敏感度」）。另外假設了 test 的 rally 長度分佈與 train 相同。
- 把打散的 rally 放回去的正確率是在 train 上模擬出來的（99.8%），test 上無法直接驗證；模擬是照 test 觀察到的樣式產生打散 rally 的，不能證明樣式本身（條件 6、7 都是從「只用較少條件就能唯一定位的 rally」歸納出來的）。還有 47 個放不回去，它們拿不到比分推移、也幾乎沒有同場 history，而 CV 上不含比分推移的集成分數是在有 history 的列上算的，對這 47 列會偏樂觀（佔 test 的 2.5%）。
- **CV 的 winner AUC 偏樂觀約 0.015**：test 裡比分能直接確定勝負的列比 train 端的模擬少（見「截斷假設與比分推移組成的敏感度」）。正式的巢狀估計與 test 預期沒有扣掉這一項；換算 overall 約 −0.001（action / point 在同一個情境下略高）。
- 終結拍更新假設「落點模型的 P(落點 0) 就是 P(終結拍)」以及「非終結拍時擊球者最後輸掉的比例只跟前綴長度有關」，兩者都是近似；它有沒有用是靠巢狀估計驗證的，不是靠這個推導。放不回去的 47 列不做更新。
- 回合 2 → 回合 3 的 overall 差距，成對 bootstrap 的 95% 區間包含 0（見「各版本比較」）；進步有一部分來自序列模型多訓練了 1 個 seed、4 個 epoch（約 +0.0009）。
- 放回打散的 rally 與比分推移特徵一樣，用的是 test 檔本身的欄位（選手、比分），是否符合課堂競賽規則請自行判斷。`--preset noprog` 不用比分推移特徵，但 history 特徵仍用到放回去的 rally；要完全不用，訓練與 arena 都要加 `--no-relocate`。
- 直接分類的樹模型不用 test 內部轉移；序列模型使用。檢查過序列模型的平均 P(落點 0) 與平均最大機率在 CV 與 test 一致，但更細的偏差（例如個別類別）沒有逐一檢查。終結拍分解的樹模型（`_fa`）在 test 上比 CV 更有把握，最終集成沒有採用（見專節）。
- 採用不含 `_fa` 的版本，依據是「CV 量不到增益」加上「test 與 CV 的行為不同」，不是它的分數顯著較高：巢狀估計高 0.0020、in-sample 的 bootstrap 低 0.0007，都在雜訊內。
- XGBoost 在不同執行緒數下不能逐位元重現（LightGBM 可以），重訓後集成的成員會變、巢狀估計幾乎不變。集成配方不應解讀成方法的優劣排名。
- 序列模型併入 test 內部轉移時的對齊權重（`blend_aug_weights`）是用整個 `ap_base` 算的，不是每個 fold 各算一次（影響約 1%）。
- 「隱藏損失」是估計值：假設 test 的終結拍比例如 F0，而且只模擬了類別 0 的預測比例，沒有模擬其他類別的變化。
- 樹模型的 early stopping 用的是驗證 fold（而且驗證列沒有加權），OOF 分數會略為樂觀（各樹模型同樣受影響）。序列模型沒有 early stopping，固定跑完指定的 epoch 數，沒有這層樂觀；所以樹模型與序列模型之間的單模比較，對樹模型略為有利。
- history 特徵的 train 模擬與 test 仍有小差異：test 的攻擊率略高（0.37 vs 0.34）、落點 9 比例略高（0.26 vs 0.23）。
- 類別權重用整個 ap_base 計算（影響極小）。
- 巢狀估計只評一半比賽、而且不含版本挑選這一層；0.002 以內的差異不應視為顯著。成對 bootstrap 的區間量的是抽樣雜訊，兩邊的集成都是 in-sample 挑的。

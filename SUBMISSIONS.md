# Kaggle 繳交記錄

競賽：[Introduction to Data Science Competition — TTMATCH](https://www.kaggle.com/competitions/introduction-to-data-science-competition-ttmatch)，2026-11-16 約 23:59（台灣時間）截止。
public 榜用 67% 的 test、private 榜用另外 33%。分數 = 0.4·action Macro-F1 + 0.4·point Macro-F1 + 0.2·winner AUC。

**正式成績只有第 1 筆**。第 2–4 筆是截止後的 Late Submission，不計入排名，只用來對照 CV，沒有拿來調參。README 最後採用回合 3 不含 `_fa` 的版本，這個決定是在看過第 2、3 筆的分數之後依 CV 做的（理由與時間點見 README）。第 4 筆就是這個版本，是最後一次上傳，之後不再上傳。

| # | 上傳時間（台灣） | 計入成績 | 版本 | 檔案 md5 | CV 巢狀估計 | CV 的 test 預期 | public | private |
|---|---|---|---|---|---:|---:|---:|---:|
| 1 | 2026-10-29 06:59 | 是 | PR #2：`46715d7`（merge commit `e5e1e1b`），`python arena.py` 產生的 `outputs/submission.csv`（含比分推移） | 未記錄 | 當時沒有巢狀估計；in-sample 0.4090 | 0.4055（PR #3 補算） | 0.41744 | 0.39028 |
| 2 | 2026-11-17 00:52 | 否（Late） | 回合 3：PR #7，`main` `4961e12` | `ada43a38098d046335e3f49500393161` | 0.4290 | 0.4280 | 0.44107 | 0.40235 |
| 3 | 2026-11-17 01:01 | 否（Late） | 回合 2：PR #6，`6bdf6b1` | `5dfb3beb576fd2255ca05eadba643b8d` | 0.4246 | 0.4231 | 0.42919 | 0.40268 |
| 4 | 2026-11-17 20:25 | 否（Late） | 回合 3 不含 `_fa`：`analysis/post-deadline` `773fa1e`（PR #10），`python arena.py --preset nofa --tag nofa` 產生的 `outputs/submission_nofa.csv` | `d1b20a5fb18b6bbb91d74d143b2a1e15` | 0.4309 | 0.4298 | 0.44451 | 0.40132 |

Kaggle 的 submission ref：第 1 筆 57750635、第 2 筆 58437910、第 3 筆 58438160、第 4 筆 58468450。

以下配方的權重是貪婪集成逐步加入後換算成的最終權重（四捨五入到小數點後兩位），取自各次執行的 arena 報告或 log（`outputs/` 不進版控）。「不含比分推移的集成」用在放不回原比賽的打散 rally，也當作終結拍更新的底。

## 各筆的模型

### 第 4 筆：回合 3 不含 `_fa`（`python arena.py --preset nofa`，30 個候選方法）

實際的結果是用 `--preset final --exclude` 排除 7 個 `_fa` 方法跑的（log：`outputs/logs/r3fix_arena_nofa2.log`），`--preset nofa` 的候選清單與順序相同。

| 任務 | 主集成（權重） | 不含比分推移的集成（權重） | 最後的輸出 |
|---|---|---|---|
| action | transformer_hist2 0.39、xgb_eq 0.24、transformer_hist2_prog_al 0.20、lgbm_hist 0.17 | transformer_hist2 0.35、xgb_eq 0.22、xgb_hist2 0.18、lgbm_hist 0.15、gru_pemb_hist_al 0.10 | 以不含比分推移的集成為底做終結拍更新，λ = 1（主集成沒有用到） |
| point | lgbm_hist_prog 0.50、xgb_hist2_prog 0.30、transformer_hist2 0.20 | lgbm_hist 0.60、transformer_hist2 0.40 | 0.9 ×（以不含比分推移的集成為底，λ = 0.75）+ 0.1 × 主集成 |
| winner | cat_prog_bag3 0.70、xgb_prog 0.30 | gru_hist_al 0.40、xgb 0.40、lgbm_note 0.20 | 主集成；47 個打散 rally 用不含比分推移的集成 |

決策規則都是全域規則（沒有分深度、沒有規則修正）。放不回原比賽的 47 個打散 rally，三個任務都用不含比分推移的集成，不做終結拍更新。CV 巢狀估計：action 0.4298、point 0.2149、winner 0.8653。

### 第 2 筆：回合 3（`python arena.py --preset final`，37 個候選方法）

| 任務 | 集成成員（權重） | 終結拍更新 |
|---|---|---|
| action | transformer_hist2 0.31、xgb_hist2_eq 0.27、gru_pemb_hist_al 0.19、lgbm_hist 0.13、xgb_eq 0.10 | 以主集成為底，λ = 0.5 |
| point | xgb_hist2_prog 0.38、lgbm_hist_prog_fa 0.27、transformer_hist2 0.25、gru 0.10 | 0.6 ×（以不含比分推移的集成為底，λ = 0.75）+ 0.4 × 原集成 |
| winner | cat_prog_bag3 0.70、xgb_prog 0.30 | — |

不含比分推移的集成（log：`outputs/logs/r3_arena_final.log`）：action 與上表的主集成相同（主集成沒有選到含比分推移的方法）；point 是 lgbm_hist 0.60、transformer_hist2 0.40（也是 point 終結拍更新的底）；winner 是 gru_hist_al 0.48、xgb 0.32、lgbm_note 0.20。

放不回原比賽的 47 個打散 rally 改用不含比分推移的集成。CV 巢狀估計：action 0.4274、point 0.2124、winner 0.8653。

### 第 3 筆：回合 2（PR #6 的 `--preset final`，30 個候選方法）

沒有終結拍分解（`_fa`）與終結拍更新；放回規則是 6 個條件（放回 187 個，回合 3 是 215 個）。CV 巢狀估計：action 0.4233、point 0.2074、winner 0.8617。配方取自 `outputs/r2/arena_report.md`：

| 任務 | 主集成（權重） | 不含比分推移的集成（權重，用於放不回去的 75 列） |
|---|---|---|
| action | transformer_hist2 0.34、xgb_hist2_prog 0.32、gru_hist_al 0.20、lgbm_note 0.14 | transformer_hist2 0.41、lgbm_note 0.17、lgbm_hist2 0.16、xgb_hist2_eq 0.10、gru_mirror 0.09、xgb_hist2 0.06 |
| point | xgb_hist2_prog 0.54、lgbm_hist_prog 0.36、lgbm_hist_eq 0.10 | transformer_hist2_al 0.24、transformer_hist2 0.22、lgbm_hist_eq 0.20、lgbm_hist 0.20、xgb_hist2 0.13 |
| winner | cat_prog_bag3 0.90、gru_hist_al 0.10 | gru_hist_al 0.40、xgb 0.40、lgbm_note 0.20 |

決策規則都是全域規則（沒有分深度、沒有規則修正）。

### 第 1 筆：正式繳交（PR #2，`e5e1e1b`）

| 任務 | 集成成員（權重） | 當時的 CV（in-sample） |
|---|---|---:|
| action | transformer 0.38、lgbm_note 0.26、gru_pemb 0.20、xgb 0.16 | 0.3971 |
| point | lgbm 0.48、gru_mirror 0.32、gru_pemb 0.20 | 0.2023 |
| winner | cat_prog 0.45、xgb 0.45、transformer 0.10 | 0.8462 |
| winner（test 異常 match 340–354 的列） | 不含比分推移的集成：cat 0.384、xgb 0.256、lgbm 0.20、transformer 0.16 | 0.6959 |
| **overall** | | **0.4090** |

「方法競技場」的第一版：樹模型、GRU／Transformer、比分推移特徵、arena 貪婪集成；分深度決策與規則修正未採用。還沒有同場歷史特徵、截斷機制對齊、打散 rally 放回、終結拍相關的做法。test 異常 match（340–354）的 winner 已經改用不含比分推移的集成（`e5e1e1b` 的 `arena.py` 第 204–213 行；配方見上表最後一列）；PR #3 補上的是 0.4055 這個 test 預期的估計，不是這個替換。當時的 CV 是舊的評估方式（沒有對齊 test 的截斷機制），與第 2、3 筆的 CV 數字不能直接比較。

版本是這樣定位的（當時沒有記錄檔案雜湊）：
- 上傳的檔名是 `submission.csv`，上傳時間 2026-10-28 22:59 UTC。
- 在那之前最後一次產生 `outputs/submission.csv` 的是 2026-10-28 19:26 UTC 的 `python arena.py`（PR #2 的程式碼）；PR #2 在 19:54 UTC 合併。
- PR #3 的修改從 22:45 UTC 才開始，到上傳時還沒有重新產生提交檔。
- 以上是從開發當時的指令記錄推定的；如果當時上傳的是另一份檔案（例如 `submission_noprog.csv` 改名），這個推定就不成立。

## 從這四筆能看到的事

- 回合 2、回合 3 在 private 上比正式繳交高約 0.012（0.3903 → 0.4027／0.4024）。
- 回合 2 → 回合 3 在 public 上多 0.012，但在 private 上沒有差別（0.40268 對 0.40235）。這與 README「各版本比較」的成對 bootstrap 結論一致：回合 3 的進步在 95% 區間內包含 0。
- 第 4 筆（不含 `_fa`）public 是四筆最高（0.44451），private 則比第 2、3 筆低約 0.001（0.40132）。三個截止後版本的 private 都在 0.401–0.403，差距遠小於 private 的抽樣雜訊，與 README 的成對 bootstrap（含不含 `_fa` 的差在 95% 區間內包含 0）一致；這筆分數沒有用來改任何東西。
- CV 的 test 預期（0.423、0.428、0.430）落在 public 與 private 之間；private 只有約 609 列，Macro-F1 的抽樣雜訊不小，單一筆的差異不宜過度解讀。

# Kaggle 繳交記錄

競賽：[Introduction to Data Science Competition — TTMATCH](https://www.kaggle.com/competitions/introduction-to-data-science-competition-ttmatch)，2026-11-16 約 23:59（台灣時間）截止。
public 榜用 67% 的 test、private 榜用另外 33%。分數 = 0.4·action Macro-F1 + 0.4·point Macro-F1 + 0.2·winner AUC。

**正式成績只有第 1 筆**。第 2、3 筆是截止後的 Late Submission，不計入排名，只用來對照，沒有拿來挑模型或調參。回合 3 不含 `_fa` 的版本（README 最後採用的）預定再上傳一次，之後不再上傳。

| # | 上傳時間（台灣） | 計入成績 | 版本 | 檔案 md5 | CV 巢狀估計 | CV 的 test 預期 | public | private |
|---|---|---|---|---|---:|---:|---:|---:|
| 1 | 2026-10-29 06:59 | 是 | PR #2：`46715d7`（merge commit `e5e1e1b`），`python arena.py` 產生的 `outputs/submission.csv`（含比分推移） | 未記錄 | 當時沒有巢狀估計；in-sample 0.4090 | 0.4055（PR #3 補算） | 0.41744 | 0.39028 |
| 2 | 2026-11-17 00:52 | 否（Late） | 回合 3：PR #7，`main` `4961e12` | `ada43a38098d046335e3f49500393161` | 0.4290 | 0.4280 | 0.44107 | 0.40235 |
| 3 | 2026-11-17 01:01 | 否（Late） | 回合 2：PR #6，`6bdf6b1` | `5dfb3beb576fd2255ca05eadba643b8d` | 0.4246 | 0.4231 | 0.42919 | 0.40268 |

Kaggle 的 submission ref：第 1 筆 57750635、第 2 筆 58437910、第 3 筆 58438160。

## 各筆的模型

### 第 2 筆：回合 3（`python arena.py --preset final`，37 個候選方法）

| 任務 | 集成成員（權重） | 終結拍更新 |
|---|---|---|
| action | transformer_hist2 0.31、xgb_hist2_eq 0.27、gru_pemb_hist_al 0.19、lgbm_hist 0.13、xgb_eq 0.10 | 以主集成為底，λ = 0.5 |
| point | xgb_hist2_prog 0.38、lgbm_hist_prog_fa 0.27、transformer_hist2 0.25、gru 0.10 | 0.6 ×（以不含比分推移的集成為底，λ = 0.75）+ 0.4 × 原集成 |
| winner | cat_prog_bag3 0.70、xgb_prog 0.30 | — |

放不回原比賽的 47 個打散 rally 改用不含比分推移的集成。CV 巢狀估計：action 0.4274、point 0.2124、winner 0.8653。

### 第 3 筆：回合 2（PR #6 的 `--preset final`，30 個候選方法）

沒有終結拍分解（`_fa`）與終結拍更新；放回規則是 6 個條件（放回 187 個，回合 3 是 215 個）。CV 巢狀估計：action 0.4233、point 0.2074、winner 0.8617。配方見當時的 `outputs/r2/arena_report.md`（`outputs/` 不進版控）。

### 第 1 筆：正式繳交（PR #2，`e5e1e1b`）

| 任務 | 集成成員（權重） | 當時的 CV（in-sample） |
|---|---|---:|
| action | transformer 0.38、lgbm_note 0.26、gru_pemb 0.20、xgb 0.16 | 0.3971 |
| point | lgbm 0.48、gru_mirror 0.32、gru_pemb 0.20 | 0.2023 |
| winner | cat_prog 0.45、xgb 0.45、transformer 0.10 | 0.8462 |
| **overall** | | **0.4090** |

「方法競技場」的第一版：樹模型、GRU／Transformer、比分推移特徵、arena 貪婪集成；分深度決策與規則修正未採用。還沒有同場歷史特徵、截斷機制對齊、打散 rally 放回、終結拍相關的做法，test 異常 match（340–354）也還沒有改用不含比分推移的集成（那是 PR #3 才加的）。當時的 CV 是舊的評估方式（沒有對齊 test 的截斷機制），與第 2、3 筆的 CV 數字不能直接比較。

版本是這樣定位的（當時沒有記錄檔案雜湊）：
- 上傳的檔名是 `submission.csv`，上傳時間 2026-10-28 22:59 UTC。
- 在那之前最後一次產生 `outputs/submission.csv` 的是 2026-10-28 19:26 UTC 的 `python arena.py`（PR #2 的程式碼）；PR #2 在 19:54 UTC 合併。
- PR #3 的修改從 22:45 UTC 才開始，到上傳時還沒有重新產生提交檔。
- 以上是從開發當時的指令記錄推定的；如果當時上傳的是另一份檔案（例如 `submission_noprog.csv` 改名），這個推定就不成立。

## 從這三筆能看到的事

- 回合 2、回合 3 在 private 上比正式繳交高約 0.012（0.3903 → 0.4027／0.4024）。
- 回合 2 → 回合 3 在 public 上多 0.012，但在 private 上沒有差別（0.40268 對 0.40235）。這與 README「各版本比較」的成對 bootstrap 結論一致：回合 3 的進步在 95% 區間內包含 0。
- CV 的 test 預期（0.423、0.428）落在 public 與 private 之間；private 只有約 609 列，Macro-F1 的抽樣雜訊不小，單一筆的差異不宜過度解讀。

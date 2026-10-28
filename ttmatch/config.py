"""全域設定：路徑、類別空間、評分權重、CV 設定。

改寫自 0000workjason/aicup2026-tabletennis-prediction（AI CUP 2026 私榜第 3 名）solution/config.py。
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_SLUG = "introduction-to-data-science-competition-ttmatch"


def data_dir() -> Path:
    """資料位置優先序：環境變數 TTMATCH_DATA_DIR → repo 內 data/ → kagglehub 下載快取。"""
    env = os.environ.get("TTMATCH_DATA_DIR")
    if env:
        return Path(env)
    local = ROOT / "data"
    if (local / "train.csv").exists():
        return local
    import kagglehub
    return Path(kagglehub.competition_download(KAGGLE_SLUG))


OUT_DIR = Path(os.environ.get("TTMATCH_OUT_DIR", ROOT / "outputs"))
OOF_DIR = OUT_DIR / "oof"

# 下一拍不可能是發球（15–18 只出現在第 1 拍，且第 1 拍一定已觀測到）
ACTION_CLASSES = list(range(0, 15))
POINT_CLASSES = list(range(0, 10))
N_ACTION = len(ACTION_CLASSES)
N_POINT = len(POINT_CLASSES)

# Score = 0.4*MacroF1(action) + 0.4*MacroF1(point) + 0.2*AUC(serverGetPoint)
W_ACTION, W_POINT, W_WINNER = 0.4, 0.4, 0.2

N_FOLDS = 5
SEED = 42

ATTACK_IDS = list(range(1, 8))
CONTROL_IDS = list(range(8, 12))
DEFENSIVE_IDS = list(range(12, 15))
SERVE_IDS = list(range(15, 19))


def ensure_dirs() -> None:
    OOF_DIR.mkdir(parents=True, exist_ok=True)

"""Datalume 品牌资产定位。"""

from __future__ import annotations

from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parent
LOGO_PATH = PACKAGE_ROOT / "assets" / "datalume_logo.png"


def ensure_logo_asset() -> Path | None:
    """返回仓库内的 Datalume Logo；资产缺失时安全降级。"""

    return LOGO_PATH if LOGO_PATH.exists() else None

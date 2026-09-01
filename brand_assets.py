"""Datalume 品牌资产定位与首次运行复制。"""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile


PROJECT_ROOT = Path(__file__).resolve().parent
LOGO_PATH = PROJECT_ROOT / "assets" / "datalume_logo.png"
GENERATED_LOGO_SOURCE = Path(
    "/Users/wyy/.codex/generated_images/01a0341b-4e4a-7da3-89fd-3a2aec48b90b/"
    "exec-7e4187fa-655b-4c94-8c3c-d3b38f8329fe.png"
)


def ensure_logo_asset() -> Path | None:
    """将本次参考图提取的 Logo 固化到项目 assets；缺失时安全降级。"""

    if not LOGO_PATH.exists() and GENERATED_LOGO_SOURCE.exists():
        LOGO_PATH.parent.mkdir(parents=True, exist_ok=True)
        copyfile(GENERATED_LOGO_SOURCE, LOGO_PATH)
    return LOGO_PATH if LOGO_PATH.exists() else None

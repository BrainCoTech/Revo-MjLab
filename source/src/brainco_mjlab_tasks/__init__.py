"""BrainCo mjlab task package."""

from pathlib import Path

BRAINCO_MJLAB_TASKS_SOURCE_PATH: Path = Path(__file__).parent

from brainco_mjlab_tasks.dexsuite.config import brainco as _dexsuite_brainco  # noqa: F401,E402
from brainco_mjlab_tasks.inhand.config import revo3_right as _inhand_revo3_right  # noqa: F401,E402

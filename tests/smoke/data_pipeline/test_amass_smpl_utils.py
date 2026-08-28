from pathlib import Path

import numpy as np

from data_converter.amass_smpl_utils import load_motion_source


def test_load_motion_source_accepts_soma_frame_rate_alias(tmp_path: Path) -> None:
    """SOMA Stage-II 的 FPS 别名应与 AMASS 标准字段保持相同语义。"""

    motion_path = tmp_path / "SOMA" / "motion_stageii.npz"
    motion_path.parent.mkdir(parents=True)
    np.savez(
        motion_path,
        poses=np.zeros((5, 66), dtype=np.float32),
        trans=np.zeros((5, 3), dtype=np.float32),
        mocap_frame_rate=np.asarray(30.0, dtype=np.float32),
        gender=np.asarray("neutral"),
    )

    source = load_motion_source(
        path=motion_path,
        amass_dir=tmp_path,
        target_fps=30.0,
    )

    assert source.relative_path == Path("SOMA/motion_stageii.npz")
    assert source.source_fps == 30.0
    assert source.poses.shape == (5, 66)
    assert source.trans.shape == (5, 3)

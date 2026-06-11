from __future__ import annotations

import sys
import warnings

from data_converter.amass_to_realtime_pose import main as _main
from data_loaders.sensor_masking import REALTIME_POSE_V2_CONTACT_SCHEMA_NAME


def main(argv: list[str] | None = None) -> dict[str, int]:
    """legacy v2 转换入口，仅用于显式生成旧 realtime_pose_v2_contact 数据。"""

    warnings.warn(
        "amass_to_realtime_pose_v2 is a legacy wrapper for realtime_pose_v2_contact; "
        "the current main pipeline is realtime_pose_body_fbx_local_root_y0_v1.",
        stacklevel=2,
    )
    args = list(sys.argv[1:] if argv is None else argv)
    if "--schema" not in args:
        args.extend(["--schema", REALTIME_POSE_V2_CONTACT_SCHEMA_NAME])
    return _main(args)


if __name__ == "__main__":
    main()

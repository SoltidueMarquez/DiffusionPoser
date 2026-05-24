from __future__ import annotations

import sys

from data_converter.amass_to_realtime_pose import main as _main
from data_loaders.sensor_masking import REALTIME_POSE_V2_CONTACT_SCHEMA_NAME


def main(argv: list[str] | None = None) -> dict[str, int]:
    """v2 转换入口，默认生成带 root motion/contact 的当前推荐 schema。"""

    args = list(sys.argv[1:] if argv is None else argv)
    if "--schema" not in args:
        args.extend(["--schema", REALTIME_POSE_V2_CONTACT_SCHEMA_NAME])
    return _main(args)


if __name__ == "__main__":
    main()

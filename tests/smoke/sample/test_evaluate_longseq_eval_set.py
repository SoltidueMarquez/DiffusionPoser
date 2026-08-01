from __future__ import annotations

import numpy as np

import sample.evaluate_longseq_eval_set as longseq


def test_longseq_defaults_to_five_steps_and_latency_summary_excludes_warmup():
    args = longseq.build_arg_parser().parse_args(["--model_path", "model.pt"])
    assert args.inference_steps == 5
    summary = longseq.summarize_latency(np.asarray([100.0, 10.0, 20.0]), warmup_frames=1)
    assert summary["frames"] == 2
    assert summary["mean_ms"] == 15.0
    assert summary["p95_ms"] == 19.5

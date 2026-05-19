from __future__ import annotations

import unittest

import torch

from data_loaders.sensor_masking import MODEL_INPUT_DIM
from sample.reconstruct_stream import build_previous_frame_conditioned_sequence, build_stream_window


def make_reference(seq_len: int) -> torch.Tensor:
    values = torch.arange(MODEL_INPUT_DIM * seq_len, dtype=torch.float32)
    return values.reshape(MODEL_INPUT_DIM, seq_len)


class ReconstructStreamWindowTest(unittest.TestCase):
    def test_current_unknown_features_are_seeded_from_previous_reconstruction(self):
        reference = make_reference(seq_len=4)
        reconstructed = reference + 10_000.0
        task_inpaint_mask = torch.zeros_like(reference, dtype=torch.bool)
        task_inpaint_mask[[0, 216, 272], 2] = True
        valid_frame_mask = torch.ones(4, dtype=torch.bool)

        conditioned, inpaint_mask, window_valid, current_dest = build_stream_window(
            reference=reference,
            reconstructed=reconstructed,
            task_inpaint_mask=task_inpaint_mask,
            valid_frame_mask=valid_frame_mask,
            frame_index=2,
            seq_len=3,
        )

        self.assertEqual(current_dest, 2)
        torch.testing.assert_close(conditioned[0, :, 0], reconstructed[:, 0])
        torch.testing.assert_close(conditioned[0, :, 1], reconstructed[:, 1])

        expected_current = reference[:, 2].clone()
        frame_mask = task_inpaint_mask[:, 2]
        expected_current[frame_mask] = reconstructed[:, 1][frame_mask]
        torch.testing.assert_close(conditioned[0, :, current_dest], expected_current)
        self.assertTrue(torch.equal(inpaint_mask[0, :, current_dest], frame_mask))
        self.assertFalse(inpaint_mask[0, :, :current_dest].any())
        self.assertTrue(torch.equal(window_valid, torch.ones((1, 3), dtype=torch.bool)))

    def test_first_frame_unknown_features_use_zero_cold_start(self):
        reference = make_reference(seq_len=3)
        reconstructed = reference + 10_000.0
        task_inpaint_mask = torch.zeros_like(reference, dtype=torch.bool)
        task_inpaint_mask[[0, 216], 0] = True
        valid_frame_mask = torch.ones(3, dtype=torch.bool)

        conditioned, inpaint_mask, window_valid, current_dest = build_stream_window(
            reference=reference,
            reconstructed=reconstructed,
            task_inpaint_mask=task_inpaint_mask,
            valid_frame_mask=valid_frame_mask,
            frame_index=0,
            seq_len=3,
        )

        expected_current = reference[:, 0].clone()
        frame_mask = task_inpaint_mask[:, 0]
        expected_current[frame_mask] = 0.0
        self.assertEqual(current_dest, 2)
        torch.testing.assert_close(conditioned[0, :, current_dest], expected_current)
        self.assertTrue(torch.equal(inpaint_mask[0, :, current_dest], frame_mask))
        self.assertTrue(torch.equal(window_valid, torch.tensor([[False, False, True]])))

    def test_saved_conditioned_sequence_matches_streaming_previous_frame_seed(self):
        reference = make_reference(seq_len=4)
        reconstructed = reference + 10_000.0
        task_inpaint_mask = torch.zeros_like(reference, dtype=torch.bool)
        task_inpaint_mask[[0, 216], 1] = True
        task_inpaint_mask[[10, 272], 2] = True
        valid_frame_mask = torch.tensor([True, True, True, False])

        conditioned = build_previous_frame_conditioned_sequence(
            reference=reference,
            reconstructed=reconstructed,
            task_inpaint_mask=task_inpaint_mask,
            valid_frame_mask=valid_frame_mask,
        )

        expected = reference.clone()
        # 保存/可视化条件只反映当前窗口最后一帧的推理输入。
        expected[task_inpaint_mask[:, 2], 2] = reconstructed[task_inpaint_mask[:, 2], 1]
        expected[:, 3] = 0.0
        torch.testing.assert_close(conditioned, expected)


if __name__ == "__main__":
    unittest.main()

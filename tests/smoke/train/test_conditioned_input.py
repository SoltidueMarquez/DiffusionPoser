from __future__ import annotations

import unittest

import torch

from data_loaders.sensor_masking import MODEL_INPUT_DIM, X277_FEATURE_DIM
from train.training_loop import TrainLoop


class TrainConditionedInputTest(unittest.TestCase):
    def test_mask_manager_uses_conditioned_x_for_inpainting_conditions(self):
        loop = object.__new__(TrainLoop)
        sample = torch.ones((1, MODEL_INPUT_DIM, 11), dtype=torch.float32)
        conditioned_x = torch.zeros_like(sample)
        inpaint_mask = torch.zeros_like(sample, dtype=torch.bool)
        inpaint_mask[:, 0, 10] = True
        inpaint_mask[:, X277_FEATURE_DIM:, 10] = True
        batch = {
            "x": sample,
            "conditioned_x": conditioned_x,
            "valid_frame_mask": torch.ones((1, 11), dtype=torch.bool),
            "inpaint_mask": inpaint_mask,
        }

        model_kwargs = loop.mask_manager(batch, sample)

        self.assertTrue(torch.equal(model_kwargs["y"]["inpainted_motion"], conditioned_x))
        self.assertFalse(model_kwargs["inpaint_cond"][:, X277_FEATURE_DIM:, :].any())


if __name__ == "__main__":
    unittest.main()

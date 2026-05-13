# Project Layout and Style Anchors

Use this reference as a map of code organization and naming conventions for the DiffusionPoser reproduction.

## Paper files in this workspace

- `2024_DiffusionPoser_Real_time_Human_Motion_Reconstruction_from_Arbitrary_Sparse_Sensors.pdf`
- `2023_MDM_Human_Motion_Diffusion_Model.pdf`
- `2023_EDGE_Editable_Dance_Generation_From_Music.pdf`

## StableMotion anchors

- `diffusion/gaussian_diffusion.py`: diffusion schedules, training losses, masking, and sampling methods.
- `diffusion/respace.py`: timestep respacing and spaced diffusion wrapper.
- `model/stablemotion.py`: DiT-style backbone, AdaLN conditioning, rotary embeddings, and inpaint conditioning.
- `model/cfg_sampler.py`: classifier-free guidance wrapper style.
- `utils/model_util.py`: model and diffusion factory pattern.
- `utils/parser_util.py`: grouped `argparse` options.
- `train/train_stablemotion_smpl_glob.py`: thin training entrypoint.
- `train/training_loop_smpl.py`: training loop, EMA, checkpointing, and loss application.
- `sample/fix_globsmpl.py`: inference entrypoint and result writing pattern.
- `data_loaders/get_data.py`: dataset loader factory.
- `data_loaders/globsmpl_dataset.py`: motion tensor dataset conventions.
- `data_loaders/corrupting_globsmpl_dataset.py`: sparse or corrupted motion data preparation.

## Preferred file ownership

- `model/`: neural network modules and conditioning blocks.
- `diffusion/`: beta schedules, diffusion objectives, sampler loops, respacing.
- `data_loaders/`: dataset classes, preprocessing wrappers, collate logic.
- `train/`: training entrypoints and loop orchestration.
- `sample/`: reconstruction and inference entrypoints.
- `eval/`: paper metrics, benchmark scripts, result summaries.
- `utils/`: configuration, seed control, devices, checkpoints, normalization, logging helpers.

## Naming cues

- Use paper-faithful names for important concepts, then clarify them with comments when needed.
- Name tensors by their role, not by temporary implementation details.
- Keep shape-changing variables readable: `motion_bct`, `motion_btd`, `sensor_pos`, `sensor_rot`, `valid_frame_mask`.
- Avoid single-letter names except for local math variables in compact formulas.

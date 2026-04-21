from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import gpu_utils
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.models.cuda_timing import CudaTimer, time_function

BasePolicy: TypeAlias = _base_policy.BasePolicy

class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        denoise_steps_range: tuple[int,int] = None
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            denoise_steps_range: (min, max) The minimun and maximun number of steps the model will execute in denoise stage.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup - don't use module_jit here since sample_actions handles JIT internally
            self._sample_actions = model.sample_actions
            self._rng = rng or jax.random.key(0)

        if denoise_steps_range is not None:
            if sample_kwargs and "num_steps" in sample_kwargs:
                logging.warning(f"'denoise_steps_range' is set to {denoise_steps_range}, the keyword argument 'num_steps' will not be adopted.")
            self._denoise_steps_range = denoise_steps_range
            if self._is_pytorch_model:
                self._windowed_trajectory_distance = gpu_utils.GPUWindowedTrajectoryDistanceTorch(
                    window_size=5, device=pytorch_device
                )
                self._max_steps = torch.tensor(denoise_steps_range[1], device=pytorch_device)
                self._min_steps = torch.tensor(denoise_steps_range[0], device=pytorch_device)
            else:
                self._windowed_trajectory_distance = gpu_utils.GPUWindowedTrajectoryDistance(
                    window_size=5, device="cuda" if jax.default_backend() == "cuda" else "cpu"
                )
                self._max_steps = jnp.int32(denoise_steps_range[1])
                self._min_steps = jnp.int32(denoise_steps_range[0])
            self._replan_steps = 5
            self._focused_action_dim = 3  # we calculate the trajectory distance based on the end-effector velocity, which is represented by the first three dimensions of the action

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        timer = CudaTimer()
        timer.start()
        if hasattr(self, "_denoise_steps_range"):
            # Calculate the number of denoising steps based on the trajectory distance of the last action chunk. The smaller the distance, the more steps (up to max_steps).
            # Keep computations on GPU as long as possible, only convert to Python int at the end
            td = self._windowed_trajectory_distance.get_last()
            max_td = self._windowed_trajectory_distance.get_max()
            min_td = self._windowed_trajectory_distance.get_min()
            if self._is_pytorch_model:
                num_steps = torch.ceil(self._max_steps + (self._min_steps - self._max_steps) * (td - min_td) / (max_td - min_td + 1e-5))
                sample_kwargs["num_steps"] = int(num_steps)
            else:
                num_steps = jnp.ceil(self._max_steps + (self._min_steps - self._max_steps) * (td - min_td) / (max_td - min_td + 1e-5))
                sample_kwargs["num_steps"] = int(num_steps)
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        if hasattr(self, "_denoise_steps_range"):
            # calculate the maximun velocity of the end effector in this action chunk
            discarded_actions = actions[0, self._replan_steps:, :self._focused_action_dim]
            if self._is_pytorch_model:
                trajectory_distance = torch.diff(discarded_actions, dim=0).norm(dim=1).sum().item()
            else:
                trajectory_distance = jnp.linalg.norm(jnp.diff(discarded_actions, axis=0), axis=1).sum()
            self._windowed_trajectory_distance.add(trajectory_distance)
        inference_time = timer.stop(actions)
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": inference_time,
        }
        if hasattr(self, "_denoise_steps_range"):
            outputs["num_steps"] = sample_kwargs["num_steps"]

        if hasattr(self._model, "collect_v_t_angles") and self._model.collect_v_t_angles:
            outputs["max_v_t_angle"] = self._model.max_v_t_angle
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

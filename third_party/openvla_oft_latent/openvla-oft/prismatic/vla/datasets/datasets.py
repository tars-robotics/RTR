"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights

from omegaconf import DictConfig, OmegaConf
import hydra
from rtr_async_sys.models.reactive_diffusion_policy.model.vae.model import VAE
@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        # print(f"before action tokenizer, future_actions is {future_actions}")
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        # print(f"after action tokenizer, future_actions_string is {future_actions_string}")

        # future_actions_ids = self.base_tokenizer(future_actions_string, add_special_tokens=True).input_ids
        # print(f"future_actions_ids is {future_actions_ids}")
        # decoded_future_actions = self.action_tokenizer.decode_token_ids_to_actions(np.array(future_actions_ids))
        # print(f"decoded_future_actions is {decoded_future_actions}")
        # import pdb;pdb.set_trace()

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        only_for_vae:bool=False,
        two_stage_with_vla:bool=False,
        t_downsample_ratio:int = 4,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        if two_stage_with_vla:
            future_action_window_size = NUM_ACTIONS_CHUNK*t_downsample_ratio - 1
        else:
            future_action_window_size = NUM_ACTIONS_CHUNK-1
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=4#16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on
        self.only_for_vae = only_for_vae
        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            if self.only_for_vae:
                # import pdb; pdb.set_trace()
                yield rlds_batch
            else:
                yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class RLDSDataset_with_rdp_vae(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        use_vae:bool = True,
        t_downsample_ratio:int = 4,
        horizon:int = 48,
        n_embed:int = 10,
        vae_config_path:str = 'configs/rdp_vae/rdp_vae.yaml',
        ckpt_path:str = 'data/ckpts/vase_sponge_test1_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/latest.ckpt',
        vae_latent_dataset_statistics_path:str = 'data/ckpts/vase_sponge_test1_60hz/ckpts_abs/rdp_vae/n_embed_10/horizon48/dataset_stats.json'
    ) -> None:
        self.use_vae = use_vae
        self.t_downsample_ratio = t_downsample_ratio
        # self.use_vae = False
        # self.t_downsample_ratio = 1
        self.horizon = horizon
        self.n_embed = n_embed# must match action_dim
        self.normalize_latent = True
        vae_config_path = vae_config_path
        ckpt_path = ckpt_path
        if not self.use_vae:
            raise NotImplementedError("must use_vae for RLDSDataset_with_vae")
            # assert self.t_downsample_ratio ==1, "t_downsample_ratio must be 1 if not use_vae"
        else:
            if isinstance(vae_config_path, str):
                model = OmegaConf.load(vae_config_path)
            if isinstance(model, DictConfig):
                model['horizon'] = self.horizon
                model['n_embed'] = self.n_embed
                # model['normalize_latent'] = self.normalize_latent
                model = hydra.utils.instantiate(model)
            self.vae:VAE = model
            if ckpt_path != None:
                payload = torch.load(ckpt_path, weights_only=False, map_location="cpu")
                # Load the model weights
                self.vae.load_state_dict(payload['state_dicts']['model'])
            # for normalize action for naction statistics collection in data_utils
            print(f"vae vae_dataset_statistics_path is {vae_latent_dataset_statistics_path}")
            self.vae._load_latent_dataset_statistics(path=vae_latent_dataset_statistics_path)
            # Move vae to the correct device and set to evaluation mode, just on cpu
            self.vae.to('cpu')
            self.vae.eval()
            print(f"self.vae is {self.vae}")

        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK*self.t_downsample_ratio - 1,      # For action chunking # return NUM_ACTIONS_CHUNK*self.t_downsample_ratio-1+1 for action_chunk. then vae downsample to NUM_ACTIONS_CHUNK
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=4#16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

        # print(f"self.dataset_statistics is {self.dataset_statistics}")
        # self.vae._load_dataset_statistics(dataset_statistics=self.dataset_statistics)

    def make_dataset(self, rlds_config,vae=None):
        return make_interleaved_dataset(**rlds_config,vae=None,use_rdp_vae=True)# not normalize action

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            if self.use_vae:
                # import pdb;pdb.set_trace()
                action = rlds_batch["action"].copy()
                action[:, -1] = 0.47058823704719543 # to match with rdp_vae
                rlds_batch["action"] = action
                # rlds_batch['action'][:,-1] = 0.47058823704719543 # to match with rdp_vae 
                original_actions = rlds_batch['action'].copy() # have not been normalized
                actions = torch.tensor(rlds_batch['action'][None,:])#[1,T,A]
                actions = self.vae.encode_to_latent(actions)
                if self.normalize_latent:
                    actions = self.vae.normalize_from_dataset(actions, is_latent=True)
                actions = actions.detach().numpy()[0]
                rlds_batch['action'] = actions
                # Do the latent encode before batch_transform — not to bypass normalization, but to avoid changing the shape of input_ids.
                data = self.batch_transform(rlds_batch)# data['actions'].shape is (NUM_ACTIONS_CHUNK, action_dim)
                
                data['original_actions'] = original_actions
                # import pdb; pdb.set_trace()
                yield data
            else:
                data = self.batch_transform(rlds_batch)# data['actions'].shape is (NUM_ACTIONS_CHUNK, action_dim)
                yield data

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")




class RLDSDataset_with_vae(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        use_vae:bool = True,
        t_downsample_ratio:int = 4,
        horizon:int = 48,
        n_embed:int = 10,
        vae_config_path:str = 'src/rtr_async_sys/configs/user/model_wrapper/model/rdp/openvla_oft_vae.yaml',
        ckpt_path:str = 'data/ckpts/vase_sponge_test1_60hz/ckpts_abs/openvla_oft/vae/horizon48_compress4_n_embed_10/latest.ckpt',
        normalize_latent:bool = False,
        vae_dataset_statistics_path:str = 'data/ckpts/vase_sponge_test1_60hz/ckpts_abs/openvla_oft/vae/horizon48_compress4_n_embed_10/dataset_statistics.json'
    ) -> None:
        self.use_vae = use_vae
        self.t_downsample_ratio = t_downsample_ratio
        # self.use_vae = False
        # self.t_downsample_ratio = 1
        self.horizon = horizon
        self.n_embed = n_embed# must match action_dim
        self.normalize_latent = normalize_latent
        vae_config_path = vae_config_path
        ckpt_path = ckpt_path
        if not self.use_vae:
            raise NotImplementedError("must use_vae for RLDSDataset_with_vae")
            # assert self.t_downsample_ratio ==1, "t_downsample_ratio must be 1 if not use_vae"
        else:
            if isinstance(vae_config_path, str):
                model = OmegaConf.load(vae_config_path)
            if isinstance(model, DictConfig):
                model['horizon'] = self.horizon
                model['n_embed'] = self.n_embed
                model['normalize_latent'] = self.normalize_latent
                model = hydra.utils.instantiate(model)
            self.vae = model
            if ckpt_path != None:
                payload = torch.load(ckpt_path)
                # Load the model weights
                self.vae.load_state_dict(payload['state_dicts']['model'])
            # for normalize action for naction statistics collection in data_utils
            print(f"vae vae_dataset_statistics_path is {vae_dataset_statistics_path}")
            self.vae._load_dataset_statistics(path=vae_dataset_statistics_path)
            # Move vae to the correct device and set to evaluation mode, just on cpu
            self.vae.to('cpu')
            self.vae.eval()
            print(f"self.vae is {self.vae}")

        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK*self.t_downsample_ratio - 1,      # For action chunking # return NUM_ACTIONS_CHUNK*self.t_downsample_ratio-1+1 for action_chunk. then vae downsample to NUM_ACTIONS_CHUNK
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=4#16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config,vae=self.vae)

        print(f"self.dataset_statistics is {self.dataset_statistics}")
        self.vae._load_dataset_statistics(dataset_statistics=self.dataset_statistics)

    def make_dataset(self, rlds_config,vae=None):
        return make_interleaved_dataset(**rlds_config,vae=vae)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            if self.use_vae:
                # import pdb; pdb.set_trace()

                original_actions = rlds_batch['action'].copy()
                actions = torch.tensor(rlds_batch['action'][None,:])#[1,T,A]
                actions = self.vae.encode_to_latent(actions)
                if self.normalize_latent:
                    actions = self.vae.normalize_from_dataset(actions, is_latent=True)
                actions = actions.detach().numpy()[0]
                rlds_batch['action'] = actions
                # Do the latent encode before batch_transform — not to bypass normalization, but to avoid changing the shape of input_ids.
                data = self.batch_transform(rlds_batch)# data['actions'].shape is (NUM_ACTIONS_CHUNK, action_dim)
                
                data['original_actions'] = original_actions
                # import pdb; pdb.set_trace()
                yield data
            else:
                data = self.batch_transform(rlds_batch)# data['actions'].shape is (NUM_ACTIONS_CHUNK, action_dim)
                # print(data['pixel_values'].shape)
                # print(data['input_ids'].shape)
                # print(data['labels'].shape)
                # print(data['actions'].shape)
                # print(data['proprio'].shape)
                yield data

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
    

if __name__ == '__main__':
    rlds_dataset = RLDSDataset(
        data_root_dir=None,
        data_mix=None,
        batch_transform=None,
        resize_resolution=None
    )

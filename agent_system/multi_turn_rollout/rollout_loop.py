# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager, suppress
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.trajectory_grpo import (
    require_filter_target_reached,
    take_complete_uid_groups,
)
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


def _record_session_cleanup_error(primary_error, cleanup_error):
    with suppress(BaseException):
        primary_error.rollout_session_cleanup_error = cleanup_error

    with suppress(BaseException):
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note(f"rollout session cleanup also failed: {cleanup_error!r}")

    with suppress(BaseException):
        logger.error(
            "Rollout session cleanup failed while preserving the primary rollout error",
            exc_info=(
                type(cleanup_error),
                cleanup_error,
                cleanup_error.__traceback__,
            ),
        )


@contextmanager
def _rollout_session(actor_rollout_wg):
    begin = getattr(actor_rollout_wg, "begin_rollout_session", None)
    end = getattr(actor_rollout_wg, "end_rollout_session", None)
    if not callable(begin) or not callable(end):
        yield
        return

    primary_error = None
    try:
        begin()
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            end()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            _record_session_cleanup_error(primary_error, cleanup_error)


def _select_observation_rows(obs: Dict[str, Any], indices: np.ndarray) -> Dict[str, Any]:
    selected = {}
    for key, value in obs.items():
        if value is None:
            selected[key] = None
        elif isinstance(value, torch.Tensor):
            selected[key] = value[torch.as_tensor(indices, device=value.device)]
        elif isinstance(value, np.ndarray):
            selected[key] = value[indices]
        else:
            selected[key] = [value[index] for index in indices]
    return selected


def _scatter_observation_rows(
    obs: Dict[str, Any],
    selected_obs: Dict[str, Any],
    indices: np.ndarray,
) -> Dict[str, Any]:
    scattered = {}
    for key, value in obs.items():
        selected_value = selected_obs.get(key)
        if value is None or selected_value is None:
            scattered[key] = value
        elif isinstance(value, torch.Tensor):
            updated = value.clone()
            updated[torch.as_tensor(indices, device=value.device)] = selected_value
            scattered[key] = updated
        elif isinstance(value, np.ndarray):
            updated = value.copy()
            updated[indices] = selected_value
            scattered[key] = updated
        else:
            updated = list(value)
            for selected_index, original_index in enumerate(indices):
                updated[original_index] = selected_value[selected_index]
            scattered[key] = updated
    return scattered


def _scatter_sequence_rows(values, selected_values, indices: np.ndarray):
    updated = list(values)
    for selected_index, original_index in enumerate(indices):
        updated[original_index] = selected_values[selected_index]
    return updated


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self._sokoban_image_save_error_reported = False

    @staticmethod
    def _object_array(values: List[Any]) -> np.ndarray:
        array = np.empty(len(values), dtype=object)
        for idx, value in enumerate(values):
            array[idx] = value
        return array

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return str(value)
        if isinstance(value, torch.Tensor):
            value = torch_to_numpy(value, is_object=True)
        if isinstance(value, np.ndarray):
            try:
                value = value.tolist()
            except Exception:
                pass
        return str(value)

    @staticmethod
    def _get_indexed(values: Any, index: int, default: Any = None) -> Any:
        if values is None:
            return default
        try:
            return values[index]
        except Exception:
            return default

    def _config_select(self, key: str, default: Any = None) -> Any:
        try:
            value = OmegaConf.select(self.config, key)
        except Exception:
            value = default
        return default if value is None else value

    def _config_bool(self, key: str, default: bool = False) -> bool:
        value = self._config_select(key, default)
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _sokoban_image_saving_enabled(self) -> bool:
        env_name = str(self._config_select("env.env_name", ""))
        return "sokoban" in env_name.lower() and self._config_bool("env.sokoban.save_images", False)

    def _sokoban_image_root(self) -> Optional[str]:
        image_save_dir = self._config_select("env.sokoban.image_save_dir")
        if image_save_dir:
            return os.path.expanduser(str(image_save_dir))
        default_local_dir = self._config_select("trainer.default_local_dir")
        if not default_local_dir:
            return None
        return os.path.join(os.path.expanduser(str(default_local_dir)), "sokoban_images")

    @staticmethod
    def _sanitize_path_component(value: Any) -> str:
        text = str(value)
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "unknown"

    @staticmethod
    def _global_step_dir_name(global_step: Any) -> str:
        try:
            return f"global_step_{int(global_step)}"
        except (TypeError, ValueError):
            return "global_step_unknown"

    @staticmethod
    def _image_to_uint8_array(image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)

        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[:, :, 0]

        if image.dtype != np.uint8:
            image = image.astype(np.float32, copy=False)
            if image.size and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _save_sokoban_observation_images(
        self,
        obs: Dict[str, Any],
        *,
        traj_uid: np.ndarray,
        sample_ids: np.ndarray,
        rollout_ids: np.ndarray,
        step_num: int,
        global_step: Any,
        active_masks: np.ndarray,
        phase: str,
    ) -> None:
        if not self._sokoban_image_saving_enabled():
            return

        images = obs.get("image")
        if images is None:
            return

        root = self._sokoban_image_root()
        if root is None:
            return

        try:
            from PIL import Image

            batch_size = len(traj_uid)
            for sample_idx in range(batch_size):
                if not bool(active_masks[sample_idx]):
                    continue
                image = self._get_indexed(images, sample_idx)
                if image is None:
                    continue

                sequence_name = (
                    f"{phase}_sample_{int(sample_ids[sample_idx]):06d}"
                    f"_rollout_{int(rollout_ids[sample_idx]):03d}"
                    f"_{self._sanitize_path_component(traj_uid[sample_idx])}"
                )
                sequence_dir = os.path.join(root, self._global_step_dir_name(global_step), sequence_name)
                os.makedirs(sequence_dir, exist_ok=True)
                image_array = self._image_to_uint8_array(image)
                Image.fromarray(image_array).save(
                    os.path.join(sequence_dir, f"step_{int(step_num):03d}.png")
                )
        except Exception as exc:
            if not self._sokoban_image_save_error_reported:
                print(f"Warning: failed to save Sokoban observation images: {exc}")
                self._sokoban_image_save_error_reported = True

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_response_action_text(self, response_text: str) -> str:
        text = str(response_text or "")
        return self._extract_tag(text, "search") or self._extract_tag(text, "action") or text.strip()

    def _env_aux_metadata_enabled(self) -> bool:
        try:
            actor_config = self.config.actor_rollout_ref.actor
        except Exception:
            return False
        collect_only = bool(actor_config.get("collect_env_aux_data", False))
        sp_coef = float(actor_config.get("sp_coef", 0.0) or 0.0)
        id_coef = float(actor_config.get("id_coef", 0.0) or 0.0)
        return collect_only or sp_coef > 0.0 or id_coef > 0.0

    def _extract_observation_text(self, obs: Dict[str, Any], index: int) -> str:
        for key in ("anchor", "text_base", "text"):
            value = self._get_indexed(obs.get(key), index)
            text = self._to_text(value).strip()
            if text:
                return text
        return ""

    def _extract_admissible_actions(self, info: Any) -> List[str]:
        if not isinstance(info, dict):
            return []

        for key in ("admissible_commands", "admissible_actions", "valid", "possible_actions"):
            actions = info.get(key)
            if actions is not None:
                if isinstance(actions, str):
                    return [line.strip(" '-,") for line in actions.splitlines() if line.strip()]
                if isinstance(actions, (list, tuple, np.ndarray)):
                    return [self._to_text(action).strip() for action in actions if self._to_text(action).strip()]
                return [self._to_text(actions).strip()]

        available_actions = info.get("available_actions")
        if isinstance(available_actions, dict):
            actions = []
            if available_actions.get("has_search_bar"):
                actions.append("search[<your query>]")
            for clickable in available_actions.get("clickables", []) or []:
                actions.append(f"click[{clickable}]")
            return actions
        if available_actions is not None:
            return [self._to_text(available_actions).strip()]

        return []

    @staticmethod
    def _normalize_prompt_images(images: Any) -> List[Any]:
        if images is None:
            return []
        if isinstance(images, (list, tuple)):
            return list(images)
        if isinstance(images, np.ndarray) and images.ndim == 4:
            return [images[idx] for idx in range(images.shape[0])]
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            return [images[idx] for idx in range(images.shape[0])]
        return [images]

    def build_prompt_sample(
        self,
        obs_content: str,
        data_source: Optional[str] = None,
        max_prompt_length: Optional[int] = None,
        images: Any = None,
    ) -> Dict:
        """
        Build a prompt sample using the same chat-template path as rollout.
        This is used by SEED teacher scoring to reconstruct prompt-enhanced inputs.
        """
        prompt_length = int(max_prompt_length or self.config.data.max_prompt_length)
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        prompt_images = self._normalize_prompt_images(images)
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        row_dict = {}

        if prompt_images:
            if self.processor is None:
                raise RuntimeError("Multimodal prompt construction requires a processor.")
            placeholder_count = prompt_with_chat_template.count("<image>")
            if placeholder_count == 0:
                prompt_with_chat_template = ("<image>\n" * len(prompt_images)) + prompt_with_chat_template
                placeholder_count = len(prompt_images)
            if placeholder_count != len(prompt_images):
                raise RuntimeError(
                    f"Prompt has {placeholder_count} <image> placeholder(s), "
                    f"but {len(prompt_images)} image(s) were provided."
                )

            raw_prompt = prompt_with_chat_template.replace(
                "<image>",
                "<|vision_start|><|image_pad|><|vision_end|>",
            )
            row_dict["multi_modal_data"] = {
                "image": [
                    process_image(self._image_to_uint8_array(image))
                    for image in prompt_images
                ]
            }
            image_inputs = self.processor.image_processor(
                row_dict["multi_modal_data"]["image"],
                return_tensors="pt",
            )
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict["multi_modal_inputs"] = {
                key: val for key, val in image_inputs.items()
            }
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                for image_idx in range(len(prompt_images)):
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[image_idx].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )

                prompt_with_chat_template = prompt_with_chat_template.replace(
                    "<|placeholder|>",
                    self.processor.image_token,
                )
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )

        if prompt_images:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-prompt_length:]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = prompt_length // 2
                right_half = prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {prompt_length}."
                )

        row_dict.update({
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0] if isinstance(position_ids, list) else position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "obs_text": obs_content,
            "data_source": data_source,
        })
        return row_dict

    def build_text_prompt_sample(
        self,
        obs_content: str,
        data_source: Optional[str] = None,
        max_prompt_length: Optional[int] = None,
    ) -> Dict:
        """
        Build a text-only prompt sample using the same chat-template path as rollout.
        """
        return self.build_prompt_sample(
            obs_content=obs_content,
            data_source=data_source,
            max_prompt_length=max_prompt_length,
            images=None,
        )

    def build_prompt_batch(
        self,
        obs_contents: List[str],
        data_sources: Optional[List[Optional[str]]] = None,
        meta_info: Optional[Dict] = None,
        max_prompt_length: Optional[int] = None,
        images: Optional[List[Any]] = None,
    ) -> DataProto:
        """
        Build a batch of text or multimodal prompts. Used for SEED analysis
        and teacher scoring.
        """
        processed_samples = []
        for sample_idx, obs_content in enumerate(obs_contents):
            data_source = None if data_sources is None else data_sources[sample_idx]
            sample_images = None if images is None else images[sample_idx]
            processed_samples.append(
                self.build_prompt_sample(
                    obs_content=obs_content,
                    data_source=data_source,
                    max_prompt_length=max_prompt_length,
                    images=sample_images,
                )
            )
        batch = collate_fn(processed_samples)
        return DataProto.from_single_dict(data=batch, meta_info=meta_info)

    def build_text_prompt_batch(
        self,
        obs_contents: List[str],
        data_sources: Optional[List[Optional[str]]] = None,
        meta_info: Optional[Dict] = None,
        max_prompt_length: Optional[int] = None,
    ) -> DataProto:
        """
        Build a batch of text-only prompts. Used for SEED teacher scoring.
        """
        return self.build_prompt_batch(
            obs_contents=obs_contents,
            data_sources=data_sources,
            meta_info=meta_info,
            max_prompt_length=max_prompt_length,
            images=None,
        )

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_base_texts = obs.get('text_base', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_text_base = obs_base_texts[item] if obs_base_texts is not None else obs_text
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(self._image_to_uint8_array(obs_image))]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'obs_text': obs_content,
            'obs_text_base': "" if obs_text_base is None else str(obs_text_base),
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def _prepare_and_generate_batch(
        self,
        gen_batch: DataProto,
        obs: Dict[str, Any],
        actor_rollout_wg,
        original_indices=None,
    ) -> DataProto:
        batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
        if original_indices is not None:
            if "index" in batch.batch:
                index_values = batch.batch["index"]
                batch.batch["index"] = torch.as_tensor(
                    original_indices,
                    dtype=index_values.dtype,
                    device=index_values.device,
                )
            elif "index" in batch.non_tensor_batch:
                batch.non_tensor_batch["index"] = np.asarray(
                    original_indices,
                    dtype=batch.non_tensor_batch["index"].dtype,
                )

        non_tensor_batch_keys = ["raw_prompt_ids"]
        for key in ("multi_modal_data", "raw_prompt", "tools_kwargs"):
            if key in batch.non_tensor_batch:
                non_tensor_batch_keys.append(key)

        batch_input = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=non_tensor_batch_keys,
        )
        batch_input.meta_info = gen_batch.meta_info
        batch_input, pad_size = pad_dataproto_to_divisor(
            batch_input,
            actor_rollout_wg.world_size,
        )
        batch_output = actor_rollout_wg.generate_sequences(batch_input)
        return batch.union(unpad_dataproto(batch_output, pad_size=pad_size))


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        rollout_group_size = int(self.config.env.rollout.n) if self.config.env.rollout.n > 0 else 1
        for bs in range(batch_size):
            sample_id = bs // rollout_group_size
            rollout_id = bs % rollout_group_size
            # sum the rewards for each data in total_batch_list[bs]
            for step_position, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    step_num = int(data.get('step_num', step_position))
                    data['sample_id'] = sample_id
                    data['rollout_id'] = rollout_id
                    data['step_num'] = step_num
                    data['step_id'] = f"{sample_id}_{rollout_id}_{step_num}"
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            phase: str = "train",
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)
        global_step = (gen_batch.meta_info or {}).get("global_step")

        # Initial observations from the environment
        reset_result = cast(Any, envs).reset(
            kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None)
        )
        obs = cast(Dict[str, Any], reset_result[0])
        infos = list(reset_result[1])

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        rollout_group_size = int(self.config.env.rollout.n) if self.config.env.rollout.n > 0 else 1
        sample_ids = np.asarray(
            [i // rollout_group_size for i in range(batch_size)],
            dtype=np.int64,
        )
        rollout_ids = np.asarray(
            [i % rollout_group_size for i in range(batch_size)],
            dtype=np.int64,
        )
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        collect_env_aux_data = self._env_aux_metadata_enabled()
        env_aux_histories = [[] for _ in range(batch_size)]
        with _rollout_session(actor_rollout_wg):
            for _step in range(self.config.env.max_steps):
                active_masks = np.logical_not(is_done)
                self._save_sokoban_observation_images(
                    obs,
                    traj_uid=traj_uid,
                    sample_ids=sample_ids,
                    rollout_ids=rollout_ids,
                    step_num=_step,
                    global_step=global_step,
                    active_masks=active_masks,
                    phase=phase,
                )

                if hasattr(envs, "step_selected"):
                    active_indices = np.flatnonzero(active_masks)
                    active_gen_batch = cast(DataProto, gen_batch[active_indices])
                    active_obs = _select_observation_rows(obs, active_indices)
                    batch = self._prepare_and_generate_batch(
                        active_gen_batch,
                        active_obs,
                        actor_rollout_wg,
                        original_indices=active_indices,
                    )

                    batch.non_tensor_batch['uid'] = uid_batch[active_indices]
                    batch.non_tensor_batch['traj_uid'] = traj_uid[active_indices]
                    active_sample_ids = sample_ids[active_indices]
                    active_rollout_ids = rollout_ids[active_indices]
                    active_step_nums = np.full(
                        len(active_indices),
                        _step,
                        dtype=np.int64,
                    )
                    batch.non_tensor_batch['sample_id'] = active_sample_ids
                    batch.non_tensor_batch['rollout_id'] = active_rollout_ids
                    batch.non_tensor_batch['step_num'] = active_step_nums
                    batch.non_tensor_batch['step_id'] = np.asarray(
                        [
                            f"{int(sample_id)}_{int(rollout_id)}_{int(step_num)}"
                            for sample_id, rollout_id, step_num in zip(
                                active_sample_ids,
                                active_rollout_ids,
                                active_step_nums,
                            )
                        ],
                        dtype=object,
                    )
                    if collect_env_aux_data:
                        batch.non_tensor_batch["history"] = self._object_array(
                            [
                                list(env_aux_histories[index])
                                for index in active_indices
                            ]
                        )
                        batch.non_tensor_batch["admissibles"] = self._object_array(
                            [
                                self._extract_admissible_actions(infos[index])
                                for index in active_indices
                            ]
                        )

                    text_actions = self.tokenizer.batch_decode(
                        batch.batch['responses'],
                        skip_special_tokens=True,
                    )
                    selected_step = cast(Any, envs).step_selected
                    next_active_obs, rewards, dones, active_infos = selected_step(
                        text_actions,
                        active_indices.tolist(),
                    )
                    if collect_env_aux_data:
                        batch.non_tensor_batch["next_obs"] = self._object_array(
                            [
                                self._extract_observation_text(
                                    next_active_obs,
                                    selected_index,
                                )
                                for selected_index in range(len(active_indices))
                            ]
                        )

                    if len(rewards.shape) == 2:
                        rewards = rewards.squeeze(1)
                    if len(dones.shape) == 2:
                        dones = dones.squeeze(1)

                    if 'is_action_valid' in active_infos[0]:
                        batch.non_tensor_batch['is_action_valid'] = np.array(
                            [
                                info['is_action_valid']
                                for info in active_infos
                            ],
                            dtype=bool,
                        )
                    else:
                        batch.non_tensor_batch['is_action_valid'] = np.ones(
                            len(active_indices),
                            dtype=bool,
                        )

                    if 'tool_calling' in active_infos[0]:
                        tool_callings[active_indices] += np.array(
                            [
                                info['tool_calling']
                                for info in active_infos
                            ],
                            dtype=np.float32,
                        )
                    episode_rewards[active_indices] += torch_to_numpy(rewards)
                    episode_lengths[active_indices] += 1

                    assert len(rewards) == len(active_indices), (
                        "env should return rewards for selected environments, "
                        f"got {len(rewards)} rewards for {len(active_indices)} "
                        "environments"
                    )
                    batch.non_tensor_batch['rewards'] = torch_to_numpy(
                        rewards,
                        is_object=True,
                    )
                    batch.non_tensor_batch['active_masks'] = np.full(
                        len(active_indices),
                        True,
                        dtype=object,
                    )

                    batch_list: list[dict] = to_list_of_dict(batch)
                    for selected_index, original_index in enumerate(active_indices):
                        total_batch_list[original_index].append(
                            batch_list[selected_index]
                        )
                        total_infos[original_index].append(
                            active_infos[selected_index]
                        )

                    if collect_env_aux_data:
                        for selected_index, original_index in enumerate(
                            active_indices
                        ):
                            current_obs = self._get_indexed(
                                batch.non_tensor_batch.get("anchor_obs"),
                                selected_index,
                            )
                            current_obs_text = self._to_text(current_obs).strip()
                            if not current_obs_text:
                                current_obs_text = self._to_text(
                                    self._get_indexed(
                                        batch.non_tensor_batch.get("obs_text"),
                                        selected_index,
                                    )
                                ).strip()
                            env_aux_histories[original_index].append(
                                {
                                    "text_obs": current_obs_text,
                                    "action": self._extract_response_action_text(
                                        text_actions[selected_index]
                                    ),
                                }
                            )

                    is_done[active_indices] = np.logical_or(
                        is_done[active_indices],
                        dones,
                    )
                    obs = _scatter_observation_rows(
                        obs,
                        next_active_obs,
                        active_indices,
                    )
                    infos = _scatter_sequence_rows(
                        infos,
                        active_infos,
                        active_indices,
                    )
                    if is_done.all():
                        break
                    continue

                batch = self._prepare_and_generate_batch(
                    gen_batch,
                    obs,
                    actor_rollout_wg,
                )
                batch.non_tensor_batch['uid'] = uid_batch
                batch.non_tensor_batch['traj_uid'] = traj_uid
                step_nums = np.full(batch_size, _step, dtype=np.int64)
                batch.non_tensor_batch['sample_id'] = sample_ids
                batch.non_tensor_batch['rollout_id'] = rollout_ids
                batch.non_tensor_batch['step_num'] = step_nums
                batch.non_tensor_batch['step_id'] = np.asarray(
                    [
                        f"{int(sample_ids[i])}_{int(rollout_ids[i])}_{int(step_nums[i])}"
                        for i in range(batch_size)
                    ],
                    dtype=object,
                )
                if collect_env_aux_data:
                    batch.non_tensor_batch["history"] = self._object_array(
                        [list(env_aux_histories[i]) for i in range(batch_size)]
                    )
                    batch.non_tensor_batch["admissibles"] = self._object_array(
                        [
                            self._extract_admissible_actions(infos[i])
                            for i in range(batch_size)
                        ]
                    )

                text_actions = self.tokenizer.batch_decode(
                    batch.batch['responses'],
                    skip_special_tokens=True,
                )
                next_obs, rewards, dones, infos = envs.step(text_actions)
                if collect_env_aux_data:
                    batch.non_tensor_batch["next_obs"] = self._object_array(
                        [
                            self._extract_observation_text(next_obs, i)
                            for i in range(batch_size)
                        ]
                    )

                if len(rewards.shape) == 2:
                    rewards = rewards.squeeze(1)
                if len(dones.shape) == 2:
                    dones = dones.squeeze(1)

                if 'is_action_valid' in infos[0]:
                    batch.non_tensor_batch['is_action_valid'] = np.array(
                        [info['is_action_valid'] for info in infos],
                        dtype=bool,
                    )
                else:
                    batch.non_tensor_batch['is_action_valid'] = np.ones(
                        batch_size,
                        dtype=bool,
                    )

                if 'tool_calling' in infos[0]:
                    tool_callings[active_masks] += np.array(
                        [info['tool_calling'] for info in infos],
                        dtype=np.float32,
                    )[active_masks]
                episode_rewards[active_masks] += torch_to_numpy(rewards)[
                    active_masks
                ]
                episode_lengths[active_masks] += 1

                assert len(rewards) == batch_size, (
                    "env should return rewards for all environments, "
                    f"got {len(rewards)} rewards for {batch_size} environments"
                )
                batch.non_tensor_batch['rewards'] = torch_to_numpy(
                    rewards,
                    is_object=True,
                )
                batch.non_tensor_batch['active_masks'] = torch_to_numpy(
                    active_masks,
                    is_object=True,
                )

                batch_list: list[dict] = to_list_of_dict(batch)
                for i in range(batch_size):
                    total_batch_list[i].append(batch_list[i])
                    total_infos[i].append(infos[i])

                if collect_env_aux_data:
                    for i in range(batch_size):
                        if active_masks[i]:
                            current_obs = self._get_indexed(
                                batch.non_tensor_batch.get("anchor_obs"),
                                i,
                            )
                            current_obs_text = self._to_text(current_obs).strip()
                            if not current_obs_text:
                                current_obs_text = self._to_text(
                                    self._get_indexed(
                                        batch.non_tensor_batch.get("obs_text"),
                                        i,
                                    )
                                ).strip()
                            env_aux_histories[i].append(
                                {
                                    "text_obs": current_obs_text,
                                    "action": self._extract_response_action_text(
                                        text_actions[i]
                                    ),
                                }
                            )

                is_done = np.logical_or(is_done, dones)
                obs = next_obs
                if is_done.all():
                    break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            phase: str = "train",
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches
        target_trajectories = (
            self.config.data.train_batch_size
            * self.config.env.rollout.n
        )
        filter_mode = str(
            self.config.algorithm.get("trajectory_grpo", {}).get(
                "filter",
                "off",
            )
        ).replace("-", "_")
        penalty_aware = filter_mode == "penalty_aware"

        while len(total_batch_list) < target_trajectories and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={target_trajectories}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                phase=phase,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )

            remaining = target_trajectories - len(total_batch_list)
            if penalty_aware and len(batch_list) > remaining:
                accepted_indices = take_complete_uid_groups(
                    [trajectory[0]["uid"] for trajectory in batch_list],
                    remaining,
                )
                batch_list = [
                    batch_list[index]
                    for index in accepted_indices
                ]
                episode_rewards = episode_rewards[accepted_indices]
                episode_lengths = episode_lengths[accepted_indices]
                success = {
                    key: value[accepted_indices]
                    for key, value in success.items()
                    if len(value) == len(traj_uid)
                }
                traj_uid = traj_uid[accepted_indices]
                tool_callings = tool_callings[accepted_indices]

            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        if penalty_aware:
            require_filter_target_reached(
                len(total_batch_list),
                target_trajectories,
                max_try_count,
            )

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        filter_mode = str(
            self.config.algorithm.get("trajectory_grpo", {}).get(
                "filter",
                "off",
            )
        ).replace("-", "_")
        if (
            self.config.algorithm.filter_groups.enable
            or filter_mode == "penalty_aware"
        ) and is_train:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                phase="train" if is_train else "val",
            )
        else:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                phase="train" if is_train else "val",
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)

        return self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import inspect
import logging
import os
import time
from collections import OrderedDict

import torch
from peft import PeftModel
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available
from verl.utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge, patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



class FSDPVLLMShardingManager(BaseShardingManager):
    supports_rollout_session = True

    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        load_format: str = 'dummy_hf',
        layered_summon: bool = True
    ):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer intialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine
        # self.model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if inference_engine else None

        if "vllm_v_0_6_3" in str(type(self.inference_engine)) or "vllm_v_0_5_4" in str(type(self.inference_engine)):
            # vLLM <= v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.worker.model_runner if self.inference_engine else None
        else:
            # vLLM > v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None

        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = 'dummy' not in load_format
        self._enter_actor_params_loaded = False
        self._enter_rollout_awake = False
        self._enter_rng_switch_attempted = False
        self._enter_rng_switched = False
        self._enter_train_mode_restore_needed = False
        self._enter_cache_cleanup_needed = False
        self._enter_succeeded = False
        self._tainted = False
        self._taint_reason = None
        if is_version_ge(pkg='vllm', minver='0.7.3'):
            VLLMHijack.hijack()

    @staticmethod
    def _format_cleanup_failure(cleanup_error, action):
        try:
            return f"FSDP-vLLM cleanup failed during {action}: {cleanup_error!r}"
        except BaseException:
            return f"FSDP-vLLM cleanup failed during {action}"

    @staticmethod
    def _add_error_note(error, note):
        if hasattr(error, "add_note"):
            try:
                error.add_note(note)
            except BaseException:
                pass

    def _record_cleanup_failure(self, original_error, cleanup_error, action):
        note = self._format_cleanup_failure(cleanup_error, action)
        self._tainted = True
        if self._taint_reason is None:
            self._taint_reason = note
        self._add_error_note(original_error, note)
        try:
            logger.error(
                note,
                exc_info=(
                    type(cleanup_error),
                    cleanup_error,
                    cleanup_error.__traceback__,
                ),
            )
        except Exception:
            pass

    @staticmethod
    def _rng_states_equal(current_state, expected_state):
        if isinstance(current_state, torch.Tensor) and isinstance(
            expected_state,
            torch.Tensor,
        ):
            return torch.equal(current_state, expected_state)
        return bool(current_state == expected_state)

    def _resolve_failed_rng_switch(self, switch_error):
        try:
            current_state = get_torch_device().get_rng_state()
            unchanged = self._rng_states_equal(
                current_state,
                self.torch_random_states,
            )
        except BaseException as inspection_error:
            self._record_cleanup_failure(
                switch_error,
                inspection_error,
                "RNG state inspection after failed switch",
            )
            return

        self._enter_rng_switch_attempted = False
        self._enter_rng_switched = not unchanged

    def _cleanup_enter_state(self, original_error=None):
        cleanup_error = original_error
        cleanup_failed = False

        def run_cleanup(action, cleanup_fn):
            nonlocal cleanup_error, cleanup_failed
            try:
                cleanup_fn()
            except BaseException as error:
                cleanup_failed = True
                if cleanup_error is None:
                    cleanup_error = error
                self._record_cleanup_failure(
                    cleanup_error,
                    error,
                    action,
                )
                return False
            return True

        if self._enter_rng_switch_attempted or self._enter_rng_switched:
            def restore_rng():
                if self._enter_succeeded:
                    self.gen_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.torch_random_states)

            if run_cleanup("RNG restoration", restore_rng):
                self._enter_rng_switch_attempted = False
                self._enter_rng_switched = False

        if self._enter_rollout_awake:
            if vllm_version in ("0.5.4", "0.6.3"):
                rollout_cleaned = run_cleanup(
                    "rollout weight offload",
                    self.inference_engine.offload_model_weights,
                )
            else:
                rollout_cleaned = run_cleanup(
                    "rollout sleep",
                    lambda: self.inference_engine.sleep(level=1),
                )
            if rollout_cleaned:
                self._enter_rollout_awake = False

        if self._enter_actor_params_loaded:
            if run_cleanup(
                "actor parameter offload",
                lambda: offload_fsdp_model_to_cpu(self.module),
            ):
                self._enter_actor_params_loaded = False

        if self._enter_train_mode_restore_needed:
            if run_cleanup("actor train mode restoration", self.module.train):
                self._enter_train_mode_restore_needed = False

        if self._enter_cache_cleanup_needed:
            if run_cleanup(
                "device cache cleanup",
                lambda: get_torch_device().empty_cache(),
            ):
                self._enter_cache_cleanup_needed = False

        if not cleanup_failed:
            self._enter_succeeded = False

        if original_error is None and cleanup_error is not None:
            raise cleanup_error

    def __enter__(self):
        if self._tainted:
            raise RuntimeError(
                f"FSDP-vLLM sharding manager is tainted: {self._taint_reason}"
            )
        if (
            self._enter_succeeded
            or self._enter_actor_params_loaded
            or self._enter_rollout_awake
            or self._enter_rng_switch_attempted
            or self._enter_rng_switched
            or self._enter_train_mode_restore_needed
            or self._enter_cache_cleanup_needed
        ):
            raise RuntimeError("FSDP-vLLM sharding manager is already entered")

        self._enter_cache_cleanup_needed = True
        try:
            self._enter_impl()
        except BaseException as enter_error:
            self._cleanup_enter_state(original_error=enter_error)
            raise
        self._enter_succeeded = True

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def _enter_impl(self):
        def __collect_lora_params():
            """
            collect lora params or full params if base model is not ready in vllm
            work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
            """
            from peft.utils.save_and_load import get_peft_model_state_dict

            lora_params = OrderedDict()
            if fsdp_version(self.module) > 0:
                if self.layered_summon:
                    if not self.base_sync_done:
                        raise ValueError("To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let rollout.load_format=safetensors")
                    lora_params = layered_summon_lora_params(self.module)
                else:
                    with FSDP.summon_full_params(self.module, writeback=False):
                        if self.base_sync_done:
                            lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                            lora_params = {name: param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu() 
                                        for name, param in lora_params.items()}
                        else:
                            model = self.module._fsdp_wrapped_module.base_model.model
                            orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device.type else 'cuda'
                            model = model.to('cpu')
                            for name, param in model.state_dict().items():
                                if any(x in name for x in ['_flat_param', 'lora_']):
                                    continue
                                name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                                lora_params[name] = param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu()
                            model = model.to(orig_dev)
                    torch.cuda.empty_cache()
            else:
                if self.base_sync_done:
                    lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                else:
                    model = self.module._fsdp_wrapped_module.base_model.model
                    orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device.type else 'cuda'
                    model = model.to('cpu')
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ['_flat_param', 'lora_']):
                            continue
                        name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                        lora_params[name] = param.detach().cpu()
                    model = model.to(orig_dev)
            return lora_params

        # NOTE: Basically, we only need `get_torch_device().empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        get_torch_device().empty_cache()
        self._enter_train_mode_restore_needed = True

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            self._enter_actor_params_loaded = True
            load_fsdp_model_to_gpu(self.module)

        peft_config = None
        if isinstance(self.module._fsdp_wrapped_module, PeftModel):
            peft_config = self.module._fsdp_wrapped_module.peft_config.get('default', None)
            params = __collect_lora_params()
        else:
            params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

        # Copy, not share memory
        load_format = "hf" if self.full_params else "dtensor"

        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self._enter_rollout_awake = True
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
        else:
            self._enter_rollout_awake = True
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            self.update_params(params, peft_config=peft_config)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
                self._enter_actor_params_loaded = False
            get_torch_device().empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            self._enter_rng_switch_attempted = True
            try:
                get_torch_device().set_rng_state(self.gen_random_states)
            except BaseException as switch_error:
                self._resolve_failed_rng_switch(switch_error)
                raise
            self._enter_rng_switch_attempted = False
            self._enter_rng_switched = True

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._enter_succeeded:
            return
        self._cleanup_enter_state()

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config=None):
        model = self.model_runner.model
        if peft_config:
            if self.base_sync_done:
                lora_int_id=int(time.time_ns() % 0x7FFFFFFF)
                lora_reqest = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                self.inference_engine.llm_engine.add_lora(lora_reqest)
                logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}")
                return
            else:
                def replace_lora_wrapper(k):
                    stacked_params = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                    if any([k.endswith(f"{s}.weight") for s in stacked_params]):
                        return k.replace(".weight", ".base_layer.weight")
                    if any([k.endswith(f"{s}.bias") for s in stacked_params]):
                        return k.replace(".bias", ".base_layer.bias")
                    return k
                updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

        patch_vllm_moe_model_weight_loader(model)
        device = get_torch_device().current_device()  # used when fsdp2 set cpu_offload_policy
        loaded_params = model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param) for name, param in updated_params.items()))

        self.base_sync_done = True
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")

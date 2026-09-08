# Copyright 2023-2026 SGLang Team
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
# ==============================================================================
"""TorchAir GE backend for the fixed-width NPU DSPark draft graph.

Unlike :class:`NPUCudaGraphBackend`, this backend does not put an outer
``torch.npu.NPUGraph`` around the model.  ``forward_fn`` is already a
max-autotune TorchAir callable, so retaining and invoking it replays the GE
graph directly.  This is required by ``torchair.ops`` FIA, whose device-tensor
sequence-length interface is unavailable in eager and reduce-overhead modes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class NpuGEGraphBackend(BaseCudaGraphBackend):
    """One max-autotune TorchAir GE callable per static DSPark shape."""

    # GE consumes the shared metadata tensors asynchronously after the Python
    # call returns.  Callers that publish a WAR event must record it after the
    # GE launch, not before it.
    shared_read_ends_after_replay = True

    def __init__(self, cuda_graph_runner: BaseCudaGraphRunner) -> None:
        self._forwards: Dict[Any, Callable[[], Any]] = {}
        self._outputs: Dict[Any, Any] = {}
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group

    @contextmanager
    def capture_session(self, stream):
        # The surrounding runner has already selected the capture stream and
        # put graph-safe communicators into capture mode.
        yield

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # First invocation compiles the GE graph; the second validates its
        # cached execution path and pays remaining one-time kernel setup.
        out = None
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            out = forward_fn()
            self._device_module.synchronize()
            if post_warmup_hook is not None:
                post_warmup_hook()

        self._forwards[shape_key] = forward_fn
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._forwards

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        out = self._forwards[shape_key]()
        self._outputs[shape_key] = out
        return out

    def cleanup(self) -> None:
        self._forwards.clear()
        self._outputs.clear()

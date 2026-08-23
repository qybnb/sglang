from contextlib import AbstractContextManager
from unittest.mock import patch

from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend


class _GraphContext(AbstractContextManager):
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("graph-enter")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.events.append("graph-exit")


class _FakeNPU:
    def __init__(self, events):
        self.events = events

    def NPUGraph(self):
        self.events.append("graph-create")
        return object()

    def graph(self, *args, **kwargs):
        return _GraphContext(self.events)


class _FakeDeviceModule:
    def __init__(self, events):
        self.events = events

    def synchronize(self):
        self.events.append("synchronize")


class _FakeTPGroup:
    def __init__(self, events):
        self.events = events

    def barrier(self):
        self.events.append("barrier")


def test_npu_graph_capture_aligns_ranks_after_final_warmup():
    events = []
    backend = object.__new__(npu_cudagraph_backend.NPUCudaGraphBackend)
    backend._device_module = _FakeDeviceModule(events)
    backend._tp_group = _FakeTPGroup(events)
    backend._enable_torch_compile = False
    backend._memory_saver_adapter = None
    backend._pool = object()
    backend._capture_stream = object()
    backend._graphs = {}
    backend._outputs = {}

    def forward():
        events.append("forward")
        return "output"

    def post_warmup():
        events.append("post-warmup")

    fake_npu = _FakeNPU(events)
    with patch.object(npu_cudagraph_backend.torch, "npu", fake_npu, create=True):
        backend.capture_one("shape", forward, post_warmup_hook=post_warmup)

    assert events == [
        "synchronize",
        "barrier",
        "forward",
        "post-warmup",
        "synchronize",
        "barrier",
        "forward",
        "post-warmup",
        # This boundary prevents the captured DeepEP epoch from overlapping
        # another rank's final warmup epoch.
        "synchronize",
        "barrier",
        "graph-create",
        "graph-enter",
        "forward",
        "graph-exit",
    ]
    assert backend._outputs["shape"] == "output"

"""Opt-in observation only; no model, kernel, or scheduling policy replacement.

The companion runner prepends this directory to its server's PYTHONPATH and
sets SGLANG_MIN_SLOTS_TEST_TRACE_DIR. Other Python processes do nothing here.
All wrapped methods call the original implementation and return its result.
"""

import contextvars
import functools
import importlib.abc
import importlib.machinery
import itertools
import json
import os
import sys
import time

TRACE_DIR = os.environ.get("SGLANG_MIN_SLOTS_TEST_TRACE_DIR")
_active_pass = contextvars.ContextVar("min_slots_observation", default=None)
_sequence = itertools.count()


def emit(event, **fields):
    row = dict(event=event, time=time.time(), pid=os.getpid(), **fields)
    with open(f"{TRACE_DIR}/trace-{os.getpid()}.jsonl", "a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")


def instrument(module):
    from sglang.srt.managers.min_free_slots_delayer import MinFreeSlotsDelayer

    scheduler = module.Scheduler
    original_prefill = scheduler._get_new_batch_prefill_raw
    original_decision = MinFreeSlotsDelayer.should_delay

    @functools.wraps(original_decision)
    def observed_decision(self, *, running_bs, num_allocatable_reqs):
        delayed = original_decision(
            self, running_bs=running_bs, num_allocatable_reqs=num_allocatable_reqs
        )
        observation = _active_pass.get()
        row = dict(
            running=running_bs,
            slots=num_allocatable_reqs,
            threshold=self._min_free_slots,
            delayed=delayed,
        )
        if observation is not None:
            observation["decision"] = row
        emit("decision", pass_id=observation["pass_id"] if observation else None, **row)
        return delayed

    @functools.wraps(original_prefill)
    def observed_prefill(self, prefill_delayer_single_pass, running_batch):
        observation = {"pass_id": next(_sequence)}
        running = len(running_batch.reqs)
        slots = self.get_num_allocatable_reqs(running, running_batch=running_batch)
        waiting = len(self.waiting_queue)
        chunk = self.chunked_req is not None
        delayer = self.min_free_slots_delayer
        threshold = delayer._min_free_slots if delayer is not None else None
        token = _active_pass.set(observation)
        try:
            result = original_prefill(self, prefill_delayer_single_pass, running_batch)
        finally:
            _active_pass.reset(token)
        batch = result[0]
        decision = observation.get("decision")
        if waiting or chunk or batch is not None or decision is not None:
            emit(
                "prefill",
                pass_id=observation["pass_id"],
                running=running,
                slots=decision["slots"] if decision else slots,
                waiting=waiting,
                chunk=chunk,
                threshold=threshold,
                overlap=self.enable_overlap,
                max_running_requests=self.max_running_requests,
                admitted=len(batch.reqs) if batch is not None else 0,
                decision_called=decision is not None,
                delayed=decision["delayed"] if decision else None,
            )
        return result

    scheduler._get_new_batch_prefill_raw = observed_prefill
    MinFreeSlotsDelayer.should_delay = observed_decision
    emit("instrumented", source=module.__file__, kernel_fallback=False)


class ObserverLoader:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        try:
            instrument(module)
        except Exception as exc:
            emit("observer_error", error=repr(exc))
            raise


class ObserverFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang.srt.managers.scheduler":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = ObserverLoader(spec.loader)
        return spec


if TRACE_DIR:
    try:
        emit("observer_ready")
        sys.meta_path.insert(0, ObserverFinder())
    except Exception as exc:
        # Python normally prints and ignores sitecustomize import errors. The
        # runner also requires trace evidence, so this can never be a PASS.
        print(f"min-free-slots observer failed: {exc!r}", file=sys.stderr)
        raise

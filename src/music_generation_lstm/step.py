from dataclasses import dataclass
from functools import wraps
import logging
import time

REGISTERED_STEPS = []  # For debugging purposes

def pipeline_step(*, kind=None, context=False):
    def decorator(fn):
        step_kind = kind
        step_context = context

        REGISTERED_STEPS.append({
            "kind": step_kind,
            "func": fn
        })

        @wraps(fn)
        def wrapper(*args, **kwargs):
            step_count = kwargs.pop("step_count", 1)
            max_step_count = kwargs.pop("max_step_count", 1)
            logging.info(f"Created step {step_count}/{max_step_count}: {step_kind}")
            #if isinstance(result, collections.abc.Iterable) and not isinstance(result, str | bytes | dict):
            #    return _LoggedIterable(result, step_name)
            logging.info(f"Start step {step_count}/{max_step_count}: {step_kind}")
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            logging.info(f"Done step {step_count}/{max_step_count}: {step_kind} in {time.perf_counter() - start:.3f}s")
            return result

        wrapper._pipeline_step = True
        wrapper._step_kind = step_kind
        wrapper._step_context = step_context

        return wrapper
    return decorator

@dataclass
class Context:
    pass

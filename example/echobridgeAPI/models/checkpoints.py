import sys
import torch
from typing import Callable, Dict, Optional


_model_functions: Dict[str, Callable] = {}  # mapping of model names to entrypoint fns


def register_model(fn: Callable) -> Callable:
    mod = sys.modules[fn.__module__]
    model_name = fn.__name__
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]
    _model_functions[model_name] = fn
    return fn


def _has_meta_params(mdl: torch.nn.Module) -> bool:
    for p in mdl.parameters(recurse=True):
        if getattr(p, "is_meta", False):
            return True
    for b in mdl.buffers(recurse=True):
        if getattr(b, "is_meta", False):
            return True
    return False


def _strip_module_prefix(state_dict: dict) -> dict:
    # common case: checkpoints saved from DDP have "module." prefix
    if not state_dict:
        return state_dict
    k0 = next(iter(state_dict.keys()))
    if k0.startswith("module."):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def build_mdl(
    model_fn,
    pretrained: bool = False,
    pretrained_url: Optional[str] = None,
    **model_kwargs,
):
    mdl = model_fn(**model_kwargs)

    if pretrained and pretrained_url is not None:
        if "http" in pretrained_url:
            dump = torch.hub.load_state_dict_from_url(pretrained_url, map_location="cpu")
        else:
            dump = torch.load(pretrained_url, map_location="cpu")

        if isinstance(dump, dict) and "model" in dump:
            dump = dump["model"]
        if not isinstance(dump, dict):
            raise ValueError("Checkpoint format not recognized: expected a state_dict-like dict.")

        dump = _strip_module_prefix(dump)

        # ---- Key fix for your warnings ----
        # If current model weights are on meta device, normal in-place copy is a no-op.
        # We materialize tensors first, then load with assign=True when available.
        if _has_meta_params(mdl):
            # PyTorch >= 2.0: allocate empty real tensors on CPU
            try:
                mdl.to_empty(device="cpu")  # type: ignore[attr-defined]
            except Exception:
                # As a fallback, we still try assign=True below (works on newer torch),
                # otherwise you'll still see warnings on old torch.
                pass

        # Load state dict, prefer assign=True to avoid meta no-op and warnings.
        try:
            mdl.load_state_dict(dump, strict=False, assign=True)
        except TypeError:
            # Older PyTorch doesn't support assign=
            mdl.load_state_dict(dump, strict=False)

    return mdl


def list_models():
    return _model_functions.keys()

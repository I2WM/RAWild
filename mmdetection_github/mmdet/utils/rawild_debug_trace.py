from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


_FIRSTBAD_EMITTED = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {'', '0', 'false', 'no', 'off'}


def trace_enabled() -> bool:
    return bool(os.environ.get('RAWILD_RPD_FIRSTBAD_DIR')) or bool(
        os.environ.get('RAWILD_RPD_DEBUG_DUMP')) or _env_flag(
            'RAWILD_RPD_STOP_ON_FIRSTBAD', False)


def stop_on_firstbad() -> bool:
    return _env_flag('RAWILD_RPD_STOP_ON_FIRSTBAD', False)


def _default_dump_dir() -> Path:
    legacy_dump = os.environ.get('RAWILD_RPD_DEBUG_DUMP')
    if legacy_dump:
        return Path(legacy_dump).expanduser().resolve().parent
    return Path(tempfile.gettempdir()) / 'rawild_rpd_firstbad'


def get_dump_dir() -> Path:
    root = os.environ.get('RAWILD_RPD_FIRSTBAD_DIR')
    dump_dir = Path(root).expanduser().resolve() if root else _default_dump_dir()
    dump_dir.mkdir(parents=True, exist_ok=True)
    return dump_dir


def _safe_name(raw: str) -> str:
    out = []
    for ch in str(raw):
        out.append(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_')
    return ''.join(out)[:120]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return scalar_stats(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def scalar_stats(value: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        arr = value.detach().float().cpu()
        finite = torch.isfinite(arr)
        out = {
            'type': 'torch.Tensor',
            'shape': list(arr.shape),
            'nan_count': int(torch.isnan(arr).sum().item()),
            'inf_count': int(torch.isinf(arr).sum().item()),
            'finite_ratio': float(finite.float().mean().item())
            if arr.numel() > 0 else 1.0,
        }
        if finite.any():
            vals = arr[finite]
            out.update({
                'min': float(vals.min().item()),
                'max': float(vals.max().item()),
                'mean': float(vals.mean().item()),
                'abs_max': float(vals.abs().max().item()),
            })
        return out

    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float32)
        finite = np.isfinite(arr)
        out = {
            'type': 'numpy.ndarray',
            'shape': list(arr.shape),
            'nan_count': int(np.isnan(arr).sum()),
            'inf_count': int(np.isinf(arr).sum()),
            'finite_ratio': float(finite.mean()) if arr.size > 0 else 1.0,
        }
        if finite.any():
            vals = arr[finite]
            out.update({
                'min': float(vals.min()),
                'max': float(vals.max()),
                'mean': float(vals.mean()),
                'abs_max': float(np.abs(vals).max()),
            })
        return out

    if isinstance(value, (list, tuple)):
        return {
            'type': type(value).__name__,
            'items': [scalar_stats(v) for v in value],
        }

    if isinstance(value, dict):
        return {
            'type': 'dict',
            'items': {str(k): scalar_stats(v) for k, v in value.items()},
        }

    return {'type': type(value).__name__, 'value': _json_safe(value)}


def _find_first_nonfinite(name: str, value: Any):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            return name, scalar_stats(value)
        return None

    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            return name, scalar_stats(value)
        return None

    if isinstance(value, dict):
        for key, item in value.items():
            found = _find_first_nonfinite(f'{name}.{key}', item)
            if found is not None:
                return found
        return None

    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            found = _find_first_nonfinite(f'{name}[{idx}]', item)
            if found is not None:
                return found
        return None

    return None


def _write_event(kind: str, stage: str, payload: dict[str, Any]) -> Path:
    global _FIRSTBAD_EMITTED
    if kind == 'firstbad' and _FIRSTBAD_EMITTED:
        return get_dump_dir() / 'firstbad_already_emitted.json'

    now = time.strftime('%Y%m%d_%H%M%S')
    suffix = f'{now}_{socket.gethostname()}_pid{os.getpid()}_{_safe_name(kind)}_{_safe_name(stage)}.json'
    out_path = get_dump_dir() / suffix

    event = {
        'kind': kind,
        'stage': stage,
        'hostname': socket.gethostname(),
        'pid': os.getpid(),
        'local_rank': os.environ.get('LOCAL_RANK'),
        'rank': os.environ.get('RANK'),
        'world_size': os.environ.get('WORLD_SIZE'),
        'timestamp': now,
        **payload,
    }
    out_path.write_text(json.dumps(event, indent=2, ensure_ascii=False))
    if kind == 'firstbad':
        _FIRSTBAD_EMITTED = True
    return out_path


def maybe_dump_nonfinite(stage: str,
                         named_tensors: dict[str, Any],
                         extra: dict[str, Any] | None = None,
                         raise_on_nonfinite: bool | None = None) -> bool:
    if not trace_enabled():
        return False

    found = None
    for name, value in named_tensors.items():
        found = _find_first_nonfinite(name, value)
        if found is not None:
            break

    if found is None:
        return False

    bad_name, bad_stats = found
    payload = {
        'first_bad_name': bad_name,
        'first_bad_stats': bad_stats,
        'named_tensor_stats': {k: scalar_stats(v) for k, v in named_tensors.items()},
        'extra': _json_safe(extra or {}),
        'stack': traceback.format_stack(limit=16),
    }
    out_path = _write_event('firstbad', stage, payload)
    if raise_on_nonfinite or (raise_on_nonfinite is None and stop_on_firstbad()):
        raise FloatingPointError(
            f'Non-finite tensor detected at stage={stage} name={bad_name}. '
            f'Debug dump written to {out_path}')
    return True


def write_snapshot(stage: str,
                   named_tensors: dict[str, Any],
                   extra: dict[str, Any] | None = None) -> Path | None:
    if not trace_enabled():
        return None
    payload = {
        'named_tensor_stats': {k: scalar_stats(v) for k, v in named_tensors.items()},
        'extra': _json_safe(extra or {}),
    }
    return _write_event('snapshot', stage, payload)


def register_gradient_hooks(stage: str,
                            named_tensors: dict[str, Any],
                            extra: dict[str, Any] | None = None) -> None:
    if not trace_enabled():
        return

    safe_extra = _json_safe(extra or {})

    def _register(name: str, value: Any) -> None:
        if torch.is_tensor(value) and value.requires_grad:
            def _hook(grad):
                maybe_dump_nonfinite(
                    f'backward.{stage}.{name}',
                    {f'{name}.grad': grad},
                    extra=safe_extra,
                    raise_on_nonfinite=None)
                return grad

            value.register_hook(_hook)
            return

        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                _register(f'{name}.{sub_key}', sub_val)
            return

        if isinstance(value, (list, tuple)):
            for idx, sub_val in enumerate(value):
                _register(f'{name}[{idx}]', sub_val)

    for key, tensor in named_tensors.items():
        _register(key, tensor)


def summarize_img_metas(img_metas: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if img_metas is None:
        return []
    keep_keys = (
        'img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape',
        'scale_factor', 'flip', 'flip_direction')
    summary = []
    for meta in img_metas:
        row = {}
        for key in keep_keys:
            if key in meta:
                row[key] = _json_safe(meta[key])
        summary.append(row)
    return summary

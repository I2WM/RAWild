"""Minimal compatibility shim for legacy Swin/MMDet2 codepaths."""
from mmengine.model import is_model_wrapper
from mmengine.runner.checkpoint import load_checkpoint as _mmengine_load_checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(mod, prefix=''):
        if is_model_wrapper(mod):
            mod = mod.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        mod._load_from_state_dict(
            state_dict, prefix, local_metadata, True,
            all_missing_keys, unexpected_keys, err_msg)
        for name, child in mod._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    missing_keys = [k for k in all_missing_keys if 'num_batches_tracked' not in k]
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: ' + ', '.join(unexpected_keys) + '\n')
    if missing_keys:
        err_msg.append('missing keys in source state_dict: ' + ', '.join(missing_keys) + '\n')
    if err_msg:
        msg = 'The model and loaded state dict do not match exactly\n' + '\n'.join(err_msg)
        if strict:
            raise RuntimeError(msg)
        if logger is not None:
            logger.warning(msg)
        else:
            print(msg)


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None, revise_keys=[(r'^module\.', '')]):
    return _mmengine_load_checkpoint(
        model,
        filename,
        map_location=map_location,
        strict=strict,
        logger=logger,
        revise_keys=revise_keys)

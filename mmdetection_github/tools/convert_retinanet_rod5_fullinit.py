"""Build retinanet_r50_fpn_1x_coco_rod5_fullinit.pth, the ResNet-50 ROD init.

Copies the official COCO RetinaNet weights unchanged and rebuilds retina_cls
for the 5 ROD classes; each class averages its mapped COCO classes. No ROD
data is used.

Source SHA256: c2398f9ec0843ed9a29e72d7a788741abbf2b4250e64f8f793092eaecaa0ab4f
Output tensor SHA256 (names, dtypes, shapes, values; torch-version independent):
80811c4ee9a818daad86b2ee6feb7b4e1db78dff10fcd0d8bdd594acff27720c
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch

SOURCE_URL = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
SOURCE_SHA256 = 'c2398f9ec0843ed9a29e72d7a788741abbf2b4250e64f8f793092eaecaa0ab4f'
TENSOR_SHA256 = '80811c4ee9a818daad86b2ee6feb7b4e1db78dff10fcd0d8bdd594acff27720c'
NUM_COCO_CLASSES = 80
COCO_INDEX = {'person': 0, 'bicycle': 1, 'car': 2, 'bus': 5, 'train': 6, 'truck': 7}
# Same order as ROD_CLASSES in config/RAWild_resnet50/base.py.
ROD_MAPPING = (
    ('Tram', ('train', 'bus')),
    ('Car', ('car',)),
    ('Truck', ('truck',)),
    ('Cyclist', ('bicycle', 'person')),
    ('Pedestrian', ('person',)),
)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def tensor_sha256(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(f'{k}|{v.dtype}|{tuple(v.shape)}'.encode())
        h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()

def rebuild_retina_cls(weight, bias):
    anchors = weight.shape[0] // NUM_COCO_CLASSES
    weight = weight.view(anchors, NUM_COCO_CLASSES, *weight.shape[1:])
    bias = bias.view(anchors, NUM_COCO_CLASSES)
    ids = [[COCO_INDEX[name] for name in names] for _, names in ROD_MAPPING]
    new_weight = torch.stack([weight[:, i].mean(dim=1) for i in ids], dim=1)
    new_bias = torch.stack([bias[:, i].mean(dim=1) for i in ids], dim=1)
    return new_weight.reshape(-1, *weight.shape[2:]), new_bias.reshape(-1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get(
        'RAWILD_PRETRAINED_ROOT', 'checkpoints/pretrained')))
    args = parser.parse_args()
    out = args.output_dir / 'retinanet_r50_fpn_1x_coco_rod5_fullinit.pth'
    if out.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(out))
    actual = sha256(args.source)
    if actual != SOURCE_SHA256:
        raise ValueError('Unexpected source SHA256: ' + actual)
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=True)
    state = {k: v.detach().cpu().clone() for k, v in checkpoint['state_dict'].items()}
    # retina_cls goes last, matching the key order of the checkpoint used in training.
    state['bbox_head.retina_cls.weight'], state['bbox_head.retina_cls.bias'] = rebuild_retina_cls(
        state.pop('bbox_head.retina_cls.weight'), state.pop('bbox_head.retina_cls.bias'))
    digest = tensor_sha256(state)
    if digest != TENSOR_SHA256:
        raise RuntimeError('Unexpected output tensor SHA256: ' + digest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Export parameters only; omit training metadata and local paths.
    with out.open('wb') as f:
        torch.save({'state_dict': state}, f)
    loaded = torch.load(out, map_location='cpu', weights_only=True)['state_dict']
    if list(loaded) != list(state) or tensor_sha256(loaded) != digest:
        raise RuntimeError('Round-trip tensor verification failed.')
    print(json.dumps({'output': str(out), 'torch_version': torch.__version__, 'tensor_count': len(state), 'tensor_sha256': digest, 'sha256': sha256(out)}, indent=2))

if __name__ == '__main__':
    main()

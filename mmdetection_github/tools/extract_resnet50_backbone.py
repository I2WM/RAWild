"""Extract the official COCO RetinaNet backbone; see README Tools for usage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch

SOURCE_URL = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
SOURCE_SHA256 = 'c2398f9ec0843ed9a29e72d7a788741abbf2b4250e64f8f793092eaecaa0ab4f'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get(
        'RAWILD_PRETRAINED_ROOT', 'checkpoints/pretrained')))
    args = parser.parse_args()
    out = args.output_dir / 'resnet50_backbone.pth'
    if out.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(out))
    actual = sha256(args.source)
    if actual != SOURCE_SHA256:
        raise ValueError('Unexpected source SHA256: ' + actual)
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=True)
    backbone = {k[len('backbone.'):]: v.detach().cpu().clone() for k, v in checkpoint['state_dict'].items() if k.startswith('backbone.')}
    if len(backbone) != 318:
        raise ValueError('Expected exactly 318 backbone tensors.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / 'resnet50_backbone.pth'
    if out.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(out))
    # Export parameters only; omit training metadata and local paths.
    with out.open('wb') as f:
        torch.save({'state_dict': backbone}, f)
    loaded = torch.load(out, map_location='cpu', weights_only=True)['state_dict']
    if list(loaded) != list(backbone) or not all(
            torch.equal(loaded[k], backbone[k]) for k in backbone):
        raise RuntimeError('Backbone round-trip tensor verification failed.')
    digest = sha256(out)
    print(json.dumps({'output': str(out), 'torch_version': torch.__version__, 'tensor_count': len(backbone), 'sha256': digest}, indent=2))

if __name__ == '__main__':
    main()

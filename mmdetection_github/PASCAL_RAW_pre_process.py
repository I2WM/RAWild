import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import rawpy
import torchvision


def random_noise_levels():
    """Generates random shot and read noise from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    # print('shot noise and read noise:', log_shot_noise, log_read_noise)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise

def low_light_trans(raw):
    lower, upper = 0.05, 0.4    # low-light range
    exposure_value = random.uniform(lower, upper) 
    raw_low_light = raw * exposure_value

    shot_noise, read_noise = random_noise_levels()
    var = raw_low_light * shot_noise + read_noise
    var = torch.max(var, torch.FloatTensor([1e-5]))
    noise = torch.normal(mean=0, std=torch.sqrt(var))
    raw_low_light = raw_low_light + noise
    return raw_low_light

def oe_light_trans(raw):
    lower, upper = 3.5, 5.0     # over-exp range
    exposure_value = random.uniform(lower, upper) 
    raw_over_exp = raw * exposure_value

    shot_noise, read_noise = random_noise_levels()
    var = raw_over_exp * shot_noise + read_noise
    var = torch.max(var, torch.FloatTensor([1e-5]))
    noise = torch.normal(mean=0, std=torch.sqrt(var))
    raw_over_exp = raw_over_exp + noise
    return raw_over_exp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL RAW .nef files to demosaiced normal, low-light, and over-exposure PNG files.'
    )
    parser.add_argument('--raw-root', type=Path, required=True, help='Directory containing original .nef files.')
    parser.add_argument(
        '--out-root',
        type=Path,
        default=Path('data/PASCAL_RAW/original'),
        help='Base output directory used when explicit output dirs are not set.',
    )
    parser.add_argument('--normal-dir', type=Path, default=None, help='Output directory for normal-light PNG files.')
    parser.add_argument('--low-dir', type=Path, default=None, help='Output directory for low-light PNG files.')
    parser.add_argument('--oe-dir', type=Path, default=None, help='Output directory for over-exposure PNG files.')
    parser.add_argument('--width', type=int, default=600, help='Output image width.')
    parser.add_argument('--height', type=int, default=400, help='Output image height.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    raw_root = args.raw_root.expanduser()
    out_path = args.normal_dir or (args.out_root / 'demosaic')
    out_path_low = args.low_dir or (args.out_root / 'demosaic_low')
    out_path_oe = args.oe_dir or (args.out_root / 'demosaic_oe')

    if not raw_root.exists():
        raise FileNotFoundError(f'Missing RAW input directory: {raw_root}')

    out_path.mkdir(parents=True, exist_ok=True)
    out_path_low.mkdir(parents=True, exist_ok=True)
    out_path_oe.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(sorted(raw_root.iterdir())):
        if image_path.suffix.lower() not in {'.nef'}:
            continue
        
        image_name = str(image_path)

        # Setp 0: Load RAW data
        raw = rawpy.imread(image_name)
        mosaic = raw.raw_image
        black = mosaic.min()
        saturation = mosaic.max()

        # Step 1: Linearization
        uint12_max = 2**12 - 1
        mosaic -= black  # black subtraction
        mosaic *= int(uint12_max/(saturation - black))
        mosaic = np.clip(mosaic, 0, uint12_max)  # clip to range
        
        mosaic = np.float64(mosaic) 

        # Step 2: Demosacing, RGGB --> RGB
        def demosaic(m):    
            r = m[0::2, 0::2]
            g = np.clip(m[0::2, 1::2]//2 + m[1::2, 0::2]//2,
                        0, 2 ** 12 - 1)
            b = m[1::2, 1::2]
            return np.dstack([r, g, b])

        raw_rgb = demosaic(mosaic) 

        # Step 3: Resize to match PASCAL RAW detection labels.
        raw_rgb_resize = cv2.resize(raw_rgb, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
        raw_rgb_resize = np.clip(raw_rgb_resize, 0, 2**12-1)

        raw_rgb_resize = raw_rgb_resize/(2**12-1)

        ## Normal-Light Scene 
        png_name = f'{image_path.stem}.png'
        plt.imsave(out_path / png_name, np.float64(np.clip(raw_rgb_resize, 0, 1)))
        raw_rgb_resize_t = torch.from_numpy(raw_rgb_resize).float().permute(2,0,1).unsqueeze(0)
        
        ## Low-Light Scene 
        raw_rgb_low = low_light_trans(raw_rgb_resize_t)
        torchvision.utils.save_image(raw_rgb_low, out_path_low / png_name)

        ## Over-Exposure Scene 
        raw_rgb_oe = oe_light_trans(raw_rgb_resize_t)
        torchvision.utils.save_image(raw_rgb_oe, out_path_oe / png_name)
        

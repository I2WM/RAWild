#!/usr/bin/env python3
# Copyright (c) 2025 SonyResearch
# SPDX-License-Identifier: MIT
# See LICENSE.SonyResearch in this directory.

import argparse
from pathlib import Path

import numpy as np
import rawpy
from tqdm import tqdm


def bayer2rggb(src_path):
    with rawpy.imread(str(src_path)) as raw_file:
        bayer = raw_file.raw_image
        return np.stack((
            bayer[0::2, 0::2],
            bayer[0::2, 1::2],
            bayer[1::2, 0::2],
            bayer[1::2, 1::2],
        ), axis=-1)


def process_folder(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    raw_files = sorted(
        path for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() == '.nef')
    if not raw_files:
        raise FileNotFoundError(f'No Nikon NEF files found under {input_folder}')
    output_folder.mkdir(parents=True, exist_ok=True)
    for raw_file in tqdm(raw_files):
        rggb = bayer2rggb(raw_file)
        np.save(output_folder / raw_file.with_suffix('.npy').name,
                rggb, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(
        description='Pack NOD Nikon NEF files into unnormalized four-channel RGGB NPY arrays.')
    parser.add_argument('--input-folder', '--input_folder', dest='input_folder',
                        type=Path, required=True,
                        help='Directory containing Nikon .NEF files.')
    parser.add_argument('--output-folder', '--output_folder', dest='output_folder',
                        type=Path, required=True,
                        help='Directory for packed .npy files.')
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()

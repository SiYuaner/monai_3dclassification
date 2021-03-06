#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tif_to_nii
command line executable to convert a directory of tif images
(from one image) to a nifti image stacked along a user-specified axis
call as: python tif_to_nii.py /path/to/tif/ /path/to/nifti
(append optional arguments to the call as desired)
Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
"""

import argparse
from glob import glob
import os
from pathlib import Path
import sys
import tifffile
from PIL import Image
import nibabel as nib
import numpy as np


def arg_parser():
    parser = argparse.ArgumentParser(description='merge 2d tif images into a 3d image')
    parser.add_argument('img_dir', type=str,
                        help='path to tiff image directory')
    parser.add_argument('out_dir', type=str,
                        help='path to output the corresponding tif image slices')
    parser.add_argument('-a', '--axis', type=int, default=2,
                        help='axis on which to stack the 2d images')
    return parser


def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def main():
    try:
        args = arg_parser().parse_args()
        img_dir = Path(args.img_dir)
        fns = sorted([str(fn) for fn in img_dir.glob('*.tif*')])
        if not fns:
            raise ValueError(f'img_dir ({args.img_dir}) does not contain any .tif or .tiff images.')
        imgs = []
        for fn in fns:
            _, base, ext = split_filename(fn)
            img = np.asarray(Image.open(fn)).astype(np.float32).squeeze()
            if img.ndim != 2:
                raise Exception(f'Only 2D data supported. File {base}{ext} has dimension {img.ndim}.')
            imgs.append(img)
        img = np.stack(imgs, axis=args.axis)
        nib.Nifti1Image(img,None).to_filename(os.path.join(args.out_dir, f'{base}.nii.gz'))
        return 0
    except Exception as e:
        print(e)
        return 1

def main1(input_path, output_path):
    img = tifffile.imread(input_path)
    img = np.asarray(img).astype(np.float32)
    nib.Nifti1Image(img, None).to_filename(output_path)


if __name__ == "__main__":
    input_path = r'H:\???????????????????????????\??????????????????\graph\tutorials\3d_classification\datasets\dataset40\val'
    for dir in os.listdir(input_path):
        dir_path = os.path.join(input_path, dir)
        for file in os.listdir(dir_path):
            if file[-8:] == '3dim.tif':
                file_path = os.path.join(dir_path, file)
                output_path = os.path.join(dir_path, file[:-4] + '.nii.gz')
                print(file_path, '\n', output_path)
                main1(file_path, output_path)

    # img=nib.load(output_path)
    # img_arr=img.get_fdata()
    # print(img_arr.shape)
    #
    # nib.Nifti1Image(img_arr[0], None).to_filename(r'C:\Users\Administrator\Desktop\0.nii')
    # nib.Nifti1Image(img_arr[1], None).to_filename(r'C:\Users\Administrator\Desktop\1.nii')
    # nib.Nifti1Image(img_arr[2], None).to_filename(r'C:\Users\Administrator\Desktop\2.nii')
    #
    #
    # # io.imshow(img_arr)
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# A dummy .wav file generator
# Author: Wil Kong
# Date: 12/18/2017, Mon

import argparse
import os
import scipy.io.wavfile
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Dummy .wav generator.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seconds', type=float, default=4,
            help='The length of generated .wav file')
    parser.add_argument('--rate', type=int, default=16000,
            help='Sample rate.')
    parser.add_argument('--count', type=int, default=200,
            help='Number of .wav files to generate.')
    parser.add_argument('--path', type=str, default='data/dummy',
            help='Path to store the generated .wav files.')

    return parser.parse_args()

def generate_wavs(args):
    # Create the directory if doesn't exist
    rootpath = os.path.abspath(args.path)
    wvpath = os.sep.join((rootpath, 'wav'))
    txpath = os.sep.join((rootpath, 'txt'))
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
        os.makedirs(wvpath)
        os.makedirs(txpath)

    # open manifest file
    manifest = os.sep.join((rootpath, 'dummy.csv'))
    df = pd.DataFrame(columns=['wav', 'txt'])

    for i in range(args.count):
        # Setup file name
        fname = '{:0>8d}'.format(i)
        wvname = os.sep.join((wvpath, fname + '.wav'))
        txname = os.sep.join((txpath, fname + '.txt'))

        # Generate random noise data for .wav file
        length = int(args.rate * args.seconds)
        data = np.random.randint(-32768, 32768, length, dtype=np.int16)

        # Write frames
        scipy.io.wavfile.write(wvname, args.rate, data)

        # Generate empty label file
        # Please notice that the acceptable label for deep speech 2 is 29 chars(capital).
        tx = open(txname, 'w')
        tx.write('TEST')
        tx.close()

        # Write corresponding manifest for deep speech 2
        df.loc[i] = [wvname, txname]

    df.to_csv(manifest, sep=',', header=False, index=False)

if __name__ == '__main__':
    args = parse_args()
    generate_wavs(args)


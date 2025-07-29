import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp',
                        '--dataset-path',
                        default='data',
                        type=str,
                        help='Path to the dataset')
    parser.add_argument('-fw',
                        '--frameworks',
                        nargs='+',
                        default=[
                            'cgmot_online',
                            'cgmot_global',
                            'mctrack_online',
                            'mctrack_global',
                        ],
                        type=str)
    parser.add_argument('-det',
                        '--detectors',
                        nargs='+',
                        default=['pointpillar', 'pv_rcnn'],
                        type=str)
    parser.add_argument('-tp',
                        '--tracking-path',
                        default='tracking/output',
                        type=str,
                        help='Path to the trackers result')
    parser.add_argument('-tpp',
                        '--test-path',
                        default='test_prioritization/output',
                        type=str,
                        help='Path to the test prioritization result')
    parser.add_argument('-s',
                        '--split',
                        default='val',
                        type=str,
                        choices=['val', 'test'])
    parser.add_argument('-c',
                        '--criteria',
                        nargs='+',
                        default=['sgmtp', 'random'],
                        type=str)

    args = parser.parse_args()

    print('\n#')
    for k, v in args._get_kwargs():
        print('# ' + k.upper().replace('_', ' ') + ' : ', v)
    print('#')

    return args


def get_prioritization(args, fw, det):
    RQ1_path = f'{args.test_path}/{args.split}'

    data = pd.read_csv(f'{RQ1_path}/{fw}_{det}_result.csv')
    x = data.to_numpy()

    cut = np.ceil(x.shape[0] * 0.30).astype(np.int32)

    index = x[:, 0] / (x.sum(axis=1).reshape(-1))
    index = index.argsort()[:cut]

    os.makedirs(f'{RQ1_path}/sgmtp/{fw}_{det}', exist_ok=True)

    label_02_path = f'{args.dataset_path}/{args.split}/label_02'
    new_label_02_path = f'{RQ1_path}/sgmtp/{fw}_{det}/label_02'

    if os.path.islink(new_label_02_path):
        os.remove(new_label_02_path)

    os.symlink(
        Path(label_02_path).resolve().as_posix(),
        new_label_02_path,
    )

    pd.DataFrame(
        x[index].sum(axis=0).reshape(1, -1),
        columns=data.columns,
    ).to_csv(f'{RQ1_path}/sgmtp/{fw}_{det}/count.csv', index=None)

    pd.read_csv(
        f'{args.dataset_path}/{args.split}/evaluate_tracking.seqmap.{args.split}',
        sep=' ',
        header=None,
        dtype=str,
    ).loc[index].to_csv(
        f'{RQ1_path}/sgmtp/{fw}_{det}/evaluate_tracking.seqmap.{args.split}',
        sep=' ',
        header=None,
        index=None,
    )


def get_baseline(args, fw, det):
    RQ1_path = f'{args.test_path}/{args.split}'

    data = pd.read_csv(f'{RQ1_path}/{fw}_{det}_result.csv')
    x = data.to_numpy()

    cut = np.ceil(x.shape[0] * 0.30).astype(np.int32)

    np.random.seed(123456)
    index = np.random.choice(x.shape[0], cut)

    os.makedirs(f'{RQ1_path}/random/{fw}_{det}', exist_ok=True)

    label_02_path = f'{args.dataset_path}/{args.split}/label_02'
    new_label_02_path = f'{RQ1_path}/random/{fw}_{det}/label_02'

    if os.path.islink(new_label_02_path):
        os.remove(new_label_02_path)

    os.symlink(
        Path(label_02_path).resolve().as_posix(),
        new_label_02_path,
    )

    pd.DataFrame(
        x[index].sum(axis=0).reshape(1, -1),
        columns=data.columns,
    ).to_csv(f'{RQ1_path}/random/{fw}_{det}/count.csv', index=None)

    pd.read_csv(
        f'{args.dataset_path}/{args.split}/evaluate_tracking.seqmap.{args.split}',
        sep=' ',
        header=None,
        dtype=str,
    ).loc[index].to_csv(
        f'{RQ1_path}/random/{fw}_{det}/evaluate_tracking.seqmap.{args.split}',
        sep=' ',
        header=None,
        index=None,
    )


def do_RQ1(frameworks, detectors):

    for fw in frameworks:
        for det in detectors:
            get_prioritization(args, fw, det)
            get_baseline(args, fw, det)


if __name__ == '__main__':

    args = get_args()

    do_RQ1(args.frameworks, args.detectors)
    eval(args.frameworks, args.detectors, args.tracking_path, args.test_path,
         args.split, args.criteria)

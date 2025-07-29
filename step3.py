import argparse

from test_prioritization import *


def get_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('-dp',
                        '--dataset-path',
                        default='data',
                        type=str,
                        help='Path to the dataset')
    parser.add_argument('-tp',
                        '--tracking-path',
                        default='tracking/output',
                        type=str,
                        help='Path to the trackers result')
    parser.add_argument('-s',
                        '--split',
                        default='val',
                        type=str,
                        choices=['val', 'test'])
    parser.add_argument('-o',
                        '--output-path',
                        default='test_prioritization/output',
                        type=str)

    args = parser.parse_args()

    print('\n#')
    for k, v in args._get_kwargs():
        print('# ' + k.upper().replace('_', ' ') + ' : ', v)
    print('#')

    return args


if __name__ == '__main__':

    args = get_args()

    test(args.frameworks, args.detectors, args.dataset_path,
         args.tracking_path, args.split, args.output_path)

import argparse

from tracking import *


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
    parser.add_argument('-detp',
                        '--detections-path',
                        default='detection/output',
                        type=str,
                        help='Path to the detections result')
    parser.add_argument('-s',
                        '--split',
                        default='val',
                        type=str,
                        choices=['val', 'test'])
    parser.add_argument('-o',
                        '--output-path',
                        default='tracking/output',
                        type=str)

    args = parser.parse_args()

    print('\n#')
    for k, v in args._get_kwargs():
        print('# ' + k.upper().replace('_', ' ') + ' : ', v)
    print('#')

    return args


if __name__ == '__main__':
    args = get_args()

    track(args.frameworks, args.detectors, args.dataset_path,
          args.detections_path, args.split, args.output_path)

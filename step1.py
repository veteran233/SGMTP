import argparse

from detection import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset-path', default='data', type=str, help='Path to the dataset')
    parser.add_argument('-s',
                        '--split',
                        default='val',
                        type=str,
                        choices=['val', 'test'])
    parser.add_argument('-m',
                        '--models-name',
                        nargs='+',
                        default=['pointpillar', 'pv_rcnn'],
                        type=str)
    parser.add_argument('-o',
                        '--output-path',
                        default='detection/output',
                        type=str)

    args = parser.parse_args()

    print('\n#')
    for k, v in args._get_kwargs():
        print('# ' + k.upper().replace('_', ' ') + ' : ', v)
    print('#')

    return args


if __name__ == '__main__':

    args = get_args()

    detect(args.dataset_path, args.split, args.models_name, args.output_path)

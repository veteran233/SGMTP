import os
import shutil
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


class generate_decorator:

    def __init__(self, obj):
        self.cur_scene = 0
        self.evaluate_tracking_seqmap = ''
        self.o = obj

    def __call__(self, dataset_path, split, output_path, second_per_data,
                 stride, scene_exclude):
        '''
        | Parameter         | Type       | Description                         |
        |-------------------|------------|-------------------------------------|
        | `dataset_path`    | `str`      | Path to the origin dataset          |
        | `output_path`     | `str`      | Output path for saving data         |
        | `second_per_data` | `int`      | Time duration (in seconds)          |
        | `stride`          | `int`      | Step size/window shift (in seconds) |
        | `scene_exclude`   | `list[str]`| List of scenes to exclude           |
        '''

        self.fps = 10

        self.dataset_path = dataset_path
        self.split = split
        self.dataset_type = sorted(os.listdir(f'{dataset_path}/{self.sp}'))
        self.output_path = output_path
        self.frame_per_data = self.fps * second_per_data
        stride *= self.fps

        scenes_range = []
        self.scenes_label = {}
        self.scenes_pose = {}
        for scene in sorted(os.listdir(f'{dataset_path}/{self.sp}/velodyne')):
            if scene in scene_exclude: continue

            if self.sp == 'training':
                self.scenes_label[scene] = open(
                    f'{dataset_path}/{self.sp}/label_02/{scene}.txt',
                    'r').readlines()

            self.scenes_pose[scene] = np.array(
                open(f'{dataset_path}/{self.sp}/pose/{scene}/pose.txt',
                     'r').readlines())

            scenes_range.append((
                scene,
                0,
                len(os.listdir(f'{dataset_path}/{self.sp}/velodyne/{scene}')),
            ))

        runner = []
        for _d in self.dataset_type:
            try:
                runner.append(getattr(self, f'get_{_d}'))
            except Exception as e:
                ...
        runner.append(self.to_next_scene)

        self.o(self.frame_per_data, stride, scenes_range, runner)
        self.save_data_split()

    @property
    def sp(self):
        return 'training' if self.split == 'val' else 'testing'

    def get_calib(self, scene, idx):
        calib_file = f'{self.dataset_path}/{self.sp}/calib/{scene}.txt'

        # shutil.copyfile(
        #     calib_file,
        #     self.create_dirs('calib') + f'/{self.cur_scene:04d}.txt')

        new_calib_file = self.create_dirs(
            'calib') + f'/{self.cur_scene:04d}.txt'
        if os.path.islink(new_calib_file):
            os.remove(new_calib_file)
        os.symlink(Path(calib_file).resolve().as_posix(), new_calib_file)

    def get_label_02(self, scene, idx):
        frame_id = []
        raw = []
        for label in self.scenes_label[scene]:
            label = label.split(' ')
            frame_id.append(int(label[0]))
            raw.append(label[1:])
        frame_id = np.array(frame_id)
        raw = np.array(raw)

        frame_id_mask = (idx <= frame_id) * (frame_id
                                             < idx + self.frame_per_data)
        frame_id = frame_id[frame_id_mask]
        frame_id -= frame_id.min()
        raw = raw[frame_id_mask]

        raw = np.hstack((frame_id.reshape(-1, 1), raw))
        label = []
        for line in raw:
            label.append(' '.join(line))

        open(self.create_dirs('label_02') + f'/{self.cur_scene:04d}.txt',
             'w').writelines(label)

    def get_pose(self, scene, idx):
        pose_file = self.create_dirs('pose',
                                     f'{self.cur_scene:04d}') + '/pose.txt'
        open(pose_file, 'w').writelines(self.scenes_pose[scene][np.arange(
            idx,
            idx + self.frame_per_data,
            dtype=np.int32,
        )])

        new_pose_file = self.create_dirs('pose') + f'/{self.cur_scene:04d}.txt'
        if os.path.islink(new_pose_file):
            os.remove(new_pose_file)
        os.symlink(Path(pose_file).resolve().as_posix(), new_pose_file)

    def get_sg(self, scene, idx):
        sg_file = f'{self.dataset_path}/{self.sp}/sg/{scene}.pkl'

        # shutil.copyfile(sg_file,
        #                 self.create_dirs('sg') + f'/{self.cur_scene:04d}.pkl')

        new_sg_file = self.create_dirs('sg') + f'/{self.cur_scene:04d}.pkl'
        if os.path.islink(new_sg_file):
            os.remove(new_sg_file)
        os.symlink(Path(sg_file).resolve().as_posix(), new_sg_file)

    def get_velodyne(self, scene, idx):
        for new_i, i in enumerate(range(idx, idx + self.frame_per_data)):
            lidar_file = f'{self.dataset_path}/{self.sp}/velodyne/{scene}/{i:06d}.bin'

            # shutil.copyfile(
            #     lidar_file,
            #     self.create_dirs('velodyne', f'{self.cur_scene:04d}') +
            #     f'/{new_i:06d}.bin')

            new_lidar_file = self.create_dirs(
                'velodyne', f'{self.cur_scene:04d}') + f'/{new_i:06d}.bin'
            if os.path.islink(new_lidar_file):
                os.remove(new_lidar_file)
            os.symlink(Path(lidar_file).resolve().as_posix(), new_lidar_file)

        self.evaluate_tracking_seqmap += f'{self.cur_scene:04d} empty 000000 {self.frame_per_data:06d}\n'

    def to_next_scene(self, *args):
        self.cur_scene += 1

    def create_dirs(self, dataset_type, scene=None):
        if scene is None:
            path = f'{self.output_path}/{self.split}/{dataset_type}'
        else:
            path = f'{self.output_path}/{self.split}/{dataset_type}/{scene}'

        os.makedirs(path, exist_ok=True)

        return path

    def save_data_split(self):
        os.makedirs(f'{self.output_path}/{self.split}', exist_ok=True)
        open(
            f'{self.output_path}/{self.split}/evaluate_tracking.seqmap.{self.split}',
            'w').writelines(self.evaluate_tracking_seqmap)


@generate_decorator
def generate(frame_per_data, stride, scenes_range, runner):
    try:
        scenes_range.remove(('0001', 0, 443))
        scenes_range.append(('0001', 0, 177))
        scenes_range.append(('0001', 181, 447))
    except Exception as e:
        ...
    for scene, start, end in tqdm(scenes_range):
        idx = start
        while idx < end - frame_per_data:
            for run in runner:
                run(scene, idx)
            idx += stride


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-odp',
                        '--origin-dataset-path',
                        type=str,
                        required=True,
                        help='Path to the origin dataset')
    parser.add_argument('-s',
                        '--split',
                        default='val',
                        type=str,
                        choices=['val', 'test'])
    parser.add_argument('-o', '--output-path', default='data', type=str)
    parser.add_argument('--second-per-data',
                        default=10,
                        type=int,
                        help='Time duration (in seconds)')
    parser.add_argument('--stride',
                        default=5,
                        type=int,
                        help='Step size/window shift (in seconds)')
    parser.add_argument('--scene-exclude',
                        nargs='+',
                        default=['0000', '0012', '0016', '0017'],
                        type=str,
                        help='List of scenes to exclude')

    args = parser.parse_args()

    print('\n#')
    for k, v in args._get_kwargs():
        print('# ' + k.upper().replace('_', ' ') + ' : ', v)
    print('#')

    return args


if __name__ == '__main__':

    args = get_args()

    generate(args.origin_dataset_path, args.split, args.output_path,
             args.second_per_data, args.stride, args.scene_exclude)

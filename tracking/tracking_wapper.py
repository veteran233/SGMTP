import os
import sys
import json
import yaml
import shutil
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict
from functools import partial

from .mctrack_utils.mctrack_convert import convert
from .mctrack_utils.mctrack_main import run
from .mctrack_utils.mctrack_save import save_results


class track_decorator:

    def __init__(self, obj):
        self.o = obj

    def __call__(self,
                 frameworks,
                 detectors,
                 dataset_path,
                 detections_path,
                 split,
                 output_path,
                 num_workers=4):
        '''
        | Parameter        | Type       | Description                        |
        |------------------|------------|------------------------------------|
        | `frameworks`     | `list[str]`| Names of the frameworks to be used |
        | `detectors`      | `list[str]`| Names of the detections models     |
        | `dataset_path`   | `str`      | Path to the dataset                |
        | `detections_path`| `str`      | Path to the detection results      |
        '''

        if os.path.isdir('tracking/__mctrack_preprocess_output'):
            shutil.rmtree('tracking/__mctrack_preprocess_output')

        if 'mctrack_global' in frameworks or 'mctrack_online' in frameworks:
            for detector in detectors:
                convert(dataset_path, detections_path, detector,
                        'tracking/__mctrack_preprocess_output', split)

        runner = [
            partial(
                getattr(self, i),
                dataset_path,
                detections_path,
                output_path,
                split=split,
            ) for i in frameworks
        ]

        self.o(runner, detectors, num_workers)

    @staticmethod
    def scenes(detections_path, detector, split):
        return sorted(os.listdir(f'{detections_path}/{detector}/{split}'))

    @staticmethod
    def scenes_range(detections_path, detector, split):
        return range(len(os.listdir(f'{detections_path}/{detector}/{split}')))

    @staticmethod
    def mctrack_global(dataset_path, detections_path, output_path, detector,
                       tqdm_pos, split):
        cfg = yaml.safe_load(
            open('tracking/framework/MCTrack/config/kitti_offline.yaml', 'r'))

        cfg['SPLIT'] = split
        cfg["THRESHOLD"]["GLOBAL_TRACK_SCORE"] = 0.3
        cfg['DETECTOR'] = detector
        cfg['DATASET_ROOT'] = Path(dataset_path).resolve().as_posix()
        cfg['DETECTIONS_ROOT'] = Path(
            'tracking/__mctrack_preprocess_output').resolve().as_posix()
        cfg['SAVE_PATH'] = Path(
            f'{output_path}/{split}/mctrack_global').resolve().as_posix()

        data = json.load(
            open(
                f'tracking/__mctrack_preprocess_output/{detector}/{split}.json',
                'r'))

        tracking_results = dict()
        for scene in tqdm(
                track_decorator.scenes(detections_path, detector, split),
                desc=f'Framework : mctrack_global Detector : {detector}',
                position=tqdm_pos,
        ):
            run(scene, data, cfg, tracking_results)

        save_results(tracking_results, cfg)

    @staticmethod
    def mctrack_online(dataset_path, detections_path, output_path, detector,
                       tqdm_pos, split):
        cfg = yaml.safe_load(
            open('tracking/framework/MCTrack/config/kitti.yaml', 'r'))

        cfg['SPLIT'] = split
        cfg['DETECTOR'] = detector
        cfg['DATASET_ROOT'] = Path(dataset_path).resolve().as_posix()
        cfg['DETECTIONS_ROOT'] = Path(
            'tracking/__mctrack_preprocess_output').resolve().as_posix()
        cfg['SAVE_PATH'] = Path(
            f'{output_path}/{split}/mctrack_online').resolve().as_posix()

        data = json.load(
            open(
                f'tracking/__mctrack_preprocess_output/{detector}/{split}.json',
                'r'))

        tracking_results = dict()
        for scene in tqdm(
                track_decorator.scenes(detections_path, detector, split),
                desc=f'Framework : mctrack_online Detector : {detector}',
                position=tqdm_pos,
        ):
            run(scene, data, cfg, tracking_results)

        save_results(tracking_results, cfg)

    @staticmethod
    def ug3dmot():
        raise NotImplementedError

    @staticmethod
    def cgmot_global(dataset_path, detections_path, output_path, detector,
                     tqdm_pos, split):
        try:
            from kitti_3DMOT import track_one_seq, save_one_seq
        except Exception as e:
            sys.path.insert(0, 'tracking/framework/3D-Multi-Object-Tracker')
            from kitti_3DMOT import track_one_seq, save_one_seq

        cfg = EasyDict(
            yaml.safe_load(
                open(
                    'tracking/framework/3D-Multi-Object-Tracker/config/global/second_iou_mot.yaml',
                    'r')))

        cfg.dataset_path = Path(f'{dataset_path}/{split}').resolve().as_posix()
        cfg.detections_path = Path(
            f'{detections_path}/{detector}/{split}').resolve().as_posix()
        cfg.save_path = Path(
            f'{output_path}/{split}/cgmot_global/{detector}/data').resolve(
            ).as_posix()

        for scene in tqdm(
                track_decorator.scenes_range(detections_path, detector, split),
                desc=f'Framework : cgmot_global Detector : {detector}',
                position=tqdm_pos,
        ):
            dataset, tracker, _, _ = track_one_seq(scene, cfg)
            save_one_seq(dataset, scene, tracker, cfg)

    @staticmethod
    def cgmot_online(dataset_path, detections_path, output_path, detector,
                     tqdm_pos, split):
        try:
            from kitti_3DMOT import track_one_seq, save_one_seq
        except Exception as e:
            sys.path.insert(0, 'tracking/framework/3D-Multi-Object-Tracker')
            from kitti_3DMOT import track_one_seq, save_one_seq

        cfg = EasyDict(
            yaml.safe_load(
                open(
                    'tracking/framework/3D-Multi-Object-Tracker/config/online/second_iou_mot.yaml',
                    'r')))

        cfg.dataset_path = Path(f'{dataset_path}/{split}').resolve().as_posix()
        cfg.detections_path = Path(
            f'{detections_path}/{detector}/{split}').resolve().as_posix()
        cfg.save_path = Path(
            f'{output_path}/{split}/cgmot_online/{detector}/data').resolve(
            ).as_posix()

        for scene in tqdm(
                track_decorator.scenes_range(detections_path, detector, split),
                desc=f'Framework : cgmot_online Detector : {detector}',
                position=tqdm_pos,
        ):
            dataset, tracker, _, _ = track_one_seq(scene, cfg)
            save_one_seq(dataset, scene, tracker, cfg)


__all__ = ['track_decorator']

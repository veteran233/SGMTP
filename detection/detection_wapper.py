import os
from tqdm import tqdm
from pathlib import Path
from functools import partial
from easydict import EasyDict as Dict

from torch import no_grad
from torch import from_numpy
from torch.utils.data import DataLoader

from pcdet.utils import common_utils
from pcdet.models import build_network
from pcdet.config import cfg_from_yaml_file

from dataset_utils.kitti_dataset import KittiTrackingDataset


class detect_decorator():

    def __init__(self, obj):
        self.o = obj
        self.keys = ['points', 'voxels', 'voxel_num_points', 'voxel_coords']

    def __call__(self, dataset_path, split, models_name, output_path):
        self.dataset_path = dataset_path
        self.split = split
        self.models_name = models_name
        self.output_path = output_path

        self.o(
            models_name,
            partial(
                self.infer,
                split=split,
                output_path=output_path,
                load_model=partial(
                    self.load_model,
                    dataset_path=dataset_path,
                    split=split,
                ),
                load_data_to_gpu=partial(
                    self.load_data_to_gpu,
                    keys=self.keys,
                ),
            ),
        )

    @staticmethod
    def infer(model_name, split, output_path, **func):
        func = Dict(func)
        load_model = func.load_model
        load_data_to_gpu = func.load_data_to_gpu

        model, loader = load_model(model_name)

        model.cuda()
        model.eval()
        with no_grad():
            for data_batch in tqdm(loader):
                load_data_to_gpu(data_batch)
                infer_result, _ = model(data_batch)

                loader.dataset.generate_prediction_dicts(
                    data_batch,
                    infer_result,
                    ['Car', 'Pedestrian', 'Cyclist'],
                    ['Car'],
                    f'{output_path}/{model_name}/{split}',
                )

    @staticmethod
    def load_model(model_name, dataset_path, split, batch_size=4):
        assert os.path.exists(f'detection/cfg/{model_name}.yaml')
        assert os.path.exists(f'detection/model/{model_name}.pth')

        cfg = Dict()
        cfg_from_yaml_file(f'detection/cfg/{model_name}.yaml', cfg)

        loader = DataLoader(KittiTrackingDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            dataset_path=Path(dataset_path),
            split=split),
                            batch_size=batch_size,
                            collate_fn=KittiTrackingDataset.collate_batch,
                            pin_memory=True)

        model = build_network(model_cfg=cfg.MODEL,
                              num_class=len(cfg.CLASS_NAMES),
                              dataset=loader.dataset)

        model.load_params_from_file(
            f'detection/model/{model_name}.pth',
            common_utils.create_logger(f'logfile/{model_name}.log'))

        return model, loader

    @staticmethod
    def load_data_to_gpu(batch_dict, keys):
        for k, v in batch_dict.items():
            if k in keys:
                batch_dict[k] = from_numpy(v).cuda()


__all__ = ['detect_decorator']

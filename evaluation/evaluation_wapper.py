from functools import partial

import numpy as np
from easydict import EasyDict as Dict

import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler

matplotlib.use('pdf')
matplotlib.rcParams['figure.figsize'] = 4, 2

matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['tab:red', 'tab:blue'])

matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['font.family'] = 'DeJaVu Serif'
matplotlib.rcParams['legend.loc'] = 'lower left'
matplotlib.rcParams['legend.fontsize'] = 'small'

matplotlib.rcParams['figure.constrained_layout.use'] = True

import trackeval
from trackeval.datasets import Kitti2DBox
from trackeval.metrics import HOTA, CLEAR, Identity

from .evaluation_config import *


class eval_decorator():

    def __init__(self, obj):
        self.o = obj

    def __call__(self, frameworks, detectors, tracking_path, testing_path,
                 split, criteria):

        self.frameworks = frameworks
        self.detectors = detectors
        self.tracking_path = tracking_path
        self.testing_path = testing_path
        self.split = split
        self.criteria = criteria

        array_labels = np.arange(0.05, 0.99, 0.05)
        metrics_list = [HOTA(), CLEAR(), Identity()]

        fig_ax = Dict(
            mctrack=plt.subplots(1, 2, sharey=True),
            cgmot=plt.subplots(1, 2, sharey=True),
        )

        self.o(
            frameworks,
            detectors,
            criteria,
            partial(
                self.evaluate,
                evaluator=trackeval.Evaluator(
                    dict(PRINT_ONLY_COMBINED=True,
                         TIME_PROGRESS=False,
                         PLOT_CURVES=False)),
                array_labels=array_labels,
                metrics_list=metrics_list,
                fig_ax=fig_ax,
                create_dataset_config=partial(
                    self.create_dataset_config,
                    tracking_path=tracking_path,
                    testing_path=testing_path,
                    split=split,
                ),
            ),
        )

        for k, v in fig_ax.items():
            fig, axs = v

            axs[0].set_ylabel('score')
            axs[0].set_xlabel('$\\alpha$')
            axs[1].set_xlabel('$\\alpha$')

            axs[0].legend(markerscale=0)

            fig.savefig(f'evaluation/{k}.pdf', format='pdf')

    @staticmethod
    def evaluate(fw, det, cri, evaluator, array_labels, metrics_list, fig_ax,
                 **func):

        func = Dict(func)
        create_dataset_config = func.create_dataset_config

        if fw.startswith('mctrack'):
            fig, axs = fig_ax['mctrack']
        elif fw.startswith('cgmot'):
            fig, axs = fig_ax['cgmot']
        else:
            raise NotImplementedError

        if det == 'pointpillar':
            ax = axs[0]
        elif det == 'pv_rcnn':
            ax = axs[1]
        else:
            raise NotImplementedError

        print(f'# {fw} {det} {cri}')
        dataset = Kitti2DBox(create_dataset_config(fw, det, cri))

        res, msg = evaluator.evaluate([dataset], metrics_list)

        dataset_name = dataset.get_name()
        detector_name = dataset.get_eval_info()[0][0]

        data = res[dataset_name][detector_name]['COMBINED_SEQ']['car']['HOTA'][
            'HOTA']
        if fw.endswith('online'):
            ax.plot(
                array_labels,
                data,
                '.--' if cri != 'sgmtp' else '.-',
                label=f'{get_format_name(cri)}',
            )
        elif fw.endswith('global'):
            ax.plot(
                array_labels,
                data,
                '^--' if cri != 'sgmtp' else '^-',
            )

    @staticmethod
    def create_dataset_config(fw, det, cri, tracking_path, testing_path,
                              split):

        dataset_config = Dict()

        dataset_config.GT_FOLDER = f'{testing_path}/{split}/{cri}/{fw}_{det}'
        dataset_config.TRACKERS_FOLDER = f'{tracking_path}/{split}/{fw}'
        dataset_config.OUTPUT_FOLDER = dataset_config.GT_FOLDER
        dataset_config.TRACKERS_TO_EVAL = [det]
        dataset_config.CLASSES_TO_EVAL = ['car']
        dataset_config.SPLIT_TO_EVAL = split
        dataset_config.PRINT_CONFIG = False

        return dataset_config


__all__ = ['eval_decorator']

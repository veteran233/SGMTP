import os
import numpy as np
import pandas as pd

from pcdet.utils.calibration_kitti import Calibration
from pcdet.datasets import DatasetTemplate
from pcdet.utils import box_utils

from .tracking_utils import get_tracking_from_label


class KittiTrackingDataset(DatasetTemplate):

    def __init__(self,
                 dataset_cfg,
                 class_names,
                 dataset_path,
                 split,
                 training=False):

        super().__init__(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         training=training,
                         root_path=dataset_path)

        self.dataset_path = dataset_path
        self.split = split
        self.scenes = sorted([
            int(i) for i in os.listdir(f'{dataset_path}/{self.split}/velodyne')
        ])

        self.calib = []
        self.label = []
        for scene in self.scenes:
            self.calib.append(self.get_calib(scene))
        if self.mode == 'train':
            for scene in self.scenes:
                self.label.append(self.get_label(scene))

        data_split = pd.read_csv(
            f'{dataset_path}/{split}/evaluate_tracking.seqmap.val',
            sep=' ',
            header=None).values
        self.data = []
        for scene, _, start, end in data_split:
            for frame_id in range(start, end):
                self.data.append((scene, frame_id))

    def __getitem__(self, index):
        data_dict = {}

        scene, frame_id = self.data[index]
        calib = self.calib[scene]

        if self.mode == 'train':
            label = self.label[scene]
            data_dict['label'] = label

        points = self.get_lidar(scene, frame_id)
        image_shape = (375, 1242)

        # FOV
        points = points[self.get_fov_flag(calib.lidar_to_rect(points[:, :3]),
                                          image_shape, calib)]

        data_dict['scene'] = scene
        data_dict['frame_id'] = frame_id
        data_dict['calib'] = self.calib[scene]
        data_dict['image_shape'] = image_shape
        data_dict['points'] = points

        data_dict = self.prepare_data(data_dict)

        return data_dict

    def __len__(self):
        return len(self.data)

    def get_calib(self, scene):
        calib_file = f'{self.dataset_path}/{self.split}/calib/{scene:04d}.txt'
        return Calibration(calib_file)

    def get_label(self, scene):
        label_file = f'{self.dataset_path}/{self.split}/label_02/{scene:04d}.txt'
        return get_tracking_from_label(label_file)

    def get_lidar(self, scene, idx):
        lidar_file = f'{self.dataset_path}/{self.split}/velodyne/{scene:04d}/{idx:06d}.bin'
        return np.fromfile(lidar_file, np.float32).reshape(-1, 4)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0]
                                    < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1]
                                    < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    @staticmethod
    def generate_prediction_dicts(batch_dict,
                                  pred_dicts,
                                  class_names,
                                  class_filter=None,
                                  output_path=None):

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples),
                'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]),
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]),
                'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].detach().cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].detach().cpu().numpy()
            pred_labels = box_dict['pred_labels'].detach().cpu().numpy()

            if class_filter is not None:
                mask_filter = np.array(class_names)[pred_labels - 1]
                mask_filter = (mask_filter.reshape(
                    -1, 1) == class_filter).sum(axis=-1).astype(np.bool8)

                pred_scores = pred_scores[mask_filter]
                pred_boxes = pred_boxes[mask_filter]
                pred_labels = pred_labels[mask_filter]

            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(
                pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape)

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(
                -pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            scene_id = batch_dict['scene'][index]
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            annos.append(single_pred_dict)

            if output_path is not None:
                if not os.path.exists(f'{output_path}/{scene_id:04d}'):
                    os.makedirs(f'{output_path}/{scene_id:04d}')

                cur_det_file = f'{output_path}/{scene_id:04d}/{frame_id:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                            % (single_pred_dict['name'][idx],
                               single_pred_dict['alpha'][idx], bbox[idx][0],
                               bbox[idx][1], bbox[idx][2], bbox[idx][3],
                               dims[idx][1], dims[idx][2], dims[idx][0],
                               loc[idx][0], loc[idx][1], loc[idx][2],
                               single_pred_dict['rotation_y'][idx],
                               single_pred_dict['score'][idx]),
                            file=f)

        return annos

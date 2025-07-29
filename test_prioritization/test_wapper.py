import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from functools import partial

from easydict import EasyDict as Dict
from shapely import Point

from pcdet.utils.calibration_kitti import Calibration

from dataset_utils.tracking_utils import get_tracking_from_label


class test_decorator:

    def __init__(self, obj):
        self.o = obj
        self.timeout_threadhold = 3

    def __call__(self, frameworks, detectors, dataset_path, tracking_path,
                 split, output_path):

        self.frameworks = frameworks
        self.detectors = detectors
        self.dataset_path = dataset_path
        self.tracking_path = tracking_path
        self.split = split
        self.output_path = output_path

        data_split = pd.read_csv(
            f'{dataset_path}/{split}/evaluate_tracking.seqmap.{split}',
            sep=' ',
            header=None).values

        calib = []
        pose = []
        sg = []
        for scene, _, _, _ in data_split:
            calib.append(self.get_calib(scene))
            pose.append(self.get_pose(scene))
            sg.append(self.get_sg(scene))

        self.o(
            frameworks, detectors,
            partial(
                self.run,
                split=split,
                data_split=data_split,
                calib=calib,
                timeout_threadhold=self.timeout_threadhold,
                output_path=output_path,
                get_tracking_result=partial(self.get_tracking_result,
                                            tracking_path, split),
                get_sg_location=partial(self.get_sg_location, pose, sg),
            ))

    def get_calib(self, scene):
        calib_file = f'{self.dataset_path}/{self.split}/calib/{scene:04d}.txt'
        return Calibration(calib_file)

    def get_pose(self, scene):
        pose_file = f'{self.dataset_path}/{self.split}/pose/{scene:04d}/pose.txt'
        ret = []
        for line in open(pose_file, 'r').readlines():
            line = line.split(' ')
            mat = np.array(line).reshape(-1, 4)
            mat = np.vstack((mat, [[0, 0, 0, 1]])).astype(np.float32)
            ret.append(mat)
        return ret

    def get_sg(self, scene):
        sg_file = f'{self.dataset_path}/{self.split}/sg/{scene:04d}.pkl'
        return pkl.load(open(sg_file, 'rb'))

    @staticmethod
    def run(fw, det, tqdm_pos, split, data_split, calib, timeout_threadhold,
            output_path, **func):

        func = Dict(func)
        get_tracking_result = func.get_tracking_result
        get_sg_location = func.get_sg_location

        get_loc_dims_and_heading = test_decorator.get_loc_dims_and_heading
        is_correct_tracking = test_decorator.is_correct_tracking
        is_correct_angle = test_decorator.is_correct_angle
        create_state = test_decorator.create_state

        all_error_count = Dict(TRUE=[], AD=[], MD=[], IDS=[], FA=[])
        all_RQ2 = {}

        for scene, _, start, end in tqdm(data_split,
                                         f'Framework : {fw} Detector : {det}',
                                         position=tqdm_pos):

            error_count = Dict(TRUE=0, AD=0, MD=0, IDS=0, FA=0)
            state_of_cars = []

            result = get_tracking_result(fw, det, scene)

            for frame_id in range(start, end):
                state_of_cars.append({})

                prev_state = {}
                if len(state_of_cars) > 1:
                    prev_state = state_of_cars[-2]
                cur_state = state_of_cars[-1]
                cur_pos = len(state_of_cars) - 1

                if frame_id in result:

                    init_to_true_list = []
                    for track_id, loc_lidar, dims_lidar, heading in get_loc_dims_and_heading(
                            result[frame_id],
                            calib[scene],
                    ):

                        sg_loc = get_sg_location(scene, frame_id, loc_lidar)

                        if track_id not in prev_state:
                            create_state(cur_state, track_id, 'INIT', sg_loc,
                                         loc_lidar, dims_lidar, heading, 0)
                        else:
                            if is_correct_tracking(
                                    prev_state[track_id].sg_loc,
                                    sg_loc,
                            ):
                                error_count.TRUE += 1

                                if not is_correct_angle(
                                        prev_state[track_id].heading,
                                        heading,
                                ):
                                    error_count.FA += 1

                                if prev_state[track_id].state == 'INIT':
                                    init_to_true_list.append(track_id)

                                create_state(cur_state, track_id, 'TRUE',
                                             sg_loc, loc_lidar, dims_lidar,
                                             heading, 0)
                            else:
                                if prev_state[track_id].state == 'TRUE':
                                    error_count.MD += 1
                                    test_decorator.do_RQ2(
                                        all_RQ2,
                                        prev_state[track_id].sg_loc,
                                        sg_loc,
                                    )
                                elif prev_state[track_id].state == 'INIT':
                                    error_count.AD += 1

                                create_state(cur_state, track_id, 'INIT',
                                             sg_loc, loc_lidar, dims_lidar,
                                             heading, 0)

                    # IDS
                    prev_track_id_list = [
                        k for k, v in prev_state.items() if v.state == 'MD'
                    ]
                    for true_id in init_to_true_list:

                        flag = False
                        for prev_track_id in prev_track_id_list:
                            if prev_track_id in cur_state: continue

                            if is_correct_tracking(
                                    prev_state[prev_track_id].sg_loc,
                                    cur_state[true_id].sg_loc,
                            ):
                                error_count.IDS += 1
                                flag = True
                                break

                        if flag:
                            prev_track_id_list.remove(prev_track_id)
                            prev_state.pop(prev_track_id)

                for prev_track_id in list(prev_state.keys()):
                    if prev_track_id in cur_state: continue

                    if prev_state[prev_track_id].state == 'TRUE':
                        error_count.MD += 1
                        # test_decorator.do_cluster(
                        #     all_cluster,
                        #     state_of_cars[cur_pos - 2][prev_track_id].sg_loc,
                        #     state_of_cars[cur_pos - 1][prev_track_id].sg_loc,
                        # )

                        create_state(cur_state, prev_track_id, 'MD',
                                     prev_state[prev_track_id].sg_loc,
                                     prev_state[prev_track_id].loc_lidar,
                                     prev_state[prev_track_id].dims_lidar,
                                     prev_state[prev_track_id].heading, 0)
                    elif prev_state[prev_track_id].state == 'MD':
                        create_state(cur_state, prev_track_id, 'MD',
                                     prev_state[prev_track_id].sg_loc,
                                     prev_state[prev_track_id].loc_lidar,
                                     prev_state[prev_track_id].dims_lidar,
                                     prev_state[prev_track_id].heading,
                                     prev_state[prev_track_id].timeout)
                    elif prev_state[prev_track_id].state == 'INIT':
                        error_count.AD += 1

                cur_track_id_list = list(cur_state.keys())
                for track_id in cur_track_id_list:
                    if cur_state[track_id].state != 'MD':
                        continue

                    cur_state[track_id].timeout += 1
                    if cur_state[track_id].timeout > timeout_threadhold:
                        cur_state.pop(track_id)

            for k, v in error_count.items():
                all_error_count[k].append(v)

        os.makedirs(f'{output_path}/{split}', exist_ok=True)
        pd.DataFrame(all_error_count).to_csv(
            f'{output_path}/{split}/{fw}_{det}_result.csv', index=False)

        all_RQ2 = {k: [v] for k, v in all_RQ2.items()}
        __key = sorted(all_RQ2)
        all_RQ2 = {k: all_RQ2[k] for k in __key}
        pd.DataFrame(all_RQ2).to_csv(
            f'{output_path}/{split}/{fw}_{det}_RQ2.csv', index=False)

    @staticmethod
    def get_tracking_result(tracking_path, split, framework, detector, scene):
        label_file = f'{tracking_path}/{split}/{framework}/{detector}/data/{scene:04d}.txt'
        return get_tracking_from_label(label_file)

    @staticmethod
    def get_loc_dims_and_heading(obj_list, calib):
        loc = np.concatenate([obj.loc.reshape(1, -1) for obj in obj_list],
                             axis=0)
        dims = np.array([[obj.l, obj.h, obj.w] for obj in obj_list],
                        dtype=np.float32)
        rots = np.array([obj.ry for obj in obj_list], dtype=np.float32)
        track_id = np.array([obj.track_id for obj in obj_list], dtype=np.int8)

        loc_lidar = calib.rect_to_lidar(loc)

        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        dims_lidar = np.hstack((l, w, h))

        return zip(track_id, loc_lidar, dims_lidar, -(np.pi / 2 + rots))

    @staticmethod
    def get_sg_location(pose, sg, scene, frame_id, loc_lidar):
        p = pose[scene][frame_id]

        loc_lidar = np.hstack((loc_lidar, 1)).astype(np.float32).reshape(1, -1)
        loc_lidar = (loc_lidar @ p.T)[0, :2]

        for road_id, road in enumerate(sg[scene]):
            for lane in road:
                if lane['scene_graph'].contains(Point(loc_lidar)):
                    lane_id = lane['lane_id'] if 'lane_id' in lane else -1
                    return (road_id, lane_id)
        return (-1, -1)

    @staticmethod
    def is_correct_tracking(prev_sg, cur_sg):
        prev_road_id, prev_lane_id = prev_sg
        cur_road_id, cur_lane_id = cur_sg

        return (cur_road_id == -1 or cur_road_id == prev_road_id or cur_road_id
                == prev_road_id + 1) and (cur_lane_id == -1
                                          or cur_lane_id == prev_lane_id)

    @staticmethod
    def is_correct_angle(prev_angle, cur_angle):
        return np.abs(prev_angle - cur_angle) < np.pi / 4

    @staticmethod
    def create_state(state_of_cars, track_id, state, sg_loc, loc_lidar,
                     dims_lidar, heading, timeout):

        state_of_cars[track_id] = Dict(state=state,
                                       sg_loc=sg_loc,
                                       loc_lidar=loc_lidar,
                                       dims_lidar=dims_lidar,
                                       heading=heading,
                                       timeout=timeout)

    @staticmethod
    def do_RQ2(all_cluster, prev_sg, cur_sg):
        prev_road_id, prev_lane_id = prev_sg
        cur_road_id, cur_lane_id = cur_sg

        if prev_road_id == -1 and cur_road_id != -1:
            cur_road_id = 1
        elif prev_road_id != -1 and cur_road_id == -1:
            prev_road_id = 0
        elif prev_road_id != -1 and cur_road_id != -1:
            x = min(prev_road_id, cur_road_id)
            prev_road_id -= x
            cur_road_id -= x

        key = f'{prev_road_id} {prev_lane_id} {cur_road_id} {cur_lane_id}'
        if key in all_cluster:
            all_cluster[key] += 1
        else:
            all_cluster[key] = 1


__all__ = ['test_decorator']

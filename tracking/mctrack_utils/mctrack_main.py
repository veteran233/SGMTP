import sys
from copy import deepcopy


def run(scene_id, scenes_data, cfg, tracking_results):
    """
    Info: This function tracks objects in a given scene, processes frame data, and stores tracking results.
    Parameters:
        input:
            scene_id: ID of the scene to process.
            scenes_data: Dictionary with scene data.
            cfg: Configuration settings for tracking.
            args: Additional arguments.
            tracking_results: Dictionary to store results.
        output:
            tracking_results: Updated tracking results for the scene.
    """
    try:
        from tracker.base_tracker import Base3DTracker
        from dataset.baseversion_dataset import BaseVersionTrackingDataset
    except Exception as e:
        sys.path.insert(0, 'tracking/framework/MCTrack')
        from tracker.base_tracker import Base3DTracker
        from dataset.baseversion_dataset import BaseVersionTrackingDataset

    scene_data = scenes_data[scene_id]
    dataset = BaseVersionTrackingDataset(scene_id, scene_data, cfg=cfg)
    tracker = Base3DTracker(cfg=cfg)
    all_trajs = {}

    for index in range(len(dataset)):
        frame_info = dataset[index]
        frame_id = frame_info.frame_id
        cur_sample_token = frame_info.cur_sample_token
        all_traj = tracker.track_single_frame(frame_info)
        result_info = {
            "frame_id": frame_id,
            "cur_sample_token": cur_sample_token,
            "trajs": deepcopy(all_traj),
            "transform_matrix": frame_info.transform_matrix,
        }
        all_trajs[frame_id] = deepcopy(result_info)
    if cfg["TRACKING_MODE"] == "GLOBAL":
        trajs = tracker.post_processing()
        for index in range(len(dataset)):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                for bbox in trajs[track_id].bboxes:
                    if (bbox.frame_id == frame_id and bbox.is_interpolation
                            and track_id
                            not in all_trajs[frame_id]["trajs"].keys()):
                        all_trajs[frame_id]["trajs"][track_id] = bbox

        for index in range(len(dataset)):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                det_score = 0
                for bbox in trajs[track_id].bboxes:
                    det_score = bbox.det_score
                    break
                if (track_id in all_trajs[frame_id]["trajs"].keys() and
                        det_score <= cfg["THRESHOLD"]["GLOBAL_TRACK_SCORE"]):
                    del all_trajs[frame_id]["trajs"][track_id]

    tracking_results[scene_id] = all_trajs

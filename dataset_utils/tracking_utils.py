import numpy as np


def get_tracking_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    tracking = dict()
    cur_frame_tracking = []
    for line in lines:
        cur_label = tracking_object_label(line)
        if cur_label.cls_type != 'Car': continue
        if len(cur_frame_tracking
               ) == 0 or cur_frame_tracking[-1].frame == cur_label.frame:
            cur_frame_tracking.append(cur_label)
        else:
            tracking[cur_frame_tracking[-1].frame] = cur_frame_tracking
            cur_frame_tracking = [cur_label]
    if len(cur_frame_tracking) != 0:
        tracking[cur_frame_tracking[-1].frame] = cur_frame_tracking
    return tracking


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class tracking_object_label():

    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.frame = int(label[0])
        self.track_id = int(label[1])
        self.cls_type = label[2]
        self.truncation = float(label[3])
        self.occlusion = float(label[4])
        self.alpha = float(label[5])
        self.box2d = np.array(
            (
                float(label[6]),
                float(label[7]),
                float(label[8]),
                float(label[9]),
            ),
            dtype=np.float32,
        )
        self.h = float(label[10])
        self.w = float(label[11])
        self.l = float(label[12])
        self.loc = np.array(
            (float(label[13]), float(label[14]), float(label[15])),
            dtype=np.float32,
        )
        self.ry = float(label[16])
        self.score = float(label[17]) if len(label) == 18 else -1.0

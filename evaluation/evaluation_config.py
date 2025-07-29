FORMAT_NAME = {
    'pointpillar': 'PointPillar',
    'pv_rcnn': 'PV-RCNN',
    'mctrack_global': 'MCTrack GLOBAL',
    'mctrack_online': 'MCTrack ONLINE',
    'wu_global': 'CGMOT GLOBAL',
    'wu_online': 'CGMOT ONLINE',
    'sgmtp': 'SGMTP',
    'random': 'Random'
}


def get_format_name(name):
    if name in FORMAT_NAME:
        return FORMAT_NAME[name]
    return name

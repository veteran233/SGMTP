import pandas as pd

tracker_list = [
    'mctrack_online',
    'mctrack_global',
    'cgmot_online',
    'cgmot_global',
]

detector_list = [
    'pointpillar',
    'pv_rcnn',
]

output_dir = 'test_prioritization/output/val'

if __name__ == '__main__':

    for tr in tracker_list:
        for det in detector_list:
            x = pd.read_csv(f'{output_dir}/{tr}_{det}_RQ2.csv')
            print(f'{tr} {det}')

            c0 = 0
            c1 = 0
            c2 = 0
            for col in x.columns:
                if col.startswith('-1 -1'):
                    c0 += x[col].loc[0]
                elif col.startswith('0 -1'):
                    c1 += x[col].loc[0]
                elif col.startswith('0 '):
                    c2 += x[col].loc[0]

            print(c0, c1, c2)
            print()

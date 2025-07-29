import multiprocessing

from .tracking_wapper import *


@track_decorator
def track(runner, detectors, num_workers):

    with multiprocessing.Pool(num_workers, maxtasksperchild=1) as pool:

        tasks = []
        tqdm_pos = 0

        for run in runner:
            for det in detectors:

                tasks.append(pool.apply_async(run, (det, tqdm_pos)))
                tqdm_pos += 1

        [t.get() for t in tasks]

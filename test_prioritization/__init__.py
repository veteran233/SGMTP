import multiprocessing

from .test_wapper import *


@test_decorator
def test(frameworks, detectors, run, num_workers=4):

    with multiprocessing.Pool(num_workers) as pool:

        tasks = []
        tqdm_pos = 0

        for fw in frameworks:
            for det in detectors:

                tasks.append(pool.apply_async(run, (fw, det, tqdm_pos)))
                tqdm_pos += 1

        [t.get() for t in tasks]

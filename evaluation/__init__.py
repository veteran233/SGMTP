from .evaluation_wapper import *


@eval_decorator
def eval(frameworks, detectors, criteria, run):

    for fw in frameworks:

        for det in detectors:

            for cri in criteria:

                run(fw, det, cri)

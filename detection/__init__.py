from .detection_wapper import *


@detect_decorator
def detect(models_name, run):

    for model_name in models_name:

        run(model_name)

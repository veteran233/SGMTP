import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

import uvicorn

app = FastAPI(docs_url=None, root_path=None)

python_path = '/path/to/python'


class detection_t(BaseModel):
    dataset_path: str
    split: str
    second_per_data: int
    stride: int
    check_pointpillar: bool
    check_pv_rcnn: bool


class track_and_test_t(BaseModel):
    split: str
    check_pointpillar: bool
    check_pv_rcnn: bool
    check_mctrack_online: bool
    check_mctrack_global: bool
    check_wu_online: bool
    check_wu_global: bool


@app.get('/')
async def func():
    content = open('./app/index.html', 'r').read()
    return HTMLResponse(content)


@app.get('/styles.css')
async def func():
    return FileResponse('./app/styles.css')


@app.get('/script.js')
async def func():
    return FileResponse('./app/script.js')


@app.post('/detection')
async def func(x: detection_t):
    print(x)
    ...
    # os.system(
    #     f'{python_path} tools/generate_dataset.py -dp {x.dataset_path} --second-per-data {x.second_per_data} --stride {x.stride}'
    # )


@app.post('/tracking')
async def func(x: track_and_test_t):
    print(x)
    ...
    # default_model_list = ['pointpillar', 'pv_rcnn']
    # default_tracker_list = [
    #     'mctrack_online', 'mctrack_global', 'wu_online', 'wu_global'
    # ]

    # model = [m for m in default_model_list if getattr(x, f'check_{m}')]
    # tracker = [t for t in default_tracker_list if getattr(x, f'check_{t}')]

    # model = ' '.join(model)
    # tracker = ' '.join(tracker)

    # os.system(
    #     f'{python_path} test4.py -s {x.split} -fw {tracker} -det {model}')


@app.post('/testing')
async def func(x: track_and_test_t):
    print(x)
    ...
    # default_model_list = ['pointpillar', 'pv_rcnn']
    # default_tracker_list = [
    #     'mctrack_online', 'mctrack_global', 'wu_online', 'wu_global'
    # ]

    # model = [m for m in default_model_list if getattr(x, f'check_{m}')]
    # tracker = [t for t in default_tracker_list if getattr(x, f'check_{t}')]

    # model = ' '.join(model)
    # tracker = ' '.join(tracker)

    # os.system(
    #     f'{python_path} test5.py -s {x.split} -fw {tracker} -det {model}')


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

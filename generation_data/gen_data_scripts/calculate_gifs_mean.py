import os
import glob

_DATASET_DIR  = '../'
_DATASET_NAME = 'train_data_diff_color_0526'
_DATASET_TYPE = 'valid_data'
_DATA_DIR = _DATASET_DIR + _DATASET_NAME + '/' + _DATASET_TYPE + '/object_0'

def get_all_traindata(dir):
    all_gifs = glob.glob(os.path.join(dir, '*.gif'))
    print('all gifs:', len(all_gifs))
    return all_gifs


all_gifs = get_all_traindata(_DATA_DIR)


import imageio
import numpy as np
import multiprocessing as mp

def cal_gif_rbg_mean(gif_name):
    gif = imageio.mimread(gif_name)
    gif = np.array(gif)
    gif = gif[..., :3]
        
    pic_r, pic_g, pic_b = 0, 0, 0
    for pic in gif:
        pic_r += np.mean(pic[..., 0])
        pic_g += np.mean(pic[..., 1])
        pic_b += np.mean(pic[..., 2])   
    return np.array([pic_r, pic_g, pic_b]) / gif.shape[0]


def multicore(core_num=4):
    pool = mp.Pool(processes=core_num if core_num is not None else 4)
    res = pool.map(cal_gif_rbg_mean, all_gifs)
    gif_means = np.array(res)
    # 對第0軸每個元素作平均
    all_mean = np.mean(gif_means, axis=0)
    return all_mean


import time

start = time.time()
all_mean = multicore(mp.cpu_count())
print('calculation time:', time.time() - start)
print(all_mean)


import yaml

r = float(all_mean[0])
g = float(all_mean[1])
b = float(all_mean[2])
content = {
    'dataset_name' : _DATASET_NAME,
    'dataset_type' : _DATASET_TYPE,
    'all_gifs_mean': {
        'r': r, 'g': g, 'b': b
    }
}

filename = os.path(_DATASET_DIR, _DATASET_NAME, _DATASET_TYPE + '.yaml')
print('save to', filename)
print(yaml.dump(content, default_flow_style=False))

with open(filename, 'w') as file:
    yaml.dump(content, file, default_flow_style=False)



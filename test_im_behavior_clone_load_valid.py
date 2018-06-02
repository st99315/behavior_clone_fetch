import tensorflow as tf

import os
import glob
import imageio
import numpy as np
from matplotlib import pyplot as plt

import gym
from gym.envs.robotics.fetch_env import goal_distance
from gym.envs.robotics import FetchPickAndPlaceEnv, FetchPickAndPlaceJointEnv

from im_network_one_gif import BehaviorClone
try:
    import fetch_remote.utils as frutils
except ImportError:
    import generation_data.fetch_remote.utils as frutils
import utils
from config import cfg


def load_valid(dir):
    all_gifs = glob.glob(os.path.join(dir, '*.gif'))
    return all_gifs


DEMO_TIMES = 10
MODEL_CKPT_DIR = 'checkpoints/'
_DATASET_DIR = './generation_data/train_data_diff_color_0526/'
_VALID_DATA = _DATASET_DIR + 'valid_data/object_0'

GYM_PATH = gym.__path__[0]
XML_PATH = os.path.join(GYM_PATH, 'envs/robotics/assets/fetch/myenvs/blotchy_0130_marbled_0170.xml')
# XML_PATH = os.path.join(GYM_PATH, 'envs/robotics/assets/fetch/myenvs/perforated_0016_veined_0091.xml')


args = frutils.get_args()
frutils.set_env_variable(args.display)

all_gifs = load_valid(_VALID_DATA)

env = FetchPickAndPlaceEnv(xml_file=XML_PATH)
# env = FetchPickAndPlaceJointEnv(xml_file=XML_PATH)

_, build_log, run_log, fb_log = utils.set_logger(['build', 'run', 'act_fb'], 'valid.log')

m = BehaviorClone(training=False, logger=build_log)
m.build_inputs_and_outputs()

with tf.Session() as sess:
    # -------restore------#e 
    log_dir = MODEL_CKPT_DIR

    model_file = tf.train.latest_checkpoint(log_dir)
    if model_file is not None:
        # print('Use model_file = ' + str(model_file) + '.')
        build_log.info('Use model_file = ' + str(model_file) + '.')
        saver = tf.train.Saver()
        saver.restore(sess, model_file)
    else: 
        build_log.error('No model, exit')
        # print('No model, exit')
        exit()

    for gif in all_gifs:
        # print('gif name:', gif)
        run_log.info('gif name: {}'.format(gif))
        """ Load valid data. """
        gif_pics = np.array(imageio.mimread(gif), dtype=np.float32)
        gif_pics = gif_pics[:, :, :, :3]
        feedback = np.loadtxt(gif.rpartition('.')[0]+'.csv')

        tar_dir  = gif.rpartition('/')[0]+'/target/'
        tar_name = gif.rpartition('.')[0].rpartition('/')[-1]+'.csv'
        tar_path = tar_dir + tar_name
        tar_pos = np.loadtxt(tar_path)[:2]

        obs = env.reset(tar_pos)
        total_reward = 0

        for step, gif_pic in enumerate(gif_pics):
            if args.display:
                env.render()
            else:
                rgb_obs = env.sim.render(width=128, height=128, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
   
            traject = feedback[step, :11]
            traject = traject[np.newaxis, :]

            rgb_obs = gif_pic
            rgb_obs = np.array(rgb_obs, dtype=np.float32)
            rgb_obs -= np.array([162.4602994276193, 179.77208175703808, 174.63593676080725])
            rgb_obs /= 255.

            rgb_obs = rgb_obs[np.newaxis, :]
            predict = sess.run([m.batch_prediction], feed_dict={m.batch_gif: rgb_obs, m.batch_feedback: traject})
            
            predict = np.squeeze(predict)
            actions = np.append(predict[:3], predict[3:4])
            
            # print(actions)
            obs, r, done, info = env.step(actions)
            total_reward += r

            if step % 20 == 0:
                rgb_obs = env.sim.render(width=200, height=200, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)

                plt.figure(1)
                plt.imshow(gif_pic/255.)

                plt.figure(2)
                plt.imshow(rgb_obs)
                plt.show(block=False)
                plt.pause(0.001)

            if info['is_success'] or done:
                break

        # plt.figure(1)
        # plt.imshow(gif_pic/255.)
        # plt.figure(2)
        # plt.imshow(rgb_obs)
        # plt.show(block=False)
        # plt.pause(0.001)

        run_log.info("total reward %0.2f\n" % total_reward)

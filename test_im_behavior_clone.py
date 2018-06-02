import tensorflow as tf

import os
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
from config import cfg
import utils
import load_data


CKPT_DIR = 'checkpoints/'
GIF_MEAN = load_data.get_gifs_mean('')
DEMO_TIMES = 10

GYM_PATH = gym.__path__[0]
XML_PATH = os.path.join(GYM_PATH, 'envs/robotics/assets/fetch/myenvs/blotchy_0130_marbled_0170.xml')
# XML_PATH = os.path.join(GYM_PATH, 'envs/robotics/assets/fetch/myenvs/perforated_0016_veined_0091.xml')

args = frutils.get_args()
frutils.set_env_variable(args.display)

env = FetchPickAndPlaceEnv(xml_file=XML_PATH)
# env = FetchPickAndPlaceJointEnv(xml_file=XML_PATH)

_, build_log, run_log, fb_log = utils.set_logger(['build', 'run', 'act_fb'], 'test.log')

m = BehaviorClone(training=False, logger=build_log)
m.build_inputs_and_outputs()

with tf.Session() as sess:
    # -------restore------#e 
    log_dir = CKPT_DIR

    model_file = tf.train.latest_checkpoint(log_dir)
    if model_file is not None:
        build_log.info('Use model_file = ' + str(model_file) + '.')
        saver = tf.train.Saver()
        saver.restore(sess, model_file)
    else: 
        build_log.error('No model, exit')
        exit()

    upper_sucess = 0
    grasp_sucess = 0
    for i in range(1000):
        obs = env.reset(rand_text=True, rand_shadow=True)
        total_reward = 0
        upper = 0
        grasp = 0

        finish = []
        actions = np.array([0., 0., 0., 1.])
        for step in range(500):

            if args.display:
                env.render()
            else:
                rgb_obs = env.sim.render(width=cfg['image_width'], height=cfg['image_height'], camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
                rgb_obs1 = env.sim.render(width=cfg['extra_width'], height=cfg['extra_height'], camera_name="gripper_camera_rgb", depth=False,
                    mode='offscreen', device_id=-1)
            
            traject = np.append(obs['eeinfo'][0], obs['weneed'])
            traject = np.append(traject, obs['gripper_dense'])
            traject = traject[np.newaxis, :]

            rgb_obs = np.array(rgb_obs, dtype=np.float32)
            rgb_obs -= GIF_MEAN
            rgb_obs /= 255.

            rgb_obs1 = np.array(rgb_obs1, dtype=np.float32)
            rgb_obs1 -= GIF_MEAN
            rgb_obs1 /= 255.

            rgb_obs = rgb_obs[np.newaxis, :]
            rgb_obs1 = rgb_obs1[np.newaxis, :]
            predict = sess.run([m.batch_prediction], feed_dict={m.batch_gif: rgb_obs, m.batch_ext: rgb_obs1, m.batch_feedback: traject})
            
            predict = np.squeeze(predict)
            actions = np.append(predict[:3], predict[3:4])
            
            actions *= 4.
            obs, r, done, info = env.step(actions)
            total_reward += r

            if step % 10 == 0:
                rgb_obs = env.sim.render(width=200, height=200, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
                ext_obs = env.sim.render(width=200, height=200, camera_name="gripper_camera_rgb", depth=False,
                    mode='offscreen', device_id=-1)
                plt.figure(1)
                plt.imshow(rgb_obs)
                plt.figure(2)
                plt.imshow(ext_obs)
                plt.show(block=False)
                plt.pause(0.001)

            # if (not upper and 
            #     goal_distance(obs['eeinfo'][0][:2], obs['achieved_goal'][:2]) < 0.05 and
            #     obs['eeinfo'][0][-1] > obs['achieved_goal'][-1] + .01):
            #     upper = 1
            #     break
                
            if info['is_success'] or done:
                break

        # rgb_obs = env.sim.render(width=200, height=200, camera_name="external_camera_0", depth=False,
        #     mode='offscreen', device_id=-1)
        # plt.figure(2)
        # plt.imshow(rgb_obs)
        # plt.show(block=False)
        # plt.pause(0.001)

        upper_sucess += upper
        run_log.info("%d total reward %0.2f. sucess %d rate %.2f \n" % (i, total_reward, upper_sucess, upper_sucess / (i+1)))

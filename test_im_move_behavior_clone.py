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


CKPT_DIR = '0516_315_feedbackWithoutBias/checkpoints/'
DEMO_TIMES = 10

GYM_PATH = gym.__path__[0]
XML_PATH = os.path.join(GYM_PATH, 'envs/robotics/assets/fetch/myenvs/blotchy_0130_marbled_0170.xml')
# XML_PATH = os.path.join(GYM_PATH, 'envs/robotics/assets/fetch/myenvs/perforated_0016_veined_0091.xml')

args = frutils.get_args()
frutils.set_env_variable(args.display)

env = FetchPickAndPlaceEnv(xml_file=XML_PATH)
# env = FetchPickAndPlaceJointEnv(xml_file=XML_PATH)
    
m = BehaviorClone(training=False)
m.build_inputs_and_outputs()

with tf.Session() as sess:
    # -------restore------#e 
    log_dir = CKPT_DIR

    model_file = tf.train.latest_checkpoint(log_dir)
    if model_file is not None:
        print('Use model_file = ' + str(model_file) + '.')
        saver = tf.train.Saver()
        saver.restore(sess, model_file)
    else: 
        print('No model, exit')
        exit()

    upper_sucess = 0
    grasp_sucess = 0
    for i in range(1000):
        obs = env.reset(rand_text=True, rand_shadow=True)
        total_reward = 0
        upper = 0
        grasp = 0

        for step in range(500):

            if args.display:
                env.render()
            else:
                rgb_obs = env.sim.render(width=128, height=128, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
            
            traject = np.append(obs['eeinfo'][0], obs['weneed'])
            traject = np.append(traject, obs['gripper_state'])
            # print('T:', traject)
            traject = traject[np.newaxis, :]

            rgb_obs = np.array(rgb_obs, dtype=np.float32)
            rgb_obs /= 255.
            # rgb_obs -= np.array([103.939, 116.779, 123.68])

            # if step % 10 == 0:
            #     plt.figure(2)
            #     plt.imshow(rgb_obs/255.)

            rgb_obs = rgb_obs[np.newaxis, :]
            predict = sess.run([m.batch_prediction], feed_dict={m.batch_gif: rgb_obs, m.batch_feedback: traject})
            
            predict = np.squeeze(predict)
            actions = np.append(predict[:3], predict[-1])
            # print('A:', actions)
            obs, r, done, info = env.step(actions)
            total_reward += r

            if step % 20 == 0:
                rgb_obs = env.sim.render(width=200, height=200, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
                # rgb_obs1 = env.sim.render(width=200, height=200, camera_name="external_camera_1", depth=False,
                #     mode='offscreen', device_id=-1)
                plt.figure(1)
                plt.imshow(rgb_obs)
                # plt.figure(2)
                # plt.imshow(rgb_obs1)
                plt.show(block=False)
                plt.pause(0.001)

            if (not upper and 
                goal_distance(obs['eeinfo'][0][:2], obs['achieved_goal'][:2]) < 0.05 and
                obs['eeinfo'][0][-1] > obs['achieved_goal'][-1] + .01):
                upper = 1
                break
                
            if info['is_success'] or done:
                break

        # plt.figure(1)
        # plt.imshow(gif_pic/255.)
        # plt.figure(2)
        # plt.imshow(rgb_obs)
        # plt.show(block=False)
        # plt.pause(0.001)
        upper_sucess += upper
        print(i, "total reward %0.2f. sucess %d rate %.2f" % (total_reward, upper_sucess, upper_sucess / (i+1)))


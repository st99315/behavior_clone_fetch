import tensorflow as tf

import os
import glob
import numpy as np
from matplotlib import pyplot as plt

import gym
from gym.envs.robotics.fetch_env import goal_distance
from gym.envs.robotics import FetchPickAndPlaceEnv, FetchPickAndPlaceJointEnv

from im_network_one_gif import BehaviorClone
try:
    import fetch_remote.utils as frutils
    from fetch_remote.utils.data_save import DataSaver
    from fetch_remote.utils.finite_state_machine import FSM
except ImportError:
    import utils as frutils
    from utils.data_save import DataSaver
    from utils.finite_state_machine import FSM

from config import cfg
import utils
import load_data


CKPT_DIR = 'checkpoints/'
DATASET_DIR = './generation_data/train_data_diff_color_0526/valid_data'

MAX_EPSO = 100
MAX_STEP = 300
ONE_TASK = 50
YAM_FILE = DATASET_DIR.rpartition('/')[-1]+'.yaml'
GIF_MEAN = load_data.get_gifs_mean(os.path.join(DATASET_DIR, YAM_FILE))

GRIPPER_STATE = 1
LIMIT_Z = .415
SCALE_SPEED = 4.0

GYM_PATH = gym.__path__[0]
XML_PATH = os.path.join(GYM_PATH, 'envs/robotics/assets/fetch/myenvs/blotchy_0130_marbled_0170.xml')
# XML_PATH = os.path.join(GYM_PATH, 'envs/robotics/assets/fetch/myenvs/perforated_0016_veined_0091.xml')


def get_lastnum(directory):
    try:
        allfiles = glob.glob(os.path.join(directory, 'object*'))
        sorted_files = sorted(allfiles)
        lastnum = int(sorted_files.pop().rpartition('_')[-1])
    except IndexError:
        print('Not Found object folder in', directory)
        exit()
    return lastnum


args = frutils.get_args()
frutils.set_env_variable(args.display)

env = FetchPickAndPlaceEnv(xml_file=XML_PATH)
# env = FetchPickAndPlaceJointEnv(xml_file=XML_PATH)

_, build_log, run_log = utils.set_logger(['build', 'run'], 'dagger.log')

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

    data_save_path = os.path.join(DATASET_DIR, 'object_{}'.format(get_lastnum(DATASET_DIR) + 1))
    saver    = DataSaver(data_save_path)
    tar_info = DataSaver(os.path.join(data_save_path, 'target'), info=True)

    for ep in range(MAX_EPSO):
        obs = env.reset(rand_text=True, rand_shadow=True)
        total_reward = 0

        # save object and goal pos
        tar_info.append(trajectory=np.append(obs['achieved_goal'], obs['desired_goal']))

        goal = obs['achieved_goal'].copy()
        goal[-1] = goal[-1] + .1
        simple_policy = FSM(np.append(obs['eeinfo'][0], GRIPPER_STATE), obs['achieved_goal'], goal, LIMIT_Z)
        total_reward = 0
        clip = (0, None)

        for step in range(MAX_STEP):

            rgb_obs = env.sim.render(width=cfg['image_width'], height=cfg['image_height'], camera_name="external_camera_0", depth=False,
                mode='offscreen', device_id=-1)
            # appending image to saver
            saver.append(image=rgb_obs)

            # prepocessing
            rgb_img = np.array(rgb_obs, dtype=np.float32)
            rgb_img -= GIF_MEAN
            rgb_img /= 255.
            rgb_img = rgb_img[np.newaxis, :]

            traject = np.append(obs['eeinfo'][0], obs['weneed'])
            traject = np.append(traject, obs['gripper_dense'])
            # appending current feedback: ee pos (x, y, z), all of robot joints angle and gripper state
            trajectory = traject.copy()
            traject = traject[np.newaxis, :]

            predict = sess.run([m.batch_prediction], feed_dict={m.batch_gif: rgb_img, m.batch_feedback: traject})
            expert  = simple_policy.execute()
            # appending control command: delta ee pos (x, y, z), gripper state
            trajectory = np.append(trajectory, expert)

            predict = np.squeeze(predict)
            g = predict[3:4]
            actions = np.append(predict[:3], g)
            
            # scale up action
            actions = actions * SCALE_SPEED 
            obs, r, done, info = env.step(actions)

            # update robot state
            simple_policy.robot_state = np.append(obs['eeinfo'][0], g)
            total_reward += r

            # appending auxiliary: object and gripper pos
            trajectory = np.append(trajectory, obs['achieved_goal'])
            trajectory = np.append(trajectory, obs['eeinfo'][0])
            # appending trajectory to saver
            saver.append(trajectory=trajectory)
            
            # clip data step
            finish, current = simple_policy.step
            if current > ONE_TASK:
                clip = (0, np.sum(finish) + ONE_TASK)

            if info['is_success'] or simple_policy.done:
                break

        # plt.imshow(rgb_obs)
        # plt.show(block=False)
        # plt.pause(0.001)

        saver.save(ep, clip)
        tar_info.save(ep)
        run_log.info("%d total reward %0.2f" % (ep, total_reward))

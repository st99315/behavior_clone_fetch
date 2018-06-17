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
DATASET_DIR = './generation_data/train_data_diff_color_0606/train_data'

MAX_EPSO = 10
MAX_STEP = 300
ONE_TASK = 50
YAM_FILE = DATASET_DIR.rpartition('/')[-1]+'.yaml'
GIF_MEAN = load_data.get_gifs_mean(os.path.join(DATASET_DIR, YAM_FILE))

GRIPPER_STATE = 1
SCALE_SPEED = 4.0


def get_lastnum(directory):
    try:
        allfiles = glob.glob(os.path.join(directory, 'object*'))
        sorted_files = sorted(allfiles, key=lambda x: int(x.rpartition('_')[-1]))
        lastnum = int(sorted_files.pop().rpartition('_')[-1])
    except IndexError:
        print('Not Found object folder in', directory)
        exit()
    return lastnum


args = frutils.get_args()
frutils.set_env_variable(args.display)

env = FetchPickAndPlaceEnv()
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
    run_log.info(data_save_path)
    saver    = DataSaver(data_save_path)
    tar_info = DataSaver(os.path.join(data_save_path, 'target'), info=True)
    saver.open_tf_writer(name=args.start)

    # count executed task
    task_exe = np.array([[0, 0, 0, 0, 0]])
    for ep in range(MAX_EPSO):
        obs = env.reset(rand_text=True, rand_shadow=True)
        total_reward = 0

        # save object and goal pos
        tar_info.append(trajectory=np.append(obs['achieved_goal'], obs['desired_goal']))

        goal = obs['achieved_goal'].copy()
        goal[-1] = goal[-1] + .1
        # current feedback, object pos, goal pos
        simple_policy = FSM(np.append(obs['eeinfo'][0], obs['gripper_dense']), obs['achieved_goal'], goal)
        total_reward = 0
        clip = (0, None)

        external = []
        eyehand = []
        for step in range(MAX_STEP):

            rgb_obs = env.sim.render(width=cfg['image_width'], height=cfg['image_height'], camera_name="external_camera_0", depth=False,
                mode='offscreen', device_id=-1)
            # appending image to saver
            saver.append(image=rgb_obs)

            ext_obs = env.sim.render(width=cfg['extra_width'], height=cfg['extra_width'], camera_name="gripper_camera_rgb", depth=False,
                mode='offscreen', device_id=-1)
            # appending image to saver
            saver.append(extra_img=ext_obs)

            # prepocessing
            rgb_img = np.array(rgb_obs, dtype=np.float32)
            # rgb_img -= GIF_MEAN
            rgb_img /= 255.
            ext_obs = np.array(ext_obs, dtype=np.float32)
            # ext_obs -= GIF_MEAN
            ext_obs /= 255.
            rgb_img = rgb_img[np.newaxis, :]
            ext_obs = ext_obs[np.newaxis, :]

            def enqueue(buff, img):
                if not len(buff):
                    buff = [img, img, img, img]
                else:
                    buff.pop(0)
                    buff.append(img)
                return buff
            external = enqueue(external, rgb_img)
            eyehand  = enqueue(eyehand, ext_obs)
            external_in = np.concatenate(external, axis=-1)
            eyehand_in = np.concatenate(eyehand, axis=-1)

            traject = np.append(obs['eeinfo'][0], obs['weneed'])
            traject = np.append(traject, obs['gripper_dense'])
            # appending current feedback: ee pos (x, y, z), all of robot joints angle and gripper state
            trajectory = traject.copy()
            traject = traject[np.newaxis, :]
            
            predict = sess.run([m.batch_prediction], feed_dict={m.batch_gif: external_in, m.batch_ext: eyehand_in, m.batch_fdb: traject})
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
            simple_policy.robot_state = np.append(obs['eeinfo'][0], obs['gripper_dense'])
            total_reward += r

            # appending auxiliary: object and gripper pos
            trajectory = np.append(trajectory, obs['achieved_goal'])
            trajectory = np.append(trajectory, obs['eeinfo'][0])
            # appending trajectory to saver
            saver.append(trajectory=trajectory)

            if info['is_success'] or simple_policy.done:
                break

        # plt.imshow(rgb_obs)
        # plt.show(block=False)
        # plt.pause(0.001)

        # clip data step
        finish, current = simple_policy.step
        if current > ONE_TASK:
            clip = (0, np.sum(finish) + ONE_TASK)

        # record finish task
        arr_finish = np.array(finish)
        arr_finish.resize(task_exe.shape, refcheck=False)
        arr_finish = (arr_finish > 0).astype(np.int)
        task_exe = np.concatenate((task_exe, arr_finish), axis=0)

        if len(finish) > 0:
            saver.save(ep, clip)
            tar_info.save(ep)
            run_log.info("save   {} total reward {}. finish {} clip {}".format(ep, total_reward, len(finish), clip))
        else:
            saver.flush()
            tar_info.flush()
            run_log.info("unsave {} total reward {}. finish {} clip {}".format(ep, total_reward, len(finish), clip))

        task_exe = np.sum(task_exe, axis=0)[np.newaxis, :]
        run_log.info("total finish task {}".format(task_exe))

    saver.close_tf_writer()
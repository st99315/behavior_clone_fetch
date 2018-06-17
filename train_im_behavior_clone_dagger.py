import tensorflow as tf
from tensorflow.python.platform import flags

import time
import numpy as np
import os
from os.path import join
import glob
from config import cfg
from load_data import DataLoaderTFRecord as DataLoader
from im_network_one_gif import BehaviorClone
from utils import print_all_var, recreate_dir
from utils import set_logger, show_use_time
from gym.envs.robotics import FetchPickAndPlaceEnv
try:
    import fetch_remote.utils as frutils
    from fetch_remote.utils.data_save import DataSaver
    from fetch_remote.utils.finite_state_machine import FSM
except ImportError:
    import utils as frutils
    from utils.data_save import DataSaver
    from utils.finite_state_machine import FSM


_DATASET_DIR = './generation_data/train_data_diff_color_0615/'
_TRAIN_DATA = _DATASET_DIR + 'train_data'
_VALID_DATA = _DATASET_DIR + 'valid_data'
CKPT_DIR = 'checkpoints/'

_DAGGER = 50
_EPOCHS = 100
_PRINT_STEP = 100

MAX_EPSO_T = 50
MAX_EPSO_V = 5
MAX_STEP = 300
ONE_TASK = 40
SCALE_SPEED = 4.0

FLAGS = flags.FLAGS
flags.DEFINE_bool('drop_out', False, 'if True, use drop_out for fc(fully connected!')
flags.DEFINE_string('log_dir', 'log/', 'log directory')
flags.DEFINE_string('model_dir', 'checkpoints/', 'model directory')

# get logger
logger, build_logger, train_logger, dagger_log = set_logger(['build', 'train', 'dagger'], log_dir='logging_log')
summary_writer = tf.summary.FileWriter(FLAGS.log_dir)


def get_trainable_dic():
    all_w_b = dict()
    for v in tf.trainable_variables():
        name = v.name.replace(":0", "")
        all_w_b[name] = v
        build_logger.info('{}->{}'.format(name, v))
    build_logger.info('')
    return all_w_b 


def save_model(sess, saver, epoch, show_str, g_step):
    save_model_path = 'checkpoints/'
    saver.save(sess, save_model_path+'model.ckpt', global_step=g_step)
    train_logger.info('Save Model! ' + show_str)


def train_all_batch(sess, model, epoch, datanums, training=True):
    head_str = "Train" if training else "Valid"
    start_time = time.time()
    
    train_logger.info('training = {}, head_str={}'.format(training, head_str))

    sum_pattern = 0.
    im_loss_sum = 0.
    im_loss_avg = 0.

    for i in range(1, np.ceil(datanums / cfg['batch_size']).astype(np.int32)):
        try:
            if training:
                _, total_im_loss, predict = sess.run([model.train_op, model.total_im_loss, model.batch_prediction], 
                    feed_dict={is_training: training})
                
            else:
                total_im_loss, predict = sess.run([model.total_im_loss, model.batch_prediction],
                    feed_dict={is_training: training}) 

                if i==1:
                    for line in predict:
                        train_logger.debug('predict cmd: {}, obj: {}, grip: {}'.format(line[:4], line[4:7], line[7:]))

        except tf.errors.OutOfRangeError:
            train_logger.waring('Batch out of range!')
            break
            # exit()
        
        im_loss_sum += total_im_loss
        im_loss_avg = im_loss_sum / i

        if i % _PRINT_STEP == 0:
            train_logger.info("{} -> epoch: {:0>4d}, gif_pattern iter: {:4d}, total_im_loss: {:6.2f}, im_loss_avg: {:4.2f}". \
                format(head_str, epoch, i, total_im_loss, im_loss_avg))

            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
            # summary = tf.Summary()

            if training:
                all_gif_num = epoch * datanums + i
                summary = tf.Summary()
                summary.value.add(tag="[Train] loss multi (every_100_gif)", simple_value=im_loss_avg)
                summary.value.add(tag="[Train] loss (every_100_gif)", simple_value=np.sqrt(im_loss_avg / 100.0))
                summary_writer.add_summary(summary, all_gif_num)

    train_logger.info("{} -> epoch: {:0>4d}, total iter: {:4d}". \
            format(head_str, epoch, i))
    summary_writer.flush()
    show_use_time(time.time() - start_time, head_str + ' use time:', train_logger)

    return im_loss_avg


def valid_batch(*arg, **kwargs):
    return train_all_batch(*arg, **kwargs, training=False)


def get_lastnum(directory):
    try:
        allfiles = glob.glob(os.path.join(directory, 'object*'))
        sorted_files = sorted(allfiles, key=lambda x: int(x.rpartition('_')[-1]))
        lastnum = int(sorted_files.pop().rpartition('_')[-1])
    except IndexError:
        print('Not Found object folder in', directory)
        exit()
    return lastnum


def dagger_gen_data(dataset_dir, max_epso, logger, config):
    args = frutils.get_args()
    frutils.set_env_variable(args.display)

    env = FetchPickAndPlaceEnv()

    m = BehaviorClone(training=False, logger=logger)
    m.build_inputs_and_outputs()

    with tf.Session(config=config) as sess:
        # -------restore------#e 
        log_dir = CKPT_DIR

        model_file = tf.train.latest_checkpoint(log_dir)
        if model_file is not None:
            logger.info('Use model_file = ' + str(model_file) + '.')
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
        else: 
            logger.error('No model, exit')
            exit()

        data_save_path = os.path.join(dataset_dir, 'object_{}'.format(get_lastnum(dataset_dir) + 1))
        logger.info(data_save_path)
        saver    = DataSaver(data_save_path)
        tar_info = DataSaver(os.path.join(data_save_path, 'target'), info=True)
        saver.open_tf_writer(name=args.start)

        # count executed task
        task_exe = np.array([[0, 0, 0, 0, 0]])
        for ep in range(max_epso):
            obs = env.reset(rand_text=True, rand_shadow=True)
            total_reward = 0

            # save object and goal pos
            tar_info.append(trajectory=np.append(obs['achieved_goal'], obs['desired_goal']))

            goal = obs['achieved_goal'].copy()
            goal[-1] = goal[-1] + .1
            # current feedback, object pos, goal pos
            simple_policy = FSM(np.append(obs['eeinfo'][0], obs['gripper_dense']), obs['achieved_goal'], goal)
            clip = (0, None)

            external = []
            eyehand  = []
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
                rgb_img /= 255.

                ext_obs = np.array(ext_obs, dtype=np.float32)
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
                eyehand_in  = np.concatenate(eyehand, axis=-1)

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
                logger.info("save   {} total reward {}. finish {} clip {}".format(ep, total_reward, len(finish), clip))
            else:
                saver.flush()
                tar_info.flush()
                logger.info("unsave {} total reward {}. finish {} clip {}".format(ep, total_reward, len(finish), clip))

            task_exe = np.sum(task_exe, axis=0)[np.newaxis, :]
            logger.info("total finish task {}".format(task_exe))

        saver.close_tf_writer()
        env.close()


logger.info('Start Logging')

# limit memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True                   # allocate dynamically
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # maximun alloc gpu 80% of MEM

# record start time
train_start_time = time.time()

for dagger_iter in range(_DAGGER):
    # Data Loader
    DataLoader.set_logger(build_logger)
    train_dlr = DataLoader(_TRAIN_DATA)
    valid_dlr = DataLoader(_VALID_DATA)

    train_data = train_dlr.input_pipeline()
    valid_data = valid_dlr.input_pipeline()

    is_training = tf.placeholder(dtype=bool,shape=())
    gif, ext, fdb, cmd = tf.cond(is_training, lambda:(train_data), lambda:(valid_data))

    m = BehaviorClone(logger=build_logger)
    m.set_network_property(drop_out=FLAGS.drop_out)
    m.build_inputs_and_outputs(tf.squeeze(gif), tf.squeeze(ext), tf.squeeze(fdb), tf.squeeze(cmd))
    m.build_train_op()

    with tf.Session(config=config) as sess:
        start_ep = 0

        # -------restore------
        recreate_dir(FLAGS.log_dir)
        model_file = tf.train.latest_checkpoint(FLAGS.model_dir)
        saver = tf.train.Saver(max_to_keep=2)
        if model_file is not None:
            build_logger.info('Use model_file = ' + str(model_file) + '!')
            saver.restore(sess, model_file)
            # get ckpt epoch num
            start_ep = int(model_file.rpartition('-')[-1]) + 1
        else:
            build_logger.info('Initialize all variables')
            sess.run(tf.global_variables_initializer())
            build_logger.info('Initialize all variables Finish')

        build_logger.info('--------- After build graph, get_trainable_dic() ------------')
        get_trainable_dic()

        try:
            DataLoader.start(sess)

            end_ep = int((np.floor(start_ep / (_EPOCHS - 1)) + 1) * _EPOCHS)
            for ep in range(start_ep, end_ep):
                train_logger.info('----- Train -----')
                train_avg_loss = train_all_batch(sess, m, ep, train_dlr.data_nums)
                train_logger.info('----- Valid -----')
                valid_avg_loss = valid_batch(sess, m, ep, valid_dlr.data_nums)

                epoch_train_loss = np.sqrt(train_avg_loss/100.0)
                epoch_valid_loss = np.sqrt(valid_avg_loss/100.0)
                
                #---Add Summary---#
                # summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
                # summary = tf.Summary()

                summary = tf.Summary()
                summary.value.add(tag="[Train] loss (every_epoch)", simple_value=epoch_train_loss)
                summary.value.add(tag="[Valid] loss (every_epoch)", simple_value=epoch_valid_loss)
                summary_writer.add_summary(summary, ep)
                summary_writer.flush()

                train_logger.info("epoch: {:0>4d}, epoch_train_loss: {:4.2f},  epoch_valid_loss: {:4.2f}". \
                    format(ep, epoch_train_loss, epoch_valid_loss))

                save_model(sess, saver, ep, 'epoch: ' + str(ep), g_step=ep)  
                show_use_time(time.time() - train_start_time, "All use time: ", train_logger)
            
        except KeyboardInterrupt:
            train_logger.info('Got Keyboard Interrupt!')
            exit()
        finally:
            DataLoader.close()
            logger.info('Stop Training and Logging')

    try:
        dagger_gen_data(_TRAIN_DATA, MAX_EPSO_T, dagger_log, config)
        dagger_gen_data(_VALID_DATA, MAX_EPSO_V, dagger_log, config)
    except KeyboardInterrupt:
        dagger_log.info('Got Keyboard Interrupt!')
        exit()

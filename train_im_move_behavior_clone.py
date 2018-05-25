
# Get 99% accuracy from tensorflow tutorial #

import tensorflow as tf
from tensorflow.python.platform import flags

import time
import numpy as np
from os.path import join

from config import cfg
from load_data import DataLoader
from im_network_one_gif import BehaviorClone
from utils import print_all_var, recreate_dir
from utils import set_logger, show_use_time


_DATASET_DIR = '/out/train_data_diff_color_0525/'
_TRAIN_DATA = _DATASET_DIR + 'train_data/object_0'
_VALID_DATA = _DATASET_DIR + 'valid_data/object_0'

_EPOCHS = 1000
_PRINT_STEP = 100

FLAGS = flags.FLAGS
flags.DEFINE_bool('drop_out', False, 'if True, use drop_out for fc(fully connected!')
flags.DEFINE_string('log_dir', 'log/', 'log directory')
flags.DEFINE_string('model_dir', 'checkpoints/', 'model directory')

# get logger
logger, build_logger, train_logger = set_logger(['build', 'train'], log_dir='logging_log')
summary_writer = tf.summary.FileWriter(FLAGS.log_dir)


def get_trainable_dic():
    all_w_b = dict()
    for v in tf.trainable_variables():
        name = v.name.replace(":0", "")
        all_w_b[name] = v
        build_logger.info('{}->{}'.format(name, v))
        # print('{}->{}'.format(name, v))
    build_logger.info('')
    # print()
    return all_w_b 


def save_model(sess, saver, epoch, show_str, g_step):
    save_model_path = 'checkpoints/'
    # saver.save(sess, save_model_path+'model-{:0>3d}.ckpt'.format(epoch))
    saver.save(sess, save_model_path+'model.ckpt', global_step=g_step)
    train_logger.info('Save Model! ' + show_str)
    # print('Save Model! ' + show_str)


def train_all_batch(sess, model, epoch, datanums, training=True):
    head_str = "Train" if training else "Valid"
    start_time = time.time()
    
    train_logger.info('training = {}, head_str={}'.format(training, head_str))
    # print('training = {}, head_str={}'.format(training, head_str))

    sum_pattern = 0.
    im_loss_sum = 0.
    im_loss_avg = 0.

    for i in range(1, datanums+1):
        try:
            if training:
                _, total_im_loss, predict = sess.run([model.train_op, model.total_im_loss, model.batch_prediction], 
                    feed_dict={is_training: training})
            else:
                total_im_loss, predict = sess.run([model.total_im_loss, model.batch_prediction],
                    feed_dict={is_training: training}) 

                if i==1:
                    for line in predict:
                        train_logger.debug('pred cmd: {}, obj: {}, grip: {}'.format(line[:4], line[4:7], line[7:]))
                        # print('pred cmd:', line[:4], 'obj:', line[4:7], 'grip:', line[7:])

        except tf.errors.OutOfRangeError:
            train_logger.error('Batch out of range!')
            # print('BatchOutOfRange')
            exit()
        
        im_loss_sum += total_im_loss
        im_loss_avg = im_loss_sum / i

        if i % _PRINT_STEP == 0:
            train_logger.info("{} -> epoch: {:0>4d}, gif_pattern: {:4d}, total_im_loss: {:6.2f}, im_loss_avg: {:4.2f}". \
                format(head_str, epoch, i, total_im_loss, im_loss_avg))
            # print("{} -> epoch: {:0>4d}, gif_pattern: {:4d}, total_im_loss: {:6.2f}, im_loss_avg: {:4.2f}". \
            #     format(head_str, epoch, i, total_im_loss, im_loss_avg))

            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
            # summary = tf.Summary()

            if training:
                all_gif_num = epoch * datanums + i
                summary = tf.Summary()
                summary.value.add(tag="[Train] loss multi (every_100_gif)", simple_value=im_loss_avg)
                summary.value.add(tag="[Train] loss (every_100_gif)", simple_value=np.sqrt(im_loss_avg / 100.0))
                summary_writer.add_summary(summary, all_gif_num)

    summary_writer.flush()
    show_use_time(time.time() - start_time, head_str + ' use time:', train_logger)

    return im_loss_avg


def valid_batch(*arg, **kwargs):
    return train_all_batch(*arg, **kwargs, training=False)


logger.info('Start Logging')
# Data Loader
DataLoader.set_logger(build_logger)
train_dlr = DataLoader(_TRAIN_DATA, img_size=cfg['image_height'])
valid_dlr = DataLoader(_VALID_DATA, img_size=cfg['image_height'])

train_data = train_dlr.input_pipeline()
valid_data = valid_dlr.input_pipeline()
is_training = tf.placeholder(dtype=bool,shape=())
gif, fdb, cmd, _ = tf.cond(is_training, lambda:(train_data), lambda:(valid_data))

m = BehaviorClone(logger=build_logger)
m.set_network_property(drop_out=FLAGS.drop_out)
m.build_inputs_and_outputs(tf.squeeze(gif), tf.squeeze(fdb), tf.squeeze(cmd))
m.build_train_op()

build_logger.info('--------- After build graph, get_trainable_dic() ------------')
# print('---------After build graph, get_trainable_dic()------------')
get_trainable_dic()

# limit memory
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True                   # allocate dynamically
# config.gpu_options.per_process_gpu_memory_fraction = 0.8 # maximun alloc gpu50% of MEM

with tf.Session() as sess:
    start_ep = 0

    # -------restore------
    recreate_dir(FLAGS.log_dir)
    model_file = tf.train.latest_checkpoint(FLAGS.model_dir)
    saver = tf.train.Saver(max_to_keep=5)
    if model_file is not None:
        build_logger.info('Use model_file = ' + str(model_file) + '!')
        # print('Use model_file = ' + str(model_file) + '!')
        saver.restore(sess, model_file)
        build_logger.info('--------- After build graph, get_trainable_dic() ------------')
        # print('---------After build graph,  get_trainable_dic()------------')
        get_trainable_dic()
        # get ckpt epoch num
        start_ep = int(model_file.rpartition('-')[-1]) + 1
    else:
        build_logger.info('Initialize all variables')
        # print('[I] Initialize all variables')
        sess.run(tf.global_variables_initializer())

        # if memory out check code
        # op = tf.global_variables_initializer()
        # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        # sess.run(op, options=run_options)
        build_logger.info('Initialize all variables Finish')
        # print('[I] Initialize all variables Finish')

    # record start time
    train_start_time = time.time()
    
    try:
        DataLoader.start(sess)

        for ep in range(start_ep, _EPOCHS):
            train_logger.info('----- Train -----')
            # print('---Train-----')
            train_avg_loss = train_all_batch(sess, m, ep, train_dlr.data_nums)
            train_logger.info('----- Valid -----')
            # print('---Valid-----')
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
            # print("epoch: {:0>4d}, epoch_train_loss: {:4.2f},  epoch_valid_loss: {:4.2f}". \
            #         format(ep, epoch_train_loss, epoch_valid_loss))
            save_model(sess, saver, ep, 'epoch: ' + str(ep), g_step=ep)  
            show_use_time(time.time() - train_start_time, "All use time: ", train_logger)
        
    except KeyboardInterrupt:
        train_logger.info('Got Keyboard Interrupt!')
    finally:
        DataLoader.close()
        logger.info('Stop Training and Logging')
        # print('close')

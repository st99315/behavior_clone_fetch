
# Get 99% accuracy from tensorflow tutorial #

import tensorflow as tf
from tensorflow.python.platform import flags

import time
from os.path import join
import imageio
import numpy as np 

from utils import print_all_var, save_train_batch, recreate_dir
from utils import get_recursive_file_name, show_use_time, get_files_in_dir

from load_data import DataLoader
from im_network_one_gif import BehaviorClone


_TRAIN_DATA = '../train_data_same_color_0520/train_data/object_0'
_VALID_DATA = '../train_data_same_color_0520/valid_data/object_0'
_EPOCHS = 1000
_PRINT_STEP = 100

FLAGS = flags.FLAGS
flags.DEFINE_bool('drop_out', False, 'if True, use drop_out for fc(fully connected!')
flags.DEFINE_string('log_dir', 'log/', 'log directory')
flags.DEFINE_string('model_dir', 'checkpoints/', 'model directory')

summary_writer = tf.summary.FileWriter(FLAGS.log_dir)


def get_trainable_dic():
    # print('*******get_trainable_dic*********')
    all_w_b = dict()
    for v in tf.trainable_variables():
        name = v.name.replace(":0", "")
        all_w_b[name] = v
        print('{}->{}'.format(name, v))
    print()
    return all_w_b 


#---------------Load Data------------------#
def load_training_batch(load_path):
    batch_size = 1   # batch_size always 1, feed one gif
    """
    Get           gif array: [batch_size, pic_in_gif, 240, 240, 3] 
    coreesponding action:    [batch_size, pic_in_gif, 2]
    """
    print('in load_training_batch')
    all_gif_names = get_files_in_dir(root_path=load_path, suffix='.gif')
    # print(all_gif_names)
    print('len(all_gif_names) = ' + str(len(all_gif_names)))

    for start in range(0, len(all_gif_names), batch_size):
        end = min(start + batch_size, len(all_gif_names))
        # print('Batch start(%3d) -> end(%3d)' %(start,end))
        batch_gif_names = all_gif_names[start:end]

        # get batch_gifs [batch_size, pic_in_gif, 240, 240, 3] 
        gifs_dict ={}
        gifs_dict = {gif_name: imageio.mimread(gif_name) for gif_name in batch_gif_names}
        # gifs = [imageio.mimread(gif_name)  for gif_name in batch_gif_names]

        che10000ck_all_gif_shape = True
        gifs_rgb = []
        # for g in gifs:
        for name,g in gifs_dict.items():
            g_ary = np.array(g, dtype='f')
            
            check_all_gif_shape = True
            # print('g_ary.shape', g_ary.shape)
            if len(g_ary.shape) < 4:
                # print('Strange g_ary shape')
                print('Strange g_ary shape, name = {}, g_ary.shape = {}'.format(name, g_ary.shape))
                check_all_gif_shape = False
                continue
            try:
                g_ary = g_ary[:,:,:,:3] 
                g_ary[:, :, :, 0] -= 103.939
                g_ary[:, :, :, 1] -= 116.779
                g_ary[:, :, :, 2] -= 123.68
                # g_ary.astype(float32)
                gifs_rgb.append(g_ary)
            except Exception as e: 
                # double check, maybe no need
                print('g_ary = g_ary[:3] Exception is ' + str(e))
                print('g_ary.shape = ' + str(g_ary.shape))
                # print(batch_gif_names)
                print('name = ' + name )
                check_all_gif_shape = False
        #abandon this batch
        if not check_all_gif_shape: 
            print('Abandon this batch from start(%3d) -> end(%3d)' %(start,end))
            continue
    
        batch_gifs_normal = np.array(gifs_rgb)
        # print('batch_gifs_normal.shape = %s ' % str(batch_gifs_normal.shape))

        # get batch_actions [batch_size, pic_in_gif, 2]
        batch_actions = [np.loadtxt(
                join(load_path, '{}.csv'.format(gif_name.split('/')[-1].split('.')[0]))
            ) for gif_name in batch_gif_names
        ]
        batch_actions = np.array(batch_actions)
        # print('batch_actions.shape = %s ' %  str(batch_actions.shape))

        # [[1,2]]
        # batch_gifs_normal.shape = 1,20, 100,100,3
        if batch_size == 1:
            batch_gifs_normal = np.squeeze(batch_gifs_normal)
            batch_actions = np.squeeze(batch_actions)
     
        yield batch_gifs_normal, batch_actions


def save_model(sess, saver, epoch, show_str, g_step):
    save_model_path = 'checkpoints/'
    # saver.save(sess, save_model_path+'model-{:0>3d}.ckpt'.format(epoch))
    saver.save(sess, save_model_path+'model.ckpt', global_step=g_step)

    print('Save Model! ' + show_str)


def train_all_batch(sess, model, epoch, datanums, name, training=True):

    head_str = "Train" if training else "Valid"
    start_time = time.time()
    
    print('training = {}, head_str={}'.format(training, head_str))

    sum_pattern = 0.
    im_loss_sum = 0.
    im_loss_avg = 0.

    for i in range(1, datanums+1):
        try:
            if training:
                _, total_im_loss, predict, imname = sess.run([model.train_op, model.total_im_loss, model.batch_prediction, name], 
                    feed_dict={is_training: training})
            else:
                total_im_loss, predict, imname = sess.run([model.total_im_loss, model.batch_prediction, name],
                    feed_dict={is_training: training}) 

                if i==1:
                    print(predict)

        except tf.errors.OutOfRangeError:
            print('BatchOutOfRange')
            break
        
        im_loss_sum += total_im_loss
        im_loss_avg = im_loss_sum / i

        if i % _PRINT_STEP == 0: 
            print("{} -> epoch: {:0>4d}, gif_pattern: {:4d}, total_im_loss: {:6.2f}, im_loss_avg: {:4.2f} {}". \
                format(head_str, epoch, i, total_im_loss, im_loss_avg, imname))

            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
            # summary = tf.Summary()

            if training:
                all_gif_num = epoch * datanums + i
                summary = tf.Summary()
                summary.value.add(tag="[Train] loss multi (every_100_gif)", simple_value=im_loss_avg)
                summary.value.add(tag="[Train] loss (every_100_gif)", simple_value=np.sqrt(im_loss_avg / 100.0))
                summary_writer.add_summary(summary, all_gif_num)

    summary_writer.flush()
    show_use_time(time.time() - start_time, head_str + ' use time:')

    return im_loss_avg


def valid_batch(*arg, **kwargs):
    return train_all_batch(*arg, **kwargs, training=False)


# Data Loader
train_dlr = DataLoader(_TRAIN_DATA)
valid_dlr = DataLoader(_VALID_DATA)

train_data = train_dlr.input_pipeline()
valid_data = valid_dlr.input_pipeline()
is_training = tf.placeholder(dtype=bool,shape=())
gif, fdb, cmd, name = tf.cond(is_training, lambda:(train_data), lambda:(valid_data))

m = BehaviorClone()
m.set_network_property(drop_out=FLAGS.drop_out)
m.build_inputs_and_outputs(tf.squeeze(gif), tf.squeeze(fdb), tf.squeeze(cmd))
m.build_train_op()

print('---------After build graph,  get_trainable_dic()------------')
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
        print('Use model_file = ' + str(model_file) + '!' )
        saver.restore(sess, model_file)
        print('---------After build graph,  get_trainable_dic()------------')
        get_trainable_dic()
        start_ep = int(model_file.rpartition('-')[-1]) + 1
    else:
        print('[I] Initialize all variables')
        sess.run(tf.global_variables_initializer())

        # if memory out check code
        # op = tf.global_variables_initializer()
        # run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        # sess.run(op, options=run_options)
        print('[I] Initialize all variables Finish')

    # record start time
    train_start_time = time.time()
    
    try:
        DataLoader.start(sess)

        for ep in range(start_ep, _EPOCHS):
            print('---Train-----')
            train_avg_loss = train_all_batch(sess, m, ep, train_dlr.data_nums, name)
            print('---Valid-----')
            valid_avg_loss = valid_batch(sess, m, ep, valid_dlr.data_nums, name)

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

            print("epoch: {:0>4d}, epoch_train_loss: {:4.2f},  epoch_valid_loss: {:4.2f}". \
                    format(ep, epoch_train_loss, epoch_valid_loss))
            save_model(sess, saver, ep, 'epoch: ' + str(ep), g_step=ep)  
            show_use_time(time.time() - train_start_time, "All use time: " )
        
    except KeyboardInterrupt:
        pass
    finally:
        DataLoader.close()
        print('close')

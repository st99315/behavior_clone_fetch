import tensorflow as tf
import random
import numpy as np


def print_all_var():
    print('---------tf.trainable_variables()---------')
    train_var = tf.trainable_variables()
    for v in train_var:
        print(v)
        # print(v.name)

    print('---------tf.GLOBAL_VARIABLES()---------')
    get_collection_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in get_collection_var:
        print(v)

    print('---------get all Placeholder---------')
    for x in tf.get_default_graph().get_operations() :
        if  "Placeholder" in x.type:
            print('name = {}'.format( x.name ) )
            tensor = tf.get_default_graph().get_tensor_by_name(x.name+":0")
            print('shape = {}'.format(str(tensor.shape)))

    print('---------Relu---------')
    for x in tf.get_default_graph().get_operations() :
        if  "Relu" in x.type:
            print('name = {}'.format( x.name ) )
            tensor = tf.get_default_graph().get_tensor_by_name(x.name+":0")
            print('shape = {}'.format(str(tensor.shape)))

    # print('---------tf.LOCAL_VARIABLES()---------')
    # get_collection_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
    # for v in get_collection_var:
    #     print(v)

    # print('---------tf.LOCAL_RESOURCES()---------')
    # get_collection_var = tf.get_collection(tf.GraphKeys.LOCAL_RESOURCES)
    # for v in get_collection_var:
    #     print(v)

    # print('---------tf.GLOBAL_STEP()---------')
    # get_collection_var = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
    # for v in get_collection_var:
    #     print(v)


def create_dir_not_exist(tar_dir):
    import os
    if not os.path.isdir(tar_dir):
        os.mkdir(tar_dir)
        print('create directory -> ' + tar_dir)


def recreate_dir(tar_dir):
    from shutil import rmtree
    import os
    if os.path.isdir(tar_dir):
        rmtree(tar_dir)
    os.mkdir(tar_dir)
    print('recreate directory -> ' + tar_dir)


def save_train_batch(epoch, batch_ind,batch_imgs, batch_pos, mimic_batch_imgs, mimic_batch_pos, 
        batch_prediction, im_prediction):
    tar_dir = 'log_train_batch/epoch_{:0>3d}_batch_{:0>4d}/'.format(epoch,batch_ind)
    recreate_dir(tar_dir)

    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    im_height = 240

    np.savetxt(tar_dir + '/_target_pos.out', batch_pos, fmt='%3d') 
    np.savetxt(tar_dir + '/_mimic_pos.out', mimic_batch_pos, fmt='%3d') 
    np.savetxt(tar_dir + '/_predict_mimic_pos.out', batch_prediction, fmt='%3d') 
    np.savetxt(tar_dir + '/_im_prediction.out', im_prediction, fmt='%3d') 
    
    # print('batch_imgs.shape = ' + str(batch_imgs.shape))
    # print('batch_imgs.shape[0] = ' + str(batch_imgs.shape[0]))
    for i in range(batch_imgs.shape[0]):

        batch_img = batch_imgs[i] * 255.0
        mimic_img = mimic_batch_imgs[i] * 255.0

        # cv2.imwrite(tar_dir + '/' + str(i) + '_ori.png', batch_img)
        # cv2.imwrite(tar_dir + '/' + str(i) + '_mimic_ori.png', mimic_img)

        from scipy.misc import imsave, imread
        imsave(tar_dir + '/' + str(i) + '_ori.png', batch_img)
        imsave(tar_dir + '/' + str(i) + '_mimic_ori.png', mimic_img)

        cv2.circle(batch_img,( int(batch_pos[i][0]), int(im_height- batch_pos[i][1])), 5, (255,255,255),-1)
        cv2.circle(mimic_img,(int(mimic_batch_pos[i][0]), int(im_height-mimic_batch_pos[i][1])), 5, (255,255,255),-1)
        cv2.circle(mimic_img,(int(batch_prediction[i][0]), int(im_height-batch_prediction[i][1])), 5, (0,255,0),-1)
        
        img_txt ='Target: ({:0>3.0f},{:0>3.0f})'.format(batch_pos[i][0], batch_pos[i][1])
        cv2.putText(batch_img,img_txt ,(10,10), font, 0.4,(255,255,255),2,cv2.LINE_AA)
        img_txt ='Target: ({:0>3.0f},{:0>3.0f})'.format(mimic_batch_pos[i][0], mimic_batch_pos[i][1])
        cv2.putText(mimic_img,img_txt ,(10,10), font, 0.4,(255,255,255),2,cv2.LINE_AA)
        img_txt ='Predict: ({:0>3.0f},{:0>3.0f})'.format(batch_prediction[i][0], batch_prediction[i][1])
        cv2.putText(mimic_img,img_txt ,(10,40), font, 0.4,(255,255,255),2,cv2.LINE_AA)
        
        cv2.imwrite(tar_dir + '/' + str(i) + '.png', batch_img)
        cv2.imwrite(tar_dir + '/' + str(i) + '_mimic.png', mimic_img)

        # print('Save to ' + tar_dir + '/.png')
        # print('Save to ' + tar_dir + '/1_mimic.png')
        

def v_trapezoid(t_now = 10,  t_all = 20, s = 150):
    '''
    speed use trapezoid
    (20 + 10 )* high_v / 2= 150 -> high_v =  150*2 / ( 20 + 10) 
    '''
    # half  timefor high speed
    l_r_len = t_all * 0.5 * 0.5   # trapezoid left & right len
    high_v = s * 2 / (t_all + t_all*0.5)

    if t_now < l_r_len:           # < 5
        return (t_now+1) * high_v / l_r_len
    elif t_now < (l_r_len + t_all * 0.5):
        return high_v
    elif t_now < (l_r_len + t_all * 0.5 + l_r_len):
        return (t_all - t_now -1) * high_v / l_r_len
    else:
        print('v_trapezoid() say Error t_now = ' + str(t_now))
        return 0


def get_files_in_dir(root_path='train_data/', suffix='.gif',shuffle=True):
    from os import listdir
    from os.path import isfile, join

    dir_files = [join(root_path, f) for f in listdir(root_path) if isfile(join(root_path, f)) and f.endswith(suffix)]

    if shuffle:
        random.shuffle(dir_files)
    return dir_files


def get_recursive_file_name(root_path='train_data/', suffix='.gif',start = 0, end = 400, shuffle = True):
    from os import listdir
    from os.path import isfile, join

    all_files = []
    for i in range(start, end):
        check_dir =root_path + 'color_%03d/' % i
        dir_files = [check_dir + f for f in listdir(check_dir) if isfile(join(check_dir, f)) and f.endswith(suffix)]
        # print(dir_files)

        all_files.extend(dir_files)
    
    if shuffle:
        random.shuffle(all_files)
    return all_files
    # print(all_files)
    # print('no. all files : '+ str( len(all_files)  ))


def show_use_time(total_time, head_str='Use time:', logger=None):
    h, m, s = int(total_time // 3600), int(total_time // 60 % 60), int(total_time % 60)
    if logger:
        logger.info('{} {:0>3d}h{:0>2d}m{:0>2d}s'.format(head_str, h, m, s))
    else:
        print('{} {:0>3d}h{:0>2d}m{:0>2d}s'.format(head_str, h, m, s))
    

def set_logger(filename='training.log', log_dir='./log'):
    ''' Logging Levels
        CRITICAL 50
        ERROR    40
        WARNING  30
        INFO     20
        DEBUG    10
        NOTSET	 0
    '''
    import os
    import logging

    create_dir_not_exist(log_dir)

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-10s %(levelname)-7s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(log_dir, filename),
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(levelname)-7s %(message)s')

    # tell the handler to use this format
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Now, define a couple of other loggers which might represent areas in your
    # application:
    logger1 = logging.getLogger('build_step')
    logger2 = logging.getLogger('train_step')
    return logging, logger1, logger2

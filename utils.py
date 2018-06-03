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


def show_use_time(total_time, head_str='Use time:', logger=None):
    h, m, s = int(total_time // 3600), int(total_time // 60 % 60), int(total_time % 60)
    if logger:
        logger.info('{} {:0>3d}h{:0>2d}m{:0>2d}s'.format(head_str, h, m, s))
    else:
        print('{} {:0>3d}h{:0>2d}m{:0>2d}s'.format(head_str, h, m, s))
    

def set_logger(logger_names=[], filename='training.log', log_dir='./log'):
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
                        format='%(asctime)s %(name)-10s [%(levelname)-7s] %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(log_dir, filename),
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-10s [%(levelname)-7s] %(message)s')

    # tell the handler to use this format
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Now, define a couple of other loggers which might represent areas in your
    # application:
    loggerlists = [logging]
    for name in logger_names:
        loggerlists.append(logging.getLogger(name))

    ''' Return logger lists
            first is basic logger (root)
            otherwise are requesting logger from logger_names
    '''
    return loggerlists

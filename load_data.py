"""
    Using TF Queue to Load Training Data
        - Volatile GPU-Util become 2x

    Modify from https://github.com/tianheyu927/mil
"""

import os
import tensorflow as tf
import glob
import random
import config
import numpy as np


def get_gifs_mean(path):
    import yaml
    try:
        with open(path) as file:
            content = yaml.load(file)
            mean = content['all_gifs_mean']
            mean_list = [mean['r'], mean['g'], mean['b']]
    except FileNotFoundError:
        # ImageNet or VGG dataset RGB mean
        mean_list = [123.68, 103.939, 116.779]
    finally:
        print('Load Mean:', mean_list)
        return np.array(mean_list)


class DataLoader:
    """ Need Set Logger """
    _MAX_LEN = 300
    _HEAD_LINE = 1
    _NUM_THREAD = 1
    _coord, _threads, _logger = None, None, None

    def __init__(self, directory, img_size=128, csv_cols=21, load_num=None):
        self._GIF_SHAPE = (None, img_size, img_size, 3)
        self._CSV_COLS  = csv_cols

        self.coord = None
        all_gifs, all_csvs = self.get_all_filenames(directory, shuffle=True, size=load_num)
        self.data_num = len(all_gifs)

        DataLoader._logger.info('All data: {}'.format(self.data_num))
        assert self.data_num is not 0, DataLoader._logger.error('No data loaded!')
        
        self.gif_names = tf.convert_to_tensor(all_gifs)
        self.csv_names = tf.convert_to_tensor(all_csvs)

        yaml_path = os.path.join(directory, directory.rpartition('/')[-1] + '.yaml')
        rgb_mean = get_gifs_mean(yaml_path)
        DataLoader._logger.debug('RGB mean of dataset, r: {}, g: {}, b: {}' \
            .format(rgb_mean[0], rgb_mean[1], rgb_mean[2]))
        self.data_mean = tf.convert_to_tensor(rgb_mean, tf.float32)

    @property
    def data_nums(self):
        return self.data_num

    def get_all_filenames(self, dir, shuffle=False, size=None):
        # finf all of folder
        folders = glob.glob(os.path.join(dir, 'object*'))
        # get all of gifs
        gifs = []
        for folder in folders:
            gifs.append(glob.glob(os.path.join(folder, '*.gif')))
        gifs = np.concatenate(gifs)
        # shuffle list
        if shuffle: random.shuffle(gifs)
        # get all of csvs
        csvs = []
        for name in gifs:
            csvs.append(name.rpartition('.')[0]+'.csv')
        return gifs[:size], csvs[:size]

    def _image_preprosess(self, image):
        image.set_shape(self._GIF_SHAPE)
        image = tf.cast(image, tf.float32)
        # normalize
        image -= self.data_mean
        image /= 255.

        # Subtract off the mean and divide by the variance of the pixels.
        # image = tf.image.per_image_standardization(image)

        return image

    def _read_gif_format(self, filename_queue):
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        images = tf.image.decode_gif(value)
        images = self._image_preprosess(images)
        return key, images

    def _read_csv_format(self, filename_queue):
        reader = tf.TextLineReader(skip_header_lines=self._HEAD_LINE)
        key, value = reader.read_up_to(filename_queue, self._MAX_LEN)
        record_defaults = [[0.] for i in range(self._CSV_COLS)]
        features = tf.decode_csv(value, record_defaults=record_defaults,
            field_delim=' ')
        # transpose tensor
        features = tf.transpose(features, [1, 0])
        # slice features to feedback and command
        feedback_num = config.cfg['robot_configuration_num']
        fdb = features[..., :feedback_num]
        cmd = features[..., feedback_num:]
        return key, fdb, cmd

    def input_pipeline(self, batch_size=1, num_epochs=None):
        min_after_dequeue = 16
        # capacity 一定要比 min_after_dequeue 更大一些，
        # 多出來的部分可用於預先載入資料，建議值為：
        # min_after_dequeue + (num_threads + a small safety margin) * batch_size
        capacity = min_after_dequeue + 3 * batch_size

        gifnames_queue = tf.train.string_input_producer(self.gif_names, num_epochs=num_epochs, shuffle=False, capacity=capacity)
        csvnames_queue = tf.train.string_input_producer(self.csv_names, num_epochs=num_epochs, shuffle=False, capacity=capacity)
        name, image = self._read_gif_format(gifnames_queue)
        _, fdb, cmd = self._read_csv_format(csvnames_queue)

        image_batch, fdb_batch, cmd_batch, names = tf.train.batch(
            [image, fdb, cmd, name],
            batch_size=batch_size,
            capacity=capacity,
            dynamic_pad=True,
            num_threads=self._NUM_THREAD
        )
        # name_batch, image_batch = tf.train.shuffle_batch([name, image], batch_size, capacity, min_after_dequeue)
        return image_batch, fdb_batch, cmd_batch, names

    @staticmethod
    def set_logger(logger):
        if logger is None:
            import utils
            logger = utils.set_logger()
        DataLoader._logger = logger

    @staticmethod
    def start(sess):
        if DataLoader._coord is not None: return
        # Required to get the filename matching to run.
        tf.local_variables_initializer().run()
        # Coordinate the loading of image files.
        DataLoader._coord   = tf.train.Coordinator()
        DataLoader._threads = tf.train.start_queue_runners(sess=sess, coord=DataLoader._coord, start=True)
        DataLoader._logger.info('Start loader')

    @staticmethod
    def close():
        if DataLoader._coord is None: return
        # Finish off the filename queue coordinator.
        DataLoader._coord.request_stop()
        DataLoader._coord.join(DataLoader._threads)
        DataLoader._coord, DataLoader._threads = None, None
        DataLoader._logger.info('Close loader')

    @staticmethod
    def should_stop():
        return DataLoader._coord.should_stop()


def _exe_tfop(sess, dlr, tfop, is_training, train):
    print('------------ {} ------------'.format('train' if train else 'valid'))
    for i in range(dlr.data_nums):
        # Get an image tensor and print its value.
        images, feedbacks, commands, names = sess.run(tfop, feed_dict={is_training: train})
        if not (i % 10):
            print(i, images, feedbacks, commands, names)


def _load_step():
    _TRAIN_DIRECTORY = '../train_data_same_color_0516/train_data/object_0'
    _VALID_DIRECTORY = '../train_data_same_color_0516/valid_data/object_0'
    _EPOCHS = 5

    DataLoader.set_logger(None)
    train_dlr = DataLoader(_TRAIN_DIRECTORY)
    valid_dlr = DataLoader(_VALID_DIRECTORY)

    # make data tensors
    is_training = tf.placeholder(dtype=bool, shape=(), name='istrain')

    train_data = train_dlr.input_pipeline()
    valid_data = valid_dlr.input_pipeline()
    img, fdb, cmd, name = tf.cond(is_training, lambda:(train_data), lambda:(valid_data))

    # tensor operation
    train_op = [tf.shape(img), tf.shape(fdb), tf.shape(cmd), name]

    # Start a new session to show example output.
    with tf.Session() as sess:
        DataLoader.start(sess)
        for ep in range(_EPOCHS):
            print('------------ epoch {} ------------'.format(ep))
            try:
                _exe_tfop(sess, train_dlr, train_op, is_training, True)
                _exe_tfop(sess, valid_dlr, train_op, is_training, False)
            except tf.errors.OutOfRangeError:
                print('[W] Loader: Out of Range')
                break
        DataLoader.close()


if __name__ == '__main__':
    tf.reset_default_graph()

    import time
    start = time.time()

    try:
        _load_step()
    except KeyboardInterrupt:
        pass

    end = time.time()
    print('total time', end - start)

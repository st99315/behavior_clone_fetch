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
    _MAX_LEN  = 300
    _CSV_COLS = 21
    _HEAD_LINE  = 1
    _NUM_THREAD = 1
    _coord, _threads, _logger = None, None, None

    def __init__(self, directory, load_num=None):
        self.initial(directory, load_num)

    def initial(self, directory, load_num=None):
        all_gifs, all_exts, all_csvs = self.get_all_filenames(directory, shuffle=True, size=load_num)
        self.data_num = len(all_gifs)

        DataLoader._logger.info('All data: {}'.format(self.data_num))
        assert self.data_num is not 0, DataLoader._logger.error('No data loaded!')
        
        self.gif_names = tf.convert_to_tensor(all_gifs)
        self.ext_names = tf.convert_to_tensor(all_exts)
        self.csv_names = tf.convert_to_tensor(all_csvs)
        self.set_parameter()
        self.get_mean(directory)

    def get_all_filenames(self, dir, shuffle=False, size=None):
        # finf all of folder
        folders = glob.glob(os.path.join(dir, 'object*'))
        # get all of gifs
        gifs = []
        for folder in folders:
            gifs.append(glob.glob(os.path.join(folder, '*-g.gif')))
        gifs = np.concatenate(gifs)
        # shuffle list
        if shuffle: random.shuffle(gifs)

        # get all of exts and csvs
        exts, csvs = [], []
        for name in gifs:
            filename = name.rpartition('.')[0].rpartition('-')[0]
            exts.append(filename+'-e.gif')
            csvs.append(filename+'.csv')

        return gifs[:size], exts[:size], csvs[:size]

    def set_parameter(self, img_size=256, csv_cols=21):
        self._GIF_SHAPE = (None, img_size, img_size, 3)
        self._CSV_COLS  = csv_cols

    def get_mean(self, directory):
        yaml_path = os.path.join(directory, directory.rpartition('/')[-1] + '.yaml')
        rgb_mean = get_gifs_mean(yaml_path)
        DataLoader._logger.debug('RGB mean, r: {}, g: {}, b: {}' \
            .format(rgb_mean[0], rgb_mean[1], rgb_mean[2]))
        self.data_mean = tf.convert_to_tensor(rgb_mean, tf.float32)

    def _image_preprosess(self, image):
        image = tf.cast(image, tf.float32)
        # normalize
        image -= self.data_mean
        image /= 255.
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
        extnames_queue = tf.train.string_input_producer(self.ext_names, num_epochs=num_epochs, shuffle=False, capacity=capacity)
        csvnames_queue = tf.train.string_input_producer(self.csv_names, num_epochs=num_epochs, shuffle=False, capacity=capacity)
        gn, image_g = self._read_gif_format(gifnames_queue)
        en, image_e = self._read_gif_format(extnames_queue)
        _, fdb, cmd = self._read_csv_format(csvnames_queue)
        image_g.set_shape((None, 256, 256, 3))
        image_e.set_shape((None, 128, 128, 3))

        batch = tf.train.batch(
            [image_g, image_e, fdb, cmd],
            batch_size=batch_size,
            capacity=capacity,
            dynamic_pad=True,
            num_threads=self._NUM_THREAD
        )
        return batch

    @property
    def data_nums(self):
        return self.data_num

    @staticmethod
    def set_logger(logger):
        if logger is None:
            import utils
            logger = utils.set_logger()[0]
        DataLoader._logger = logger

    @staticmethod
    def start(sess):
        print(DataLoader._coord)
        if DataLoader._coord is not None: #return
            print('to close')
            DataLoader.close()
        # Required to get the filename matching to run.
        tf.local_variables_initializer().run()
        # Coordinate the loading of image files.
        DataLoader._coord   = tf.train.Coordinator()
        DataLoader._threads = tf.train.start_queue_runners(sess=sess, coord=DataLoader._coord, start=True)
        DataLoader._logger.info('Start loader')

    @staticmethod
    def close():
        print(DataLoader._coord)
        if DataLoader._coord is None: return
        # Finish off the filename queue coordinator.
        DataLoader._coord.request_stop()
        DataLoader._coord.join(DataLoader._threads)
        DataLoader._coord, DataLoader._threads = None, None
        DataLoader._logger.info('Close loader')

    @staticmethod
    def should_stop():
        return DataLoader._coord.should_stop()


class DataLoaderTwo(DataLoader):
    def __init__(self, directory, load_num=None):
        self.initial(directory, load_num)

    def initial(self, directory, load_num=None):
        data_csvs = self.get_all_datalines(directory, shuffle=True, size=load_num)
        self.data_num = len(data_csvs)

        DataLoader._logger.info('All Csv files: {}'.format(self.data_num))
        assert self.data_num is not 0, DataLoader._logger.error('No data loaded!')
        
        self.data_csvs = tf.convert_to_tensor(data_csvs)
        self.get_mean(directory)

    def get_all_datalines(self, dir, shuffle=False, size=None):
        # finf all of csv for data pairs
        data_csvs = glob.glob(os.path.join(dir, 'object*.csv'))
        return data_csvs[:size]

    def _read_gif_format(self, filename_queue, clip):
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        images = tf.image.decode_gif(value)
        # slice images to range
        images = images[clip[0]:clip[1], ...]
        images = self._image_preprosess(images)
        return key, images

    def _read_csv_format(self, filename_queue, clip):
        reader = tf.TextLineReader(skip_header_lines=self._HEAD_LINE)
        key, value = reader.read_up_to(filename_queue, self._MAX_LEN)
        record_defaults = [[0.] for i in range(self._CSV_COLS)]
        features = tf.decode_csv(value, record_defaults=record_defaults,
            field_delim=' ')
        # transpose tensor
        features = tf.transpose(features, [1, 0])
        # slice features to feedback and command
        feedback_num = config.cfg['robot_configuration_num']
        key = key[clip[0]:clip[1]]
        fdb = features[clip[0]:clip[1], ..., :feedback_num]
        cmd = features[clip[0]:clip[1], ..., feedback_num:]
        return key, fdb, cmd
    
    def get_queue(self, num_epochs=1):
        meta_queue = tf.train.string_input_producer(self.data_csvs, num_epochs=num_epochs, shuffle=False)
        return meta_queue

    def input_pipeline(self, meta_queue, batch_size=32):
        min_after_dequeue = 16
        # capacity 一定要比 min_after_dequeue 更大一些，
        # 多出來的部分可用於預先載入資料，建議值為：
        # min_after_dequeue + (num_threads + a small safety margin) * batch_size
        capacity = min_after_dequeue + 3 * batch_size

        # read meta csv
        reader = tf.TextLineReader(skip_header_lines=self._HEAD_LINE)
        key, value = reader.read(meta_queue)
        record = [[0.], [0.], [""], [""], [""]]
        datapairs = tf.decode_csv(value, record_defaults=record, field_delim=' ')
    
        # read gif
        gif_queue = tf.train.string_input_producer([datapairs[2]], shuffle=False)
        name, gif = self._read_gif_format(gif_queue, tf.cast(datapairs[:2], tf.int32))
        gif.set_shape([4, 256, 256, 3])
        
        # read extra gif
        ext_queue = tf.train.string_input_producer([datapairs[3]], shuffle=False)
        name, ext = self._read_gif_format(ext_queue, tf.cast(datapairs[:2], tf.int32))
        ext.set_shape([4, 128, 128, 3])
        
        # read trajectory csv
        tra_queue = tf.train.string_input_producer([datapairs[-1]], shuffle=False)
        _, fdb, cmd = self._read_csv_format(tra_queue, tf.cast(datapairs[:2], tf.int32))
        fdb.set_shape([4, 11])
        cmd.set_shape([4, 10])

        batch_batch = [gif, ext, fdb, cmd, name]
        # batch_batch = tf.train.shuffle_batch(
        #     [gif, ext, fdb, cmd, name], batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        return batch_batch


def _exe_tfop(sess, dlr, tfop, is_training, train):
    print('------------ {} ------------'.format('train' if train else 'valid'))
    for i in range(dlr.data_nums):
        # Get an image tensor and print its value.
        images, images2, feedbacks, commands, names = sess.run(tfop, feed_dict={is_training: train})
        # if not (i % 10):
        #     print(i, images, images2, feedbacks, commands, names)
        if images[0] != 4:
            print(i, images, images2)
        if images2[0] != 4:
            print(i, images, images2)

def _load_step():
    import time

    _DATASET_DIR = '../train_data_diff_color_0531/'
    _TRAIN_DATA = _DATASET_DIR + 'train_data'
    _VALID_DATA = _DATASET_DIR + 'valid_data'
    _EPOCHS = 5

    DataLoader.set_logger(None)
    train_dlr = DataLoaderTwo(_TRAIN_DATA)
    valid_dlr = DataLoaderTwo(_VALID_DATA)

    # make data tensors
    is_training = tf.placeholder(tf.bool, shape=None, name="is_training")
    q_selector = tf.cond(is_training,
                         lambda: tf.constant(0),
                         lambda: tf.constant(1))
    meta_queue = tf.QueueBase.from_list(q_selector, [train_dlr.get_queue(), valid_dlr.get_queue()])
    
    train_data = train_dlr.input_pipeline(meta_queue)
    valid_data = valid_dlr.input_pipeline(meta_queue)
    # gif, ext, fdb, cmd, name = tf.cond(is_training, lambda:(train_data), lambda:(valid_data))
    gif, ext, fdb, cmd, name = train_data

    # tensor operation
    train_op = [tf.shape(gif), tf.shape(ext), tf.shape(fdb), tf.shape(cmd), name]

    # Start a new session to show example output.
    with tf.Session() as sess:
        DataLoader.start(sess)

        for ep in range(_EPOCHS):
            print('------------ epoch {} ------------'.format(ep))
            try:
                while True:
                    _exe_tfop(sess, train_dlr, train_op, is_training, True)
            except tf.errors.OutOfRangeError:
                print('[W] Train Loader: Out of Range')

            try:
                while True:
                    _exe_tfop(sess, valid_dlr, train_op, is_training, False)
            except tf.errors.OutOfRangeError:
                print('[W] Valid Loader: Out of Range')

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

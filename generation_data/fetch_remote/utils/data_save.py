"""
    Saving Training Data
"""

import os
import imageio
import numpy as np
import tensorflow as tf
from copy import deepcopy as dcopy


def check_dir(dir):
    if os.path.exists(dir): return
    try:
        os.makedirs(dir)
    except FileExistsError:
        print('File existed! No Create Folder.')


class DataSaver:
    _INFO_HEADER    = 'Object(X, Y, Z), Gosl(X, Y, Z)'
    _TRAJECT_HEADER = (
        'Feedback: EE(X, Y, Z), Joint(0, 1, 2, 3, 4, 5, 6), GripperState(Dense), ' +
        'Command: EE velocity (X, Y, Z), GripperCommand, ' +
        'Auxiliary: Object(X, Y, Z), EE(X, Y, Z)')
    _DATA_HEADER = '4PicsIdx(S, E), Gif path(Maybe2), Csv path'

    def __init__(self, directory, info=False):
        self.dir = directory
        check_dir(self.dir)
        self._images = []
        self._extra_images = []
        self._trajectories = []
        self.header = self._INFO_HEADER if info else self._TRAJECT_HEADER

    def append(self, image=None, extra_img=None, trajectory=None):
        if image is not None:
            self._images.append(dcopy(image))
        if extra_img is not None:
            self._extra_images.append(dcopy(extra_img))
        if trajectory is not None:
            self._trajectories.append(dcopy(trajectory))

    def _save_gif(self, img_buffer, file_name, clip):
        if not len(img_buffer):   return
        imgs = img_buffer[clip[0]: clip[1]]
        path = os.path.join(self.dir, file_name)
        imageio.mimsave(path, imgs, format='GIF-PIL')
        # set path
        if img_buffer is self._images:
            self.gif_path = path
        else:
            self.ext_path = path
    
    def _save_tra(self, file_name, clip):
        if not len(self._trajectories):   return
        self._trajectories = self._trajectories[clip[0]: clip[1]]
        self.tra_path = os.path.join(self.dir, file_name)
        np.savetxt(self.tra_path, self._trajectories, delimiter=' ', header=self.header)
        
    def _save_slc(self, epsoide, slice_num=4):
        ''' Slice one epsoide to many and save to csv
                unsave if gif/trajectory is empty or length of them are not equal
        '''
        gif_len, tra_len = len(self._images), len(self._trajectories)
        if (gif_len != tra_len) and not (gif_len and tra_len):   return

        slice_dir = self.dir.rpartition('/')
        file_name = os.path.join(slice_dir[0], '{}.csv'.format(slice_dir[-1]))
        header = '' if os.path.exists(file_name) else self._DATA_HEADER
        with open(file_name, 'a') as _file:
            total = gif_len - slice_num + 1
            index = [(i, i+4) for i in range(total)]

            # set data and format
            if len(self._extra_images):
                data  = [(s, e, self.gif_path, self.ext_path, self.tra_path) for s, e in index]
                npdtype = [('si', 'i4'), ('ei', 'i4'), ('gif', 'U100'), ('ext', 'U100'), ('csv', 'U100')]
                fmt='%i %i %s %s %s'
            else:
                data  = [(s, e, self.gif_path, self.tra_path) for s, e in index]
                npdtype = [('si', 'i4'), ('ei', 'i4'), ('gif', 'U100'), ('csv', 'U100')]
                fmt='%i %i %s %s'
            npdata = np.array(data, dtype=npdtype)
            np.savetxt(_file, npdata, delimiter=' ', fmt=fmt, header=header)

    def save(self, epsoide, clip=(0, None)):
        ''' saving data (buffer) to desired directory
                and clear buffer
            epsoide is int variable
         '''
        self._save_gif(self._images,       '{}-g.gif'.format(epsoide), clip)
        self._save_gif(self._extra_images, '{}-e.gif'.format(epsoide), clip)
        self._save_tra('{}.csv'.format(epsoide), clip)
        # self._save_slc(epsoide)
        self.record_data(epsoide)
        self.flush()

    def flush(self):
        # clear buffer
        self._images = []
        self._extra_images = []
        self._trajectories = []

    def open_tf_writer(self):
        self.pattern = 0
        rcfilename = os.path.join(self.dir, 'data1.tfrecords')
        # 設定以 gzip 壓縮
        compression = tf.python_io.TFRecordCompressionType.GZIP
        self.writer = tf.python_io.TFRecordWriter(rcfilename,
            options=tf.python_io.TFRecordOptions(compression))
        # self.writer = tf.python_io.TFRecordWriter(rcfilename)

    def close_tf_writer(self):
        print('dataset total pattern', self.pattern)
        self.writer.close()

    def record_data(self, epsoide, slice_num=4):
        gif_len, tra_len = len(self._images), len(self._trajectories)
        if (gif_len != tra_len) or not (gif_len and tra_len):   return

        total = gif_len - slice_num + 1
        index = [(i, i+4) for i in range(total)]

        gif = np.array(self._images,       dtype=np.float32)
        ext = np.array(self._extra_images, dtype=np.float32)
        csv = np.array(self._trajectories, dtype=np.float32)

        def byte_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        def float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        for si, ei in index:
            gifslice = gif[si:ei]
            extslice = ext[si:ei]
            csvslice = csv[si:ei]

            # make feature
            feature = {}
            for i in range(slice_num):
                feature["external_{}".format(i)] = byte_feature(value=tf.compat.as_bytes(gifslice[i].tostring()))
                feature["eye_hand_{}".format(i)] = byte_feature(value=tf.compat.as_bytes(extslice[i].tostring()))
                feature["fdb_cmd_{}".format(i)] = float_feature(value=csvslice[i].flatten())

            # unit batch data
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            self.writer.write(example.SerializeToString())
            self.pattern += 1

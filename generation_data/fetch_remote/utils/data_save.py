"""
    Saving Training Data
"""

import os
import imageio
import numpy as np


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
            self._images.append(image)
        if extra_img is not None:
            self._extra_images.append(extra_img)
        if trajectory is not None:
            self._trajectories.append(trajectory)

    def _save_gif(self, img_buffer, file_name, clip):
        if not len(img_buffer):   return
        imgs = img_buffer[clip[0]: clip[1]]
        path = os.path.join(self.dir, file_name)
        imageio.mimsave(path, imgs)
        # set path
        if img_buffer is self._images:
            self.gif_path  = path
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
        self._save_gif(self._images,       '{}-g.gif'.format(epsoide),   clip)
        self._save_gif(self._extra_images, '{}-e.gif'.format(epsoide), clip)
        self._save_tra('{}.csv'.format(epsoide), clip)
        self._save_slc(epsoide)
        self.flush()

    def flush(self):
        # clear buffer
        self._images = []
        self._extra_images = []
        self._trajectories = []

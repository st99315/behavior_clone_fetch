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
        'Feedback: EE(X, Y, Z), Joint(0, 1, 2, 3, 4, 5, 6), PastCommand(X, Y, Z, G), ' +
        'Command: EE velocity (X, Y, Z), GripperCommand' +
        'Auxiliary: Object(X, Y, Z), EE(X, Y, Z)')

    def __init__(self, directory, info=False):
        self.dir = directory
        check_dir(self.dir)
        self._images = []
        self._trajectories = []
        self.header = self._INFO_HEADER if info else self._TRAJECT_HEADER

    def append(self, image=None, trajectory=None):
        if image is not None:
            self._images.append(image)
        if trajectory is not None:
            self._trajectories.append(trajectory)

    def _save_gif(self, file_name):
        if not len(self._images):   return
        file_path = os.path.join(self.dir, file_name)
        imageio.mimsave(file_path, self._images)
        self._images = []
    
    def _save_tra(self, file_name):
        if not len(self._trajectories):   return
        file_path = os.path.join(self.dir, file_name)
        np.savetxt(file_path, self._trajectories, delimiter=' ', header=self.header)
        self._trajectories = []

    def save(self, name):
        # saving data (buffer) to desired directory
        # and clear buffer
        self._save_gif('{}.gif'.format(name))
        self._save_tra('{}.csv'.format(name))


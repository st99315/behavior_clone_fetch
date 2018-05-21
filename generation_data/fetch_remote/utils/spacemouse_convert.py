""" 
    Converting value to (x, y, z, roll, pitch, yaw) from SpaceNavigator Mouse 
    Reqeust Module: https://github.com/micropsi-industries/py3dcnx
"""

import py3dcnx


class Convert:
    _MAX_VAL = 350.

    def __init__(self):
        self._value = {'x': 0., 'y': 0., 'z': 0., 'roll': 0., 'pitch': 0., 'yaw': 0., 'grip': 1}
        self._reset = 0

        self.sm = py3dcnx.SpaceMouse()
        for etype in py3dcnx.event_types:
            self.sm.register_handler(self.event, etype)

    def normalize(self, val):
        return val / self._MAX_VAL

    def event(self, event):
        if event['type'] == 'button':
            if event['val'] == 1:
                self._value['grip'] = -self._value['grip'] if event['val'] else self._value['grip']
            elif event['val'] == 2:
                self._reset = 1
        else:
            for key in event.keys():
                if key != 'type':
                    self._value[key] = self.normalize(event[key])

    def get_val(self):
        return (self._value['x'], -self._value['y'], -self._value['z'],
            self._value['pitch'], self._value['roll'], self._value['yaw'], self._value['grip'])

    def is_reset(self):
        reset = self._reset
        self._reset = 0
        return reset


if __name__ == '__main__':
    import time

    cvt = Convert()
    while True:
        print(cvt.get_val())
        time.sleep(.1)
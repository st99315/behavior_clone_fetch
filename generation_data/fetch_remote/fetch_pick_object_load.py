"""
    Load Training Data (Joint Command) and Run it 
"""

import os
import gym
from gym.envs.robotics import FetchPickAndPlaceJointEnv
import numpy as np
from matplotlib import pyplot as plt
import fetch_remote.utils as utils
from fetch_remote.utils.spacemouse_convert import Convert
import glob


GRIPPER_STATE = 1
SCALE_SPEED = 2.0
DEMO_TIMES = 10

GYM_PATH = gym.__path__[0]
XML_PATH = os.path.join(GYM_PATH, 'envs/robotics/assets/fetch/myenvs/banded_0002_lacelike_0121.xml')
CVS_PATH = '/home/iclab/youjun/remote/data/object_0/301.csv'


def get_all_xml():
    all_xmls = glob.glob(os.path.join(XML_DIR, '*.xml'))
    print('envs', len(all_xmls))
    return all_xmls


def get_joint_data():
    data = np.loadtxt(CVS_PATH, delimiter=' ')
    # this step need to fix
    return data[:, 3:]


args = utils.get_args()
utils.set_env_variable(args.display)

env = FetchPickAndPlaceJointEnv(xml_file=XML_PATH)
joint_data = get_joint_data()

for i in range(DEMO_TIMES):
    obs = env.reset()
    total_reward = 0

    for joint_ctrl in joint_data:
        print('we', obs['weneed']*180./np.pi)

        a = np.array(joint_ctrl)
        obs, r, done, info = env.step(a)

        total_reward += r

        if args.display:
            env.render()
        else:
            rgb_obs = env.sim.render(width=200, height=200, camera_name="external_camera_0", depth=False,
                mode='offscreen', device_id=-1)
            plt.imshow(rgb_obs)
            plt.show(block=False)
            plt.pause(0.001)

        if info['is_success'] or done:
            print(i, "total reward %0.2f" % total_reward)
            break


"""
    Running Simple Policy in Fetch Enviroment
"""

import os
import gym
import numpy as np
from matplotlib import pyplot as plt
import fetch_remote.utils as utils
from fetch_remote.utils.finite_state_machine import FSM
from gym.envs.robotics import FetchPickAndPlaceEnv


args = utils.get_args()
utils.set_env_variable(args.display)

GRIPPER_STATE = 1
LIMIT_Z = .415
SCALE_SPEED = 2.0

env = FetchPickAndPlaceEnv(xml_file='fetch/myenvs/banded_0002_lacelike_0121.xml')

for i in range(10):
    obs = env.reset()
    simple_policy = FSM(np.append(obs['eeinfo'][0], GRIPPER_STATE), obs['achieved_goal'], obs['desired_goal'], LIMIT_Z)
    total_reward = 0

    while not simple_policy.is_done:
        x, y, z, g = simple_policy.execute()
        # scale up action
        a = np.array([x, y, z, g]) * SCALE_SPEED
        obs, r, done, info = env.step(a)

        # update robot state
        simple_policy.robot_state = np.append(obs['eeinfo'][0], g)
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


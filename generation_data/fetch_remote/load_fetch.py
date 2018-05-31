"""
    Using SpaceNavigator Mouse to Control Fetch Robot in Simulation
"""


import gym
from gym.envs.robotics import FetchPickAndPlaceEnv
import numpy as np
from matplotlib import pyplot as plt
try:
    import fetch_remote.utils as utils
    from fetch_remote.utils.spacemouse_convert import Convert
except ImportError:
    import utils
    from utils.spacemouse_convert import Convert


args = utils.get_args()
utils.set_env_variable(args.display)

env = FetchPickAndPlaceEnv()
cvt = Convert()

try: 
    while True:
        obs = env.reset(rand_text=True, rand_shadow=True)

        for i in range(10000):
            x, y, z, pi, ro, ya, g = cvt.get_val()
            if cvt.is_reset():
                break

            a = np.array([x, y, z, g])
            obs, r, done, info = env.step(a)
            # print('gripper state', g, obs['gripper_state'])

            if args.display:
                env.render()
            elif i % 20 == 0:
                rgb_obs = env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
                rgb_obs1 = env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
                    mode='offscreen', device_id=-1)
                plt.figure(1)
                plt.imshow(rgb_obs)
                plt.figure(2)
                plt.imshow(rgb_obs1)
                plt.show(block=False)
                plt.pause(0.001)

except KeyboardInterrupt:
    pass


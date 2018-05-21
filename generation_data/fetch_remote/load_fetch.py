"""
    Using SpaceNavigator Mouse to Control Fetch Robot in Simulation
"""


import gym
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

env = gym.envs.make('FetchPickAndPlace-v1')
cvt = Convert()

try: 
    while True:
        env.reset()

        while True:
            x, y, z, pi, ro, ya, g = cvt.get_val()
            if cvt.is_reset():
                break

            a = np.array([x, y, z, g])
            obs, r, done, info = env.step(a)
            print('gripper state', g, obs['gripper_state'])

            if args.display:
                env.render()
            else:
                rgb_obs = env.env.sim.render(width=500, height=500, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
                plt.imshow(rgb_obs)
                plt.show(block=False)
                plt.pause(0.001)

except KeyboardInterrupt:
    pass


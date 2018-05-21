"""
    A Simple Policy for Grasp Object using Fetch Robot
"""

from gym.envs.robotics.fetch_env import goal_distance

class FSM:
    _DIS_ERROR      = 0.005
    _PREGRIP_HEIGHT = 0.1
    fsm_state = ('idle', 'go_obj', 'down', 'grip', 'up', 'go_goal')

    def __init__(self, robot_state, obj_pos, goal_pos, limit_z):
        self.state = self.fsm_state[0]
        self.next_state = self.state
        self.start = False
        self._done = False
        self.robot_state = robot_state
        self.obj_pos = obj_pos.copy()
        self.goal_pos = goal_pos.copy()
        self.limit_z = limit_z  # limit ee low height

    @property
    def done(self):
        done, self._done = self._done, False
        return done

    @property
    def robot_state(self):
        return self._robot_state

    @robot_state.setter
    def robot_state(self, robot_state):
        assert robot_state.shape == (4,)
        self._robot_state = robot_state.copy()

    def execute(self):
        x, y, z, g = 0., 0., 0., 0
        if self.state == 'idle':
            if not self.start:
                self.next_state = 'go_obj'
                self.start = True
            # output
            x, y, z, g = 0., 0., 0., self.robot_state[-1]
        elif self.state == 'go_obj':
            self.next_state = 'down'
            # output
            x, y, z = self.obj_pos - self.robot_state[:3]
            z += self._PREGRIP_HEIGHT
            g = self.robot_state[-1]
        elif self.state == 'down':
            self.next_state = 'grip'
            # output
            if self.obj_pos[2] <= self.limit_z:
                self.obj_pos[2] = self.limit_z
            x, y, z = self.obj_pos - self.robot_state[:3]
            g = self.robot_state[-1]
        elif self.state == 'grip':
            self.next_state = 'up'
            # output
            x, y, z, g = 0., 0., 0., -1
        elif self.state == 'up':
            self.next_state = 'go_goal'
            # calculate target pos (up to object)
            self.tar_pos = self.obj_pos.copy()
            self.tar_pos[2] += self._PREGRIP_HEIGHT
            # output
            x, y, z = self.tar_pos - self.robot_state[:3]
            g = self.robot_state[-1]
        elif self.state == 'go_goal':
            # self.next_state = 'idle'
            # output
            x, y, z = self.goal_pos - self.robot_state[:3]
            g = self.robot_state[-1]

        self.wait_robot()
        return x, y, z, g
            
    def wait_robot(self):
        if self.state == 'go_obj':
            if goal_distance(self.robot_state[:2], self.obj_pos[:2]) > self._DIS_ERROR:
                # print(goal_distance(self.robot_state[:2], self.obj_pos[:2]))
                return
        elif self.state == 'down':
            if goal_distance(self.robot_state[:3], self.obj_pos) > self._DIS_ERROR:
                return
        # Done!!!
        elif self.state == 'up':
            if goal_distance(self.robot_state[:3], self.tar_pos) > self._DIS_ERROR:
                return
            # TODO: Revise this appoarch to change goal pos
            self._done = True
        elif self.state == 'go_goal':
            if goal_distance(self.robot_state[:3], self.goal_pos) > self._DIS_ERROR:
                return
        elif self.state == 'grip':
            if self.robot_state[-1] >= -.5:
                return
        self.state = self.next_state


if __name__ == '__main__':
    """
        Running Simple Policy in Fetch Enviroment
    """

    import os
    import gym
    from gym.envs.robotics import FetchPickAndPlaceEnv
    import numpy as np
    from matplotlib import pyplot as plt
    try:
        import fetch_remote.utils as utils
    except ImportError:
        import utils


    GRIPPER_STATE = 1
    LIMIT_Z = .415
    SCALE_SPEED = 2.0


    args = utils.get_args()
    utils.set_env_variable(args.display)

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

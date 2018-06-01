"""
    A Simple Policy for Grasp Object using Fetch Robot
"""

from gym.envs.robotics.fetch_env import goal_distance

class FSM:
    _DIS_ERROR      = 0.01
    _PREGRIP_HEIGHT = 0.1
    fsm_state = ('idle', 'go_obj', 'down', 'grip', 'up', 'go_goal')

    def __init__(self, robot_state, obj_pos, goal_pos, limit_z=.415, step=50, skip_step=2):
        self.state = self.fsm_state[0]
        self.next_state = self.state
        # every task costs steps
        self._every_task = []
        self._step = 0
        self.maxstep = step
        self._done = False
        self.robot_state = robot_state
        self.obj_pos = obj_pos.copy()
        self.goal_pos = goal_pos.copy()
        self.limit_z = limit_z  # limit ee low height
        self.skip_step = skip_step

    @property
    def done(self):
        done, self._done = self._done, False
        return done

    @property
    def step(self):
        ''' Finished task spend step, Current task spend step '''
        return self._every_task, self._step

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
            self.next_state = 'go_obj'
            # output
            x, y, z, g = 0., 0., 0., 1.
        elif self.state == 'go_obj':
            self.next_state = 'down'
            # output
            x, y, z = self.obj_pos - self.robot_state[:3]
            z += self._PREGRIP_HEIGHT
            g = 1.
        elif self.state == 'down':
            self.next_state = 'grip'
            # output
            if self.obj_pos[2] <= self.limit_z:
                self.obj_pos[2] = self.limit_z
            x, y, z = self.obj_pos - self.robot_state[:3]
            g = 1.
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
            g = -1
        elif self.state == 'go_goal':
            # self.next_state = 'idle'
            # output
            x, y, z = self.goal_pos - self.robot_state[:3]
            g = -1
        
        if self._step > self.maxstep:
            self._done = True
            return x, y, z, g

        self._step += 1
        self.wait_robot()
        return x, y, z, g
            
    def wait_robot(self):
        if self.state == 'idle':
            if self._step < self.skip_step:
                return
        if self.state == 'go_obj':
            if goal_distance(self.robot_state[:2], self.obj_pos[:2]) > self._DIS_ERROR:
                return
        elif self.state == 'down':
            if (goal_distance(self.robot_state[:2], self.obj_pos[:2]) > self._DIS_ERROR
                or self.robot_state[2] > self.obj_pos[2] + self._DIS_ERROR/2.0
                or self.robot_state[2] < self.obj_pos[2] - self._DIS_ERROR/2.0):
                return
        elif self.state == 'up':
            if goal_distance(self.robot_state[:3], self.tar_pos) > self._DIS_ERROR:
                return
        # Done!!!
        elif self.state == 'go_goal':
            if goal_distance(self.robot_state[:3], self.goal_pos) > self._DIS_ERROR:
                return
            self._done = True
        elif self.state == 'grip':
            if self._step < self.skip_step or self.robot_state[-1] >= -.5:
                return

        self.state = self.next_state
        self._every_task.append(self._step)
        self._step = 0


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

    env = FetchPickAndPlaceEnv()

    for i in range(args.start, args.end):
        obs = env.reset(rand_text=args.random, rand_shadow=args.random)
        g = GRIPPER_STATE

        goal = obs['achieved_goal'].copy()
        goal[-1] = goal[-1] + .1
        simple_policy = FSM(np.append(obs['eeinfo'][0], g), obs['achieved_goal'], goal, LIMIT_Z)
        total_reward = 0

        step = 0
        while not simple_policy.done:
            x, y, z, g = simple_policy.execute()
            # scale up action
            a = np.array([x, y, z, g]) * SCALE_SPEED
            obs, r, done, info = env.step(a)
            # update robot state
            simple_policy.robot_state = np.append(obs['eeinfo'][0], g)
            
            step += 1
            total_reward += r

            if args.display:
                env.render()
            elif step % 20 == 0:
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

            if info['is_success'] or done:
                print('done')
                break

        print(i, "total reward %0.2f" % total_reward)

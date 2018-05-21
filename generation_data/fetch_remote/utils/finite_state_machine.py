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
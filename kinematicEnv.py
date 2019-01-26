import numpy as np
import pyglet
import time

class KinematicEnv():
    viewer = None
    dt = 0.1
    state_dim = 12
    action_dim = 5
    action_bound = [-1, 1]
    def __init__(self):
        #位姿初始化
        self.STATE = 0  #0表示两端都在杆上; 1表示下端在杆上，执行stepUp(); 2表示上端在杆上，执行stepDown()
        self.on_goal = 0 #判断当前动作是否结束
        self.width = 40.
        self.pole = 100.
        self.crank = 50.
        self.downPointLocation = np.array([0., 0.])
        self.downPointAngle = np.pi / 6   #相对于X轴
        self.downJointLocation = np.array([self.pole*np.cos(self.downPointAngle), self.pole*np.sin(self.downPointAngle)]).reshape(1,2)[0]
        self.downJointAngle = 2 * np.pi / 3  #相对于下端连杆
        self.centerLocation = np.array([(self.pole - self.crank) * np.cos(self.downPointAngle), (self.pole + self.crank) * np.sin(self.downPointAngle)]).reshape(1,2)[0]
        self.centerAngle = 4 * np.pi / 3    #相对于下端曲柄
        self.upJointLocation = np.array([self.pole * np.cos(self.downPointAngle), (self.pole + 2 * self.crank) * np.sin(self.downPointAngle)]).reshape(1,2)[0]
        self.upJointAngle = 2 * np.pi / 3  #相对于上端曲柄
        self.upPointLocation = np.array([0., (2 * self.pole + 2 * self.crank) * np.sin(self.downPointAngle)]).reshape(1,2)[0]
        self.upPointAngle = 11 * np.pi / 6  #相对于X轴
        self.armState =  np.concatenate((self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation, self.upPointLocation))
        self.jointState = np.array([self.downPointAngle, self.downJointAngle, self.centerAngle, self.upJointAngle, self.upPointAngle])

    def A(self, theta, l):
        '''广义变换矩阵，先旋转再平移，由相对于新的坐标系变换，需右乘  Rot(z, theta).dot(Trans(l, 0., 0.))'''
        result = np.array([[np.cos(theta), np.sin(theta), 0., l*np.cos(theta)],
                           [np.sin(theta), -np.cos(theta), 0., l*np.sin(theta)],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
        return result

    def thetaTrans(self, theta):
        return 2 * np.pi - theta #关节的相对角度改变，从相对于下端改变到相对于上端


    def stepUp(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.jointState += action * self.dt
        self.jointState = self.jointState % (2*np.pi)  # normalize
        self.downPointAngle, self.downJointAngle, self.centerAngle, self.upJointAngle, self.upPointAngle = self.jointState
        self.jointState[4] = (np.pi + self.downPointAngle + self.downJointAngle + self.centerAngle + self.upJointAngle) % (2*np.pi)
        A1 = self.A(self.downPointAngle, self.pole)
        A2 = self.A(self.downJointAngle, self.crank)
        A3 = self.A(self.centerAngle, self.crank)
        A4 = self.A(self.upJointAngle, self.pole)
        self.downJointLocation = self.downPointLocation + (A1.dot(np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        self.centerLocation = self.downPointLocation + (A1.dot(A2).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        self.upJointLocation = self.downPointLocation + (A1.dot(A2).dot(A3).dot(np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1,4)[0][0:2]
        self.upPointLocation = self.downPointLocation + (A1.dot(A2).dot(A3).dot(A4).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]

        s = np.concatenate((self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation, self.upPointLocation, [1.], [1. if self.on_goal else 0.]))
        r = - np.sqrt((self.upPointLocation[0] / 300.) ** 2 + ((self.upPointLocation[1] - self.downPointLocation[1] - 200) / 300.) ** 2)
        if - self.width / 2 < self.upPointLocation[0] < self.width / 2:
            if - self.width / 2 < self.upPointLocation[1] - self.downPointLocation[1] - 200 < self.width / 2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 3: #while training ddpg, it should be 50
                    done = True
        else:
            self.on_goal = 0
        self.armState = np.concatenate(
            (self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation,
             self.upPointLocation))
        #print(self.armState)


        return s, r, done


    def stepDown(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.jointState += action * self.dt
        self.jointState %= 2*np.pi  # normalize
        self.downPointAngle, self.downJointAngle, self.centerAngle, self.upJointAngle, self.upPointAngle = self.jointState
        self.jointState[0] = (7 * np.pi + self.upPointAngle - self.downJointAngle - self.centerAngle - self.upJointAngle) % (2*np.pi)
        #正运动学求解关节位置
        A1 = self.A(self.thetaTrans(self.upPointAngle), self.pole)
        A2 = self.A(self.thetaTrans(self.upJointAngle), self.crank)
        A3 = self.A(self.thetaTrans(self.centerAngle), self.crank)
        A4 = self.A(self.thetaTrans(self.downJointAngle), self.pole)
        self.upJointLocation = self.upPointLocation + (A1.dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1,4)[0][0:2]
        self.centerLocation = self.upPointLocation + (A1.dot(A2).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1,4)[0][0:2]
        self.downJointLocation = self.upPointLocation + (A1.dot(A2.dot(A3)).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1,4)[0][0:2]
        self.downPointLocation = self.upPointLocation + (A1.dot(A2.dot(A3.dot(A4))).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1,4)[0][0:2]

        s = np.concatenate((self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation,
                            self.upPointLocation, [2.], [1. if self.on_goal else 0.]))
        r = - 2 * np.sqrt((self.downPointLocation[0]/300.) ** 2 + ((self.upPointLocation[1] - self.downPointLocation[1] - 100)/300)**2)
        if - self.width / 2 < self.downPointLocation[0] < self.width / 2:
            if - self.width / 2 < self.upPointLocation[1] - self.downPointLocation[1] - 100 < self.width / 2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 3:  #while training ddpg, it is set as 50
                    done = True
        else:
            self.on_goal = 0
        self.armState = np.concatenate(
            (self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation,
             self.upPointLocation))
        return s, r, done

    def initialOn(self):
        self.downPointLocation = np.array([np.random.rand(1)[0]*40. - 20., np.random.rand(1)[0]*500.])
        self.upPointLocation = np.array([np.random.rand(1)[0]*40. - 20., self.downPointLocation[1] + np.random.rand(1)[0]*200. + 50.])
        #逆运动学求解关节角度
        from scipy.optimize import fsolve
        def f(k):
            a, b, c, d = k.tolist()
            A1 = self.A(a, self.pole)
            A2 = self.A(b, self.crank)
            A3 = self.A(c, self.crank)
            A4 = self.A(d, self.pole)
            return [(A1.dot(A2.dot(A3.dot(A4))).dot(np.concatenate((np.array([0., 0.]), np.array([0., 1.])))).reshape(4, 1))[0][0] - self.upPointLocation[0] + self.downPointLocation[0],
                    (A1.dot(A2.dot(A3.dot(A4))).dot(np.concatenate((np.array([0., 0.]), np.array([0., 1.])))).reshape(4, 1))[1][0] - self.upPointLocation[1] + self.downPointLocation[1],
                    (A1.dot(A2.dot(A3.dot(A4))).dot(np.concatenate((np.array([0., 0.]), np.array([0., 1.])))).reshape(4, 1))[2][0],
                    (A1.dot(A2.dot(A3.dot(A4))).dot(np.concatenate((np.array([0., 0.]), np.array([0., 1.])))).reshape(4, 1))[3][0] - 1.
                    ]
        result = fsolve(f, [1., 1., 1., 1.])
        self.downPointAngle, self.downJointAngle, self.centerAngle, self.upJointAngle = result
        self.upPointAngle = (np.pi + self.downPointAngle + self.downJointAngle + self.centerAngle + self.upJointAngle) % (2*np.pi)
        self.jointState = np.array([self.downPointAngle, self.downJointAngle, self.centerAngle, self.upJointAngle, self.upPointAngle])
        #正运动学求解关节位置
        A1 = self.A(self.downPointAngle, self.pole)
        A2 = self.A(self.downJointAngle, self.crank)
        A3 = self.A(self.centerAngle, self.crank)
        A4 = self.A(self.upJointAngle, self.pole)
        self.downJointLocation = self.downPointLocation + (A1.dot(np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        self.centerLocation = self.downPointLocation + (A1.dot(A2).dot(np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        self.upJointLocation = self.downPointLocation + (A1.dot(A2.dot(A3)).dot(np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        s = np.concatenate((self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation, self.upPointLocation,[0.], [1.]))
        self.armState = np.concatenate((self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation,
                            self.upPointLocation))
        return s

    def initialUp(self):
        self.downPointLocation = np.array([np.random.rand(1)[0]*40. - 20., np.random.rand(1)[0] * 500.])
        self.downPointAngle, self.downJointAngle, self.centerAngle, self.upJointAngle = np.random.rand(4) * 2 * np.pi
        self.upPointAngle = (
                            np.pi + self.downPointAngle + self.downJointAngle + self.centerAngle + self.upJointAngle) % (
                            2 * np.pi)
        self.jointState = np.array(
            [self.downPointAngle, self.downJointAngle, self.centerAngle, self.upJointAngle, self.upPointAngle])
        # 正运动学求解关节位置
        A1 = self.A(self.downPointAngle, self.pole)
        A2 = self.A(self.downJointAngle, self.crank)
        A3 = self.A(self.centerAngle, self.crank)
        A4 = self.A(self.upJointAngle, self.pole)
        self.downJointLocation = self.downPointLocation + (A1.dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        self.centerLocation = self.downPointLocation + (A1.dot(A2).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        self.upJointLocation = self.downPointLocation + (A1.dot(A2.dot(A3)).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        self.upPointLocation = self.downPointLocation + (A1.dot(A2.dot(A3.dot(A4))).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        s = np.concatenate((self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation,
                            self.upPointLocation, [1.], [0.]))
        self.armState = np.concatenate(
            (self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation,
             self.upPointLocation))
        return s

    def initialDown(self):
        self.upPointLocation = np.array([np.random.rand(1)[0]*40. - 20., np.random.rand(1)[0] * 700.])
        self.downJointAngle, self.centerAngle, self.upJointAngle, self.upPointAngle = np.random.rand(4) * 2 * np.pi
        self.downPointAngle = (
                             7 * np.pi + self.upPointAngle - self.downJointAngle - self.centerAngle - self.upJointAngle) % (
                             2 * np.pi)
        # 正运动学求解关节位置
        A1 = self.A(self.thetaTrans(self.upPointAngle), self.pole)
        A2 = self.A(self.thetaTrans(self.upJointAngle), self.crank)
        A3 = self.A(self.thetaTrans(self.centerAngle), self.crank)
        A4 = self.A(self.thetaTrans(self.downJointAngle), self.pole)
        self.upJointLocation = self.upPointLocation + (A1.dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        self.centerLocation = self.upPointLocation + (A1.dot(A2).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        self.downJointLocation = self.upPointLocation + (A1.dot(A2.dot(A3)).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        self.downPointLocation = self.upPointLocation + (A1.dot(A2.dot(A3.dot(A4))).dot(
            np.concatenate((np.array([0., 0.]), np.array([0., 1.]))).reshape(4, 1))).reshape(1, 4)[0][0:2]
        s = np.concatenate((self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation, self.upPointLocation, [2.], [0.]))
        self.armState = np.concatenate(
            (self.downPointLocation, self.downJointLocation, self.centerLocation, self.upJointLocation,
             self.upPointLocation))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.armState)
        self.viewer.render(self.armState)

    def random_action(self):
        return np.random.rand(5) - 0.5


class Viewer(pyglet.window.Window):
    def __init__(self, state):
        super(Viewer, self).__init__(width = 800, height = 800, resizable=False, caption='ClimbingArms', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.state = state
        self.batch = pyglet.graphics.Batch()
        self.goal = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [380, 0, 420, 0, 420, 800, 380, 800]), ('c3B', (86, 109, 249) *4))
        self.dPole = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [int(self.state[0])+400, int(self.state[1]), int(self.state[2])+400, int(self.state[3]), int(self.state[2])+395, 5+int(self.state[3]), int(self.state[0])+395, 5+int(self.state[1])]), ('c3B', (246, 86, 86) *4))
        self.dCrank = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [int(self.state[2])+400, int(self.state[3]), int(self.state[4])+400, int(self.state[5]), int(self.state[4])+395, 5+int(self.state[5]), int(self.state[2])+395, 5+int(self.state[3])]), ('c3B', (246, 86, 86) *4))
        self.uCrank = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [int(self.state[4])+400, int(self.state[5]), int(self.state[6])+400, int(self.state[7]), int(self.state[6])+395, 5+int(self.state[7]), int(self.state[4])+395, 5+int(self.state[5])]), ('c3B', (246, 86, 86) *4))
        self.uPole = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [int(self.state[6])+400, int(self.state[7]), int(self.state[8])+400, int(self.state[9]), int(self.state[8])+395, 5+int(self.state[9]), int(self.state[6])+395, 5+int(self.state[7])]), ('c3B', (246, 86, 86) *4))


    def render(self, arm_State):
        self._update(arm_State)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update(self, arm_State):
        state = arm_State.astype(int)
        self.dPole.vertices = [int(state[0]) + 400, int(state[1]), int(state[2]) + 400, int(state[3]), int(state[2]) + 395, 5+int(state[3]), int(state[0]) + 395, 5+int(state[1])]
        self.dCrank.vertices = [int(state[2])+400, int(state[3]), int(state[4])+400, int(state[5]), int(state[4])+395, 5+int(state[5]), int(state[2])+395, 5+int(state[3])]
        self.uCrank.vertices = [int(state[4])+400, int(state[5]), int(state[6])+400, int(state[7]), int(state[6])+395, 5+int(state[7]), int(state[4])+395, 5+int(state[5])]
        self.uPole.vertices = [int(state[6])+400, int(state[7]), int(state[8])+400, int(state[9]), int(state[8])+395, 5+int(state[9]), int(state[6])+395, 5+int(state[7])]


if __name__ == '__main__':
    a = KinematicEnv()
    while True:
        time.sleep(0.1)
        a.render()
        a.stepUp(a.random_action())
from kinematicEnv import KinematicEnv
from bottomDDPG import DDPG
import numpy as np

MAX_EPISODES = 3000
MAX_EP_STEPS = 200
ON_TRAIN = False

# set env
env = KinematicEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound
steps = []
# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)
def initial():
    k = np.random.randint(0, 2)
    if k == 0:
        s = env.initialOn()
    else:
        s = env.initialDown()
    return s


def train():
    # start training
    for i in range(MAX_EPISODES):
        s = initial()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            #env.render()

            a = rl.choose_action(s)

            s_, r, done = env.stepDown(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '----' if not done else 'done', ep_r, j))
                break
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = initial()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.stepDown(a)
        if done:
            s = initial()

if __name__ == '__main__':
    if ON_TRAIN:
        #print(np.array([np.random.rand(1)[0] * 40. - 20., np.random.rand(1)[0] * 700.]))
        train()
    else:
        eval()

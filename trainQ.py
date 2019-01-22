from kinematicEnv import KinematicEnv
from QL import QL
from upDDPG import DDPG as uDDPG
import tensorflow as tf
from bottomDDPG import DDPG as bDDPG
import numpy as np
env = KinematicEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

g1 = tf.Graph()
isess1 = tf.Session(graph=g1)
with g1.as_default():
    isess1.run(tf.global_variables_initializer())
    uddpg = uDDPG(a_dim, s_dim, a_bound)
    uddpg.restore()
g2 = tf.Graph()
isess2 = tf.Session(graph=g2)
with g2.as_default():
    isess2.run(tf.global_variables_initializer())
    bddpg = bDDPG(a_dim, s_dim, a_bound)
    bddpg.restore()
g3 = tf.Graph()
isess3 = tf.Session(graph=g3)
with g3.as_default():
    isess3.run(tf.global_variables_initializer())
    Q = QL(2, s_dim)



def initial():
    tt = np.random.randint(0, 3)
    if tt == 0:
        s = env.initialUp()
    elif tt == 1:
        s = env.initialDown()
    else:
        s = env.initialOn()
    return s

def train():
    step = 0
    for i_episode in range(6000):
        s = initial()
        j = 0
        for i in range(300):
            #env.render()
            a0 = Q.choose_action(s)
            if a0 == 0:
                k = uddpg.choose_action(s)
                s_, _, _ = env.stepUp(k)
            else:
                k = bddpg.choose_action(s)
                s_, _, _ = env.stepDown(k)
            #rewardReset

            label1, label2, label3 = s[0], s[8], s[9] - s[1]
            if -20.<label1<20. and -20.<label2<20.:
                if label3 < 150.:
                    if a0 == 0: reward = 1
                    else: reward = -1
                else:
                    if a0 == 0: reward = -1
                    else: reward = 1
            elif -20.<label1<20. and abs(label2) >= 20.:
                if a0 == 0: reward = 1
                else: reward = -2
            elif abs(label1) >= 20. and -20.<label2<20.:
                if a0 == 0: reward = -2
                else: reward = 1
            Q.store_transition(s, a0, reward, s_)
            if step > 300 and step % 50 == 0:
                Q.learn()
            step+=1
            if reward == 1:
                j += 1
            if reward == -2 or i == 299:
                print('Ep: %i |  accuracy: %.2f | step: %i' % (i_episode, 1.*j/(i+1)*100, i))
                break

    with g3.as_default():
        Q.save()
    #多个计算图训练时，怎么分别存储模型

train()
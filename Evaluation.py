from kinematicEnv import KinematicEnv
from bottomDDPG import DDPG as bDDPG
from upDDPG import DDPG as uDDPG
import numpy as np
import tensorflow as tf
import time
from QL import QL

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
    Q.restore()

def initial():
    k = np.random.randint(0, 3)
    if k == 0:
        s = env.initialUp()
    elif k == 1:
        s = env.initialDown()
    else:
        s = env.initialOn()
    return s

def evalWithHardCode():
    s = initial()
    while True:
        time.sleep(0.05)
        env.render()
        label1, label2, label3 = s[0], s[8], s[9] - s[1]
        if -20. < label1 < 20. and -20. < label2 < 20.:
            if label3 < 150.:
                a = uddpg.choose_action(s)
                s, r, done = env.stepUp(a)
            else:
                a = bddpg.choose_action(s)
                s, r, done = env.stepDown(a)
        elif -20. < label1 < 20. and abs(label2) >= 20.:
            a = uddpg.choose_action(s)
            s, r, done = env.stepUp(a)
        elif abs(label1) >= 20. and -20. < label2 < 20.:
            a = bddpg.choose_action(s)
            s, r, done = env.stepDown(a)
        if s[9] > 800. or s[1] < 0.:
            s = initial()

def evalWithoutLimit():
    s = initial()
    while True:
        time.sleep(0.05)
        env.render()
        a0 = Q.choose_action(s)
        if a0 == 0:
            a = uddpg.choose_action(s)
            s, r, done = env.stepUp(a)
        else:
            a = bddpg.choose_action(s)
            s, r, done = env.stepDown(a)
        if s[9] > 800. or s[1] < 0.:
            s = initial()

def evalAll():
    s = initial()
    while True:
        time.sleep(0.05)
        env.render()
        a0 = Q.choose_action(s)
        if a0 == 0 and -20.<s[0]<20.:
            a = uddpg.choose_action(s)
            s, r, done = env.stepUp(a)
        elif a0 == 1 and -20.<s[8]<20.:
            a = bddpg.choose_action(s)
            s, r, done = env.stepDown(a)
        if s[9] > 800. or s[1] < 0.:
            s = initial()




if __name__ == '__main__':
    evalAll()
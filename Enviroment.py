import tensorflow as tf
import math

class Enviroment():
    def __init__(self):
        self.end_point = tf.constant([46, math.radians(28.6)])
        self.end_norm = tf.norm(self.end_point)
        self.acc = tf.constant(0.00351598)
        self.gain = tf.constant(0.0024262)

    def reset(self):
        radius = 1
        inclination = math.radians(51)
        azimuth = 0
        psi = 0
        beta = 0
        self.state = tf.Variable([azimuth, radius, inclination, psi, beta])
        self.Time = 0
        return self.state, False

    @tf.function
    def transition_step(self, state, action):
        h = tf.constant(0.1)
        az, radius, inclination, psi, beta = tf.unstack(state)
        dpsi = beta
        dbeta = tf.cast(action - 1, float) * self.gain
        daz = (1 / (radius ** 1.5))
        dr = 2 * self.acc * (radius ** 1.5) * tf.cos(psi)
        di = self.acc * tf.sin(psi) * (radius ** 0.5) * tf.cos(az)
        dstate = tf.stack([daz, dr, di, dpsi, dbeta])
        return dstate*h

    def step(self, action):

        done = False

        self.state.assign_add(self.transition_step(self.state, action))

        err_d = tf.norm(self.state[1:3] - self.end_point)

        cos_d = 150*tf.keras.losses.cosine_similarity(self.end_point, self.state[1:3])

        self.Time+=0.1

        if self.Time > 250:
            done = True
        if abs(self.state[3]) > 1:
            done = True


        reward = -cos_d/err_d


        return self.state, reward, done, self.Time
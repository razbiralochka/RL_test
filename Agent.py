import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from collections import deque

loss_object = tf.keras.losses.MeanSquaredError()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


class DQN(Model):
  def __init__(self):
    super().__init__()
    self.d1 = Dense(5, activation='linear')
    self.d2 = Dense(360, activation='relu')
    self.d3 = Dense(500, activation='relu')
    self.d4 = Dense(100, activation='linear')
    self.d5 = Dense(3)

  def call(self, x):

    x = tf.reshape(x, [1, 5])
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    x = self.d5(x)
    return x


class Memory():
    def __init__(self):
        self.replay_memory = deque(maxlen=2000)
    def memorize(self, state, action, state_, reward, done):
        self.replay_memory.append((state, action, state_, reward, done))

    def generate_batch(self):
        if len(self.replay_memory) <= 500:
            mini_batch = self.replay_memory
        else:
            mini_batch = random.sample(self.replay_memory, 500)
        return mini_batch

class Agent():
    def __init__(self):
        self.HAL9000 = DQN()
        self.memory  = Memory()
    @tf.function
    def select_action(self,state):
        predictions = self.HAL9000(state, training=False)
        res = tf.argmax(predictions[0])
        if tf.random.uniform(shape=[], minval=0, maxval=100, dtype=tf.int64) < 10:
            res = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int64)
        return res
    def remember(self,state, action, reward, next_state,done):
        self.memory.memorize(state, action, reward, next_state,done)

    @tf.function
    def train_step(self,state, target):
        with tf.GradientTape() as tape:
            predictions = self.HAL9000(state, training=True)
            loss = loss_object(target, predictions)
        gradients = tape.gradient(loss, self.HAL9000.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.HAL9000.trainable_variables))
        train_loss(loss)
        train_accuracy(target, predictions)

    def learn(self):
        train_loss.reset_states()
        train_accuracy.reset_states()
        mini_batch=self.memory.generate_batch()
        for state, action, state_, reward, done in mini_batch:

            if done:
                target = reward
            else:
                Qmax = tf.experimental.numpy.amax(self.HAL9000(state_))
                target = reward + 0.95*Qmax
            Q = self.HAL9000(state)

            target_f = tf.tensor_scatter_nd_update(Q,[[0, action]],[target])

            self.train_step(state, target_f)
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.initializers import GlorotNormal
from tensorflow_probability import distributions as tfd
from kodex200_5min_test import Kodex200
import matplotlib.pyplot as plt


class ContinuousA2C(tf.keras.Model):
    def __init__(self, action_size):
        super(ContinuousA2C, self).__init__()
        self.actor_fc1 = LSTM(100, activation='relu', return_sequences=True)
        self.actor_fc2 = LSTM(100, activation='relu', return_sequences=False)
        self.actor_mu = Dense(action_size, kernel_initializer=GlorotNormal())
        self.actor_sigma = Dense(action_size, activation='sigmoid', kernel_initializer=GlorotNormal())

        self.critic_fc1 = LSTM(100, activation='relu', return_sequences=True)
        self.critic_fc2 = LSTM(100, activation='relu', return_sequences=False)
        self.critic_out = Dense(1, kernel_initializer=GlorotNormal())

    def call(self, x):
        actor_x = self.actor_fc1(x)
        actor_x = self.actor_fc2(actor_x)
        mu = self.actor_mu(actor_x)
        sigma = self.actor_sigma(actor_x)
        sigma = sigma + 1e-5

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return mu, sigma, value


class AgentPPO:
    def __init__(self, action_size, max_action):
        self.action_size = action_size
        self.max_action = max_action

        self.model = ContinuousA2C(self.action_size)
        self.model.load_weights("./save/model")

    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample(1)
        action = np.clip(action, 0.0001, self.max_action)
        return action


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')

    env = Kodex200(2)
    state_size = env.observation_size
    action_size = env.action_size
    max_action = 0.02

    agent = AgentPPO(action_size, max_action)

    scores, episodes = [], []
    result = np.zeros((76, 5, 10))

    num_episode = 4
    for e in range(num_episode):
        done = False
        score = 0

        sum_volume = 0
        sum_volumeprice = 0
        vwap = np.zeros((76, 1))

        sum_volume_a = 0
        sum_volumeprice_a = 0
        vwap_action = np.zeros((76, 1))

        price_raw = np.zeros((76, 1))
        volume_raw = np.zeros((76, 1))
        predict = np.zeros((76, 1))
        sum_predict = np.zeros((76, 1))

        state = env.reset()
        step = 0
        while not done:

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, 1, state_size])

            # VWAP
            sum_volume += state[0, 0, 3]
            sum_volumeprice += state[0, 0, 3] * state[0, 0, 1]
            vwap[step] = sum_volumeprice / sum_volume

            sum_volume_a += action
            sum_volumeprice_a = action * state[0, 0, 1]
            vwap_action[step] = sum_volumeprice_a / sum_volume_a

            price_raw[step] = state[0, 0, 1]
            volume_raw[step] = state[0, 0, 3]
            predict[step] = action

            score += reward
            state = next_state
            step += 1

            if done:
                print("episode: {:3d} | score: {:3d}".format(e, int(score)))

        result[:, :, e] = np.c_[price_raw, volume_raw, predict, vwap, vwap_action]


    plt.figure(figsize=(10,8))

    plt.subplot(4, 2, 1)
    plt.plot(result[:, 0, 0], label='price')
    plt.plot(result[:, 2, 0], label='order')
    plt.legend(['price', 'order'])

    plt.subplot(4, 2, 2)
    plt.plot(result[:, 3, 0], 'g--', label='vwap')
    plt.plot(result[:, 4, 0], label='vwap_order')
    plt.legend(['vwap', 'vwap_order'])

    plt.subplot(4, 2, 3)
    plt.plot(result[:, 0, 1], label='price')
    plt.plot(result[:, 2, 1], label='order')
    plt.legend(['price', 'order'])

    plt.subplot(4, 2, 4)
    plt.plot(result[:, 3, 1], 'g--', label='vwap')
    plt.plot(result[:, 4, 1], label='vwap_order')
    plt.legend(['vwap', 'vwap_order'])

    plt.subplot(4, 2, 5)
    plt.plot(result[:, 0, 2], label='price')
    plt.plot(result[:, 2, 2], label='order')
    plt.legend(['price', 'order'])

    plt.subplot(4, 2, 6)
    plt.plot(result[:, 3, 2], 'g--', label='vwap')
    plt.plot(result[:, 4, 2], label='vwap_order')
    plt.legend(['vwap', 'vwap_order'])

    plt.subplot(4, 2, 7)
    plt.plot(result[:, 0, 3], label='price')
    plt.plot(result[:, 2, 3], label='order')
    plt.legend(['price', 'order'])

    plt.subplot(4, 2, 8)
    plt.plot(result[:, 3, 3], 'g--', label='vwap')
    plt.plot(result[:, 4, 3], label='vwap_order')
    plt.legend(['vwap', 'vwap_order'])

    plt.savefig('./save/vwap_order.png')
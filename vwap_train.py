import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotNormal
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
from kodex200_5min_train import Kodex200


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

        self.discount_factor = 0.99
        self.learning_rate = 0.0000005
        self.K_epoch = 5
        self.eps_clip = 0.2
        self.lmbda = 0.95

        self.model = ContinuousA2C(self.action_size)
        self.optimizer = Adam(lr=self.learning_rate, clipnorm=1)

    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample(1)
        action_prob = dist.prob(action)
        action = np.clip(action, 0.0001, self.max_action)
        return action, action_prob

    def train_model(self, memory):
        model_params = self.model.trainable_variables

        for i in range(self.K_epoch):
            states = tf.squeeze(tf.convert_to_tensor(memory.states), axis=1)
            actions = tf.squeeze(tf.convert_to_tensor(memory.actions))
            rewards = tf.convert_to_tensor(memory.rewards, dtype=tf.float64)
            dones = tf.convert_to_tensor(memory.dones)
            dones = tf.cast(dones, dtype=tf.float64)
            next_states = tf.squeeze(tf.convert_to_tensor(memory.next_states), axis=1)
            old_action_probs = tf.squeeze(tf.convert_to_tensor(memory.action_probs))

            with tf.GradientTape() as tape:
                mu, sigma, value = self.model(states)
                _, _, next_value = self.model(next_states)

                mu = tf.squeeze(mu)
                sigma = tf.squeeze(sigma)
                value = tf.squeeze(value)
                next_value = tf.squeeze(next_value)

                target = rewards + (1 - dones) * self.discount_factor * next_value
                delta = tf.stop_gradient(target - value)

                advantage_lst = []
                advantage = 0.0
                for delta_t in delta[::-1]:
                    advantage = self.discount_factor * self.lmbda * advantage + delta_t
                    advantage_lst.append(advantage)
                advantage_lst.reverse()
                advantage = tf.convert_to_tensor(advantage_lst)

                # surrogate objective
                dist = tfd.Normal(loc=mu, scale=sigma)
                action_probs = dist.prob(actions)
                ratio = tf.math.exp(tf.math.log(action_probs + 1e-5) - tf.math.log(old_action_probs + 1e-5))
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                actor_loss = - tf.reduce_mean(tf.math.minimum(surr1, surr2))

                critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value)
                critic_loss = tf.reduce_mean(critic_loss)

                entropy = - tf.reduce_mean(action_probs * tf.math.log(action_probs))

                loss = actor_loss + critic_loss + 0.001 * entropy

            grads = tape.gradient(loss, model_params)
            self.optimizer.apply_gradients(zip(grads, model_params))

        return loss.numpy(), np.mean(sigma)


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.action_probs = []

    def store(self, state, action, reward, done, next_state, action_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.action_probs.append(action_prob)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.action_probs = []


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')

    env = Kodex200()
    state_size = env.observation_size
    action_size = env.action_size
    max_action = 0.02
    checkpoint = './save/model'
    savegraph = './save/learning_curve.png'

    agent = AgentPPO(action_size, max_action)
    scores, episodes = [], []
    score_avg = 0

    num_episode = 1000
    for e in range(num_episode):
        done = False
        score = 0
        step = 1
        state = env.reset()
        mem = Memory()

        while not done:
            action, action_prob = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            score += reward
            mem.store(state, action, reward, done, next_state, action_prob)

            state = next_state
            step += 1
            if done:
                loss, sigma = agent.train_model(mem)
                mem.clear()
                del mem

                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print(("episode: {} | score avg: {:.3f} | loss: {:.3f} |"
                       "sigma: {:.3f} | step: {} | remained: {:.3f} | vwap_diff: {:.3f} %").format(
                    e, score_avg, loss, sigma, step, state[0, 0, 5].item(), info))

                scores.append(score_avg)
                episodes.append(e)
                plt.plot(episodes, scores, 'b')
                plt.xlabel("episode")
                plt.ylabel("average score")
                plt.savefig(savegraph)

                if score_avg >= max(scores) or (e % 10 == 0):
                    print('Saved')
                    agent.model.save_weights(checkpoint, save_format="tf")

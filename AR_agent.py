import gym
import numpy as np
import random
import note_exploration
from midi_tm import *

# create note exploration environment
env = gym.make('note_exploration/NoteWorld-v0', size=128, reward_mode="intrinsic")

random_q_table = False

# initialize q-table
state_size = env.size
action_size = env.get_action_size()

# hyperparameters
learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0
decay_rate= 0.005

# training variables
num_episodes = 1000
max_steps = 99 # per episode

if random_q_table: # just fill the q-table with random values

    qtable = np.random.rand(state_size, action_size)

else: # perform some training

    qtable = np.zeros((state_size, action_size))

    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        note_observation = state['note'][0]
        done = False

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0,1) < epsilon:
                # explore
                action = round(env.action_space.sample()[0])
                print("exploration action", action)
            else:
                # exploit
                action = np.argmax(qtable[note_observation,:])
                print("exploit action", action)

            # take action and observe reward
            new_state, reward, done, info = env.step(action)
            new_note_observation = new_state['note'][0]

            print("new_state, reward, info", new_state, reward, info)

            # Q-learning algorithm
            qtable[note_observation,action] = qtable[note_observation,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_note_observation,:])-qtable[note_observation,action])

            print("qtable", qtable)

            # Update to our new state
            state = new_state

        # Decrease epsilon
        epsilon = np.exp(-decay_rate*episode)

print(f"Training completed over {num_episodes} episodes")
input("Press Enter to watch trained agent...")

# watch trained agent
state = env.reset()['note'][0]

env.set_render_mode("human")

done = False
rewards = 0

for s in range(max_steps):

    print(f"TRAINED AGENT")
    print("Step {}".format(s+1))

    action = np.argmax(qtable[state,:])
    print("action:", action)
    new_state, reward, done, info = env.step(action)
    rewards += reward
    env.render()
    print(f"score: {rewards}")
    state = new_state['note'][0]

    if done == True:
        break

env.close()

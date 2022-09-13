import gym
import numpy as np
import random
import note_exploration
from midi_tm import *

# create note exploration environment
env = gym.make('note_exploration/NoteWorld-v0')

# functions to import notes from midi files
#path = 'midi'
#parsed_midi = parse_midi_files(path)
#long_list = get_notes_list(parsed_midi)

# clean data and make transition matrix
#long_list = notes_cleaning(long_list)
#t_matrix, ui = transition_matrix(long_list)

# initialize q-table
state_size = env.size
action_size = env.action_space.n
# qtable = np.zeros((state_size, action_size))
# TODO just to have something going on:
qtable = np.random.rand(state_size, action_size)

# hyperparameters
learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0
decay_rate= 0.005

# training variables
num_episodes = 1000
max_steps = 99 # per episode

# training
for episode in range(num_episodes):

    # reset the environment
    state = env.reset()['agent'][0] # TODO hack!
    done = False

    for s in range(max_steps):

        # exploration-exploitation tradeoff
        if random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(qtable[state,:])

        # take action and observe reward
        new_state, reward, done, info = env.step(action)

        new_state = new_state['agent'][0] # TODO another hack!

        # Q-learning algorithm
        qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

        print("qtable", qtable)

        # Update to our new state
        state = new_state

        # if done, finish episode
        if done == True:
            break

    # Decrease epsilon
    epsilon = np.exp(-decay_rate*episode)

print(f"Training completed over {num_episodes} episodes")
input("Press Enter to watch trained agent...")

# watch trained agent
state = env.reset()['agent'][0] # TODO hack!

env.set_render_mode("human")

done = False
rewards = 0

for s in range(max_steps):

    print(f"TRAINED AGENT")
    print("Step {}".format(s+1))

    action = np.argmax(qtable[state,:])
    new_state, reward, done, info = env.step(action)
    rewards += reward
    env.render()
    print(f"score: {rewards}")
    state = new_state['agent'][0] # TODO hack!

    if done == True:
        break

env.close()

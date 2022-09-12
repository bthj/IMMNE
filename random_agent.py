import gym
import numpy as np
import random
import note_exploration

# create note exploration environment
env = gym.make('note_exploration/NoteWorld-v0', render_mode="human")

# create a new instance of note explorer, and get the initial state
state = env.reset()

num_steps = 99
for s in range(num_steps+1):
    print(f"step: {s} out of {num_steps}")

    # sample a random action from the list of available actions
    action = env.action_space.sample()

    # perform this action on the environment
    env.step(action)

    # print the new state
    env.render()

# end this instance of the taxi environment
env.close()

import gym
from gym import spaces
from gym.utils.renderer import Renderer
import pygame
import numpy as np
import math
from midi_ar import (
    get_note_transition_matrix_prob_and_entropy,
    get_Shannon_entropy_and_update,
    get_Beta_entropy_and_update,
    get_Shannon_KL_and_update,
    get_Dirichlet_KL_and_update
)

# based on https://www.gymlibrary.dev/content/environment_creation/

class NoteWorldEnv(gym.Env):
    metadata = {
        "reward_modes": [
            "extrinsic", # extrinsic rewards solely
            "intrinsic", # intrinsic rewards only
            "oscillate", # oscillation between extrinsic and intrinsic rewards (sinusoid?)
            "extr_to_intr_exp_decay", # decay from one source of reward to the other
            "intr_to_extr_exp_decay" # -- exponentially (what decay constant? -configurable?)
        ],
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 4
    }

    def __init__(
            self, render_mode=None,
            reward_mode="extrinsic",
            oscillation_freq_damping=10, # higher value = lower frequency
            exp_decay_const=1,
            size=12 # 12 notes in an octave
    ):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window

        # Observations are dictionaries with the note's  location.
        self.observation_space = spaces.Dict(
            {
                "note": spaces.Box(0, size - 1, shape=(1,), dtype=int),
                # TODO "velocity"? "duration"? ...
            }
        )

        # Continuous action space (so it hould be usable for outputs from both Q-tables and DQNs):
        # -- corresponding to either one ocatave down or up
        # -- to be later discretised to the nearest note position (in step())
        # TODO: multi-dimensional, for e.g. velocities and durations?
        self.action_space = spaces.Box(0, 24, (1,), dtype=np.float32)


        # Loading / Generating Data

        # "ar" is the transition matrix as an autoregressive model
        # "par" contains transition probabilities
        # "har" represents entropy
        ar, par, har = get_note_transition_matrix_prob_and_entropy()

        self.extrinsic_transition_matrix = ar
        self.extrinsic_probability_matrix = par
        self.entropy_matrix = har


        assert reward_mode in self.metadata["reward_modes"]
        self.reward_mode = reward_mode

        self.oscillation_freq_damping = oscillation_freq_damping
        self.exp_decay_const = exp_decay_const

        self.set_render_mode(render_mode)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        # adding information on the current iteration within a training episodes
        # -- to use while determining what mixture of extrinsic and intrinsic rewards to use (if not fixed)
        self.step_iteration = 0

    def _get_obs(self):
        return {"note": self._agent_location}

    def _get_info(self):
        return {
            "distance": None # TODO: distance to ground truth (transition matrix)?
            # np.linalg.norm(
            #     self._agent_location - self._target_location, ord=1
            # )
        }

    def _action_to_note_delta(self, action):
        """
        Mapping and discretising (regardless of whether necessary or not)
        actions from the range self.action_space.low to self.action_space.high
        """
        action_size = self.get_action_size()
        return round(action - (action_size/2))

    def get_action_size(self):
        return round(self.action_space.high[0] - self.action_space.low[0]) + 1 # TODO: + 1 for a zero crossing?

    def set_render_mode(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._renderer = Renderer(self.render_mode, self._render_frame)

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=1, dtype=int)

        print("---self._agent_location",self._agent_location)
        print("self._agent_location after reset", self._agent_location)

        observation = self._get_obs()
        info = self._get_info()

        self._renderer.reset()
        self._renderer.render_step()

        self.step_iteration = 0

        return (observation, info) if return_info else observation

    def step(self, action):

        note_delta = self._action_to_note_delta(action)

        print("note_delta",note_delta)

        # We use `np.clip` to make sure we don't leave the grid (note pos. array)
        agent_location_after_action = np.clip(
            self._agent_location + note_delta, 0, self.size - 1
        )

        likeliest_action = 0
        note_location_before_action = self._agent_location[0]
        note_location_after_action = agent_location_after_action[0]
        # reward as prob. of performed action as percentage highest prob.
        # TODO: binary sparse rewards instead?
        # -- or rewards according to note-distance between action taken and the "optimal" one?
        match self.reward_mode:
            case "extrinsic": # calculate reward from extrinsic motivation
                reward = self.get_extrinsic_reward(note_location_before_action, note_location_after_action)
            case "intrinsic": # calculate reward from intrinsic motivation
                reward = self.get_intrinsic_reward(note_location_before_action, note_location_after_action)
            case "oscillate":
                extrinsic_reward = self.get_extrinsic_reward(note_location_before_action, note_location_after_action)
                intrinsic_reward = self.get_intrinsic_reward(note_location_before_action, note_location_after_action)

                cycle_pos = abs(math.sin(self.step_iteration / self.oscillation_freq_damping))
                extrinsic_part = 1 - cycle_pos
                intrinsic_part = cycle_pos

                reward = (extrinsic_reward*extrinsic_part) + (intrinsic_reward*intrinsic_part)
                print("oscillating reward", reward)
            case "extr_to_intr_exp_decay":
                extrinsic_reward = self.get_extrinsic_reward(note_location_before_action, note_location_after_action)
                intrinsic_reward = self.get_intrinsic_reward(note_location_before_action, note_location_after_action)

                # TODO

                reward = 0
            case _:
                reward = 0

        self._agent_location = agent_location_after_action

        observation = self._get_obs()
        info = self._get_info()

        done = False # gym requires step() "to return a four or five element tuple"

        self.step_iteration += 1

        self._renderer.render_step()

        return observation, reward, done, info

    def get_extrinsic_reward(self, note_location_before_action, note_location_after_action):
        likeliest_action = np.argmax(self.extrinsic_probability_matrix[note_location_before_action,:])
        highest_prob = self.extrinsic_probability_matrix[note_location_before_action][likeliest_action]
        action_prob = self.extrinsic_probability_matrix[note_location_before_action][note_location_after_action]
        reward = action_prob / highest_prob
        print("extrinsic reward", reward)
        return reward
    def get_intrinsic_reward(self, note_location_before_action, note_location_after_action):
        # likeliest_action = np.argmax(self.entropy_matrix[note_location_before_action,:])
        # highest_prob = self.entropy_matrix[note_location_before_action][likeliest_action]
        # action_prob = self.entropy_matrix[note_location_before_action][note_location_after_action]
        # reward = action_prob / highest_prob

        reward = get_Shannon_entropy_and_update(note_location_before_action, note_location_after_action, self.entropy_matrix)
        # reward = get_Beta_entropy_and_update(note_location_before_action, note_location_after_action, self.entropy_matrix)
        # reward = get_Shannon_KL_and_update(note_location_before_action, note_location_after_action, self.entropy_matrix)
        # reward = get_Dirichlet_KL_and_update(note_location_before_action, note_location_after_action, self.entropy_matrix)

        print("intrinsic reward", reward)
        return reward

    def render(self):
        return self._renderer.get_renders()

    # TODO replace with a musical keyboard
    # TODO play the action-notes (with https://www.pygame.org/docs/ref/music.html or music21 ?)
    def _render_frame(self, mode):
        assert mode is not None

        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, pix_square_size*5))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, pix_square_size*5))
        canvas.fill((255, 255, 255))

        # Now we draw the agent
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                np.append( pix_square_size * self._agent_location, pix_square_size / 2 ),
                (pix_square_size, pix_square_size*5),
            ),
        )

        # Finally, add some gridlines
        for x in range(2):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x*10),
                (self.window_size, pix_square_size * x*10),
                width=3,
            )
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, pix_square_size*5),
                width=3,
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

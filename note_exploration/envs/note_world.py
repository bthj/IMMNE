import gym
from gym import spaces
from gym.utils.renderer import Renderer
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

REWARD_CEILING = 100

class NoteWorldEnv(gym.Env):
    metadata = {
        "reward_modes": [
            "extrinsic", # extrinsic rewards solely
            "intrinsic", # intrinsic rewards only
            "oscillate", # oscillation between extrinsic and intrinsic rewards (sinusoid?)
            "extr_to_intr_exp_decay", # decay from one source of reward to the other
            "intr_to_extr_exp_decay" # -- exponentially (what decay constant? -configurable?)
        ],
        "intrinsic_reward_algorithms": [
            "Shannon_entropy",
            "Shannon_KL",
            "Dirichlet_KL"
        ],
        "render_modes": ["human", "rgb_array", "single_rgb_array", "text"],
        "render_fps": 4
    }

    def __init__(
            self, render_mode=None,
            reward_mode="extrinsic",
            intrinsic_reward_algorithm="Dirichlet_KL",
            oscillation_cycle_period=500,
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
        # self.action_space = spaces.Box(0, 24, (1,), dtype=np.float32)
        self.action_space = spaces.Box(0, size - 1, (1,), dtype=np.float32)


        # Loading / Generating Data

        # "ar" is the transition matrix as an autoregressive model
        # "har" represents entropy
        ar, har = get_note_transition_matrix_prob_and_entropy()

        self.extrinsic_matrix = ar
        self.intrinsic_matrix = har

        dampen_extrinsic_matrix = True
        if dampen_extrinsic_matrix:
            tmpmeans = self.extrinsic_matrix.mean(1)
            tmpidx = np.where(np.eye(128) > 0.1)
            self.extrinsic_matrix[tmpidx] = (
                0.25 * self.extrinsic_matrix[tmpidx] +
                0.75 * tmpmeans
            )

        assert reward_mode in self.metadata["reward_modes"]
        self.reward_mode = reward_mode

        assert intrinsic_reward_algorithm in self.metadata["intrinsic_reward_algorithms"]
        self.intrinsic_reward_algorithm = intrinsic_reward_algorithm

        self.oscillation_cycle_period = oscillation_cycle_period
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
        print("action_size", action_size)
        return round(action - (action_size/2))

    def get_action_size(self):
        return round(self.action_space.high[0] - self.action_space.low[0]) + 1 # TODO: + 1 for a zero crossing?

    def set_render_mode(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode != "text":
            self._renderer = Renderer(self.render_mode, self._render_frame)

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=1, dtype=int)

        print("self._agent_location after reset", self._agent_location)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode != "text":
            self._renderer.reset()
            self._renderer.render_step()

        self.step_iteration = 0

        return (observation, info) if return_info else observation

    def step(self, action):

        # scrapping the note delta thing for now ...which is in any case implemented in a weird way...
        # note_delta = self._action_to_note_delta(action)
        # print("note_delta",note_delta)

        # We use `np.clip` to make sure we don't leave the grid (note pos. array)
        agent_location_after_action = np.clip(
            # scrapping the note delta thing for now ...which is in any case implemented in a weird way...
            # self._agent_location + note_delta, 0, self.size - 1
            [action], 0, self.size - 1
        )

        likeliest_action = 0
        note_location_before_action = self._agent_location[0]
        note_location_after_action = agent_location_after_action[0]
        # reward as prob. of performed action as percentage highest prob.
        # TODO: binary sparse rewards instead?
        # -- or rewards according to note-distance between action taken and the "optimal" one?
        if "extrinsic" == self.reward_mode:
            reward = self.get_extrinsic_reward(note_location_before_action, note_location_after_action)
        elif "intrinsic" == self.reward_mode:
            reward = self.get_intrinsic_reward(note_location_before_action, note_location_after_action)
        elif "oscillate":
            extrinsic_reward = self.get_extrinsic_reward(note_location_before_action, note_location_after_action)
            intrinsic_reward = self.get_intrinsic_reward(note_location_before_action, note_location_after_action)

            t = 2*math.pi * self.step_iteration / self.oscillation_cycle_period
            reward = extrinsic_reward * math.cos(t)**2 + intrinsic_reward * math.sin(t)**2
            print("oscillating reward", reward)
        elif "extr_to_intr_exp_decay":
            extrinsic_reward = self.get_extrinsic_reward(note_location_before_action, note_location_after_action)
            intrinsic_reward = self.get_intrinsic_reward(note_location_before_action, note_location_after_action)

            # TODO

            reward = 0
        else:
            reward = 0

        self._agent_location = agent_location_after_action

        observation = self._get_obs()
        info = self._get_info()

        done = False # gym requires step() "to return a four or five element tuple"

        self.step_iteration += 1

        if self.render_mode != "text":
            self._renderer.render_step()

        return observation, reward, done, info

    def get_extrinsic_reward(self, note_location_before_action, note_location_after_action):
        R = self.extrinsic_matrix
        rcur = R[note_location_before_action, note_location_after_action]
        rden = R[note_location_before_action]
        reward = (rcur+1e-5) / (rden+1e-5).sum()
        # scaling to match the scale of intrinsic rewards (some of which are also scaled)
        # reward = reward
        reward = self.num_to_range(
            reward,
            0, 0.0727, # measured max reward
            0, REWARD_CEILING # range to map to
        )

        return reward

    def get_intrinsic_reward(self, note_location_before_action, note_location_after_action):
        if "Shannon_entropy" == self.intrinsic_reward_algorithm:
            reward = self.num_to_range(
                get_Shannon_entropy_and_update(note_location_before_action, note_location_after_action, self.intrinsic_matrix),
                0, 0.5307, # measured max reward
                0, REWARD_CEILING # range to map to
            )
        elif "Shannon_KL" == self.intrinsic_reward_algorithm:
            reward = self.num_to_range(
                get_Shannon_KL_and_update(note_location_before_action, note_location_after_action, self.intrinsic_matrix),
                0, 0.0142, # measured max reward
                0, REWARD_CEILING # range to map to
            )
        elif "Dirichlet_KL" == self.intrinsic_reward_algorithm:
            # observed maximums ranging from ~761 to ~789
            reward = self.num_to_range(
                get_Dirichlet_KL_and_update(note_location_before_action, note_location_after_action, self.intrinsic_matrix),
                0, 800, # measured max reward
                0, REWARD_CEILING # range to map to
            )
        
        # TODO: Beta_entropy is wonky: always returns exactly the reward: 4.06440323075953 (and takes a long time to run)
        # reward = get_Beta_entropy_and_update(note_location_before_action, note_location_after_action, self.intrinsic_matrix)
        
        reward = abs(reward)
        print("intrinsic reward", reward)
        return reward

    def num_to_range(self, num, inMin, inMax, outMin, outMax):
        return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

    def render(self, mode='human', action=0, reward=0 ):
        if "text" == mode:
            print(f"action={action} reward = {reward}") 
        else:
            return self._renderer.get_renders()

    # TODO replace with a musical keyboard
    # TODO play the action-notes (with https://www.pygame.org/docs/ref/music.html or music21 ?)
    def _render_frame(self, mode):
        import pygame
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
        if self.window is not None and self.render_mode != "text":
            import pygame
            pygame.display.quit()
            pygame.quit()

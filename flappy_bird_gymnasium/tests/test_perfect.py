# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
# Copyright (c) 2024 Akshay Gulabrao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""Tests the simple-observations version of the Flappy Bird environment with a"""

from collections import namedtuple
from math import ceil
import gymnasium
import numpy as np
import pandas as pd
import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv
import logging  # Ensure this import is present

############################ Speed and Acceleration ############################
PIPE_VEL_X = -4

PLAYER_MAX_VEL_Y = 10  # max vel along Y, max descend speed
PLAYER_MIN_VEL_Y = -8  # min vel along Y, max ascend speed

PLAYER_ACC_Y = 1  # players downward acceleration
PLAYER_VEL_ROT = 3  # angular speed

PLAYER_FLAP_ACC = -9  # players speed on flapping
################################################################################

################################## Dimensions ##################################
PLAYER_WIDTH = 34
PLAYER_HEIGHT = 24
PLAYER_PRIVATE_ZONE = (max(PLAYER_WIDTH, PLAYER_HEIGHT) + 30) / 2

LIDAR_MAX_DISTANCE = int(288 * 0.8) - PLAYER_WIDTH

PIPE_WIDTH = 52
PIPE_HEIGHT = 320

BASE_WIDTH = 336
BASE_HEIGHT = 112

BACKGROUND_WIDTH = 288
BACKGROUND_HEIGHT = 512
################################################################################

#: Player's rotation threshold.
PLAYER_ROT_THR = 20

#: Color to fill the surface's background when no background image was loaded.
FILL_BACKGROUND_COLOR = (200, 200, 200)


def play(
    audio_on: bool = True,
    render_mode: str | None = "human",
    use_lidar: bool = False,
) -> None:
    """Plays a game of Flappy Bird with a perfect agent.
    obs = [
        pipes[0][0],  # the last pipe's horizontal position
        pipes[0][1],  # the last top pipe's vertical position
        pipes[0][2],  # the last bottom pipe's vertical position
        pipes[1][0],  # the next pipe's horizontal position
        pipes[1][1],  # the next top pipe's vertical position
        pipes[1][2],  # the next bottom pipe's vertical position
        pipes[2][0],  # the next next pipe's horizontal position
        pipes[2][1],  # the next next top pipe's vertical position
        pipes[2][2],  # the next next bottom pipe's vertical position
        pos_y,  # player's vertical position
        vel_y,  # player's vertical velocity
        rot,  # player's rotation
    ]
    Args:
        audio_on (bool): If `True`, the game will play audio. Defaults to `True`.
        render_mode (Union[str, None]): The render mode. Defaults to `"human"`.
        use_lidar (bool): If `True`, the game will use LIDAR observations. Defaults
            to `False`.
    """
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=audio_on,
        render_mode=render_mode,
        use_lidar=use_lidar,
        debug=True,
        normalize_obs=False,
    )
    states: list[list[float, 12]] = []
    column_names = [
        "pipe_0_x",
        "pipe_0_top",
        "pipe_0_bottom",
        "pipe_1_x",
        "pipe_1_top",
        "pipe_1_bottom",
        "pipe_2_x",
        "pipe_2_top",
        "pipe_2_bottom",
        "player_y",
        "player_y_velocity",
        "player_rotation",
    ]

    FLAP = 1
    IDLE = 0

    def get_action(obs: np.ndarray) -> int:
        """Determines the action based on the current observation to play perfectly."""
        obs = {column_names[i]: obs[i] for i in range(len(column_names))}

        player_x = 57
        if player_x > obs["pipe_0_x"] + PIPE_WIDTH:
            distance_to_pipe_start = obs["pipe_1_x"] - player_x
            distance_to_next_pipe_start = obs["pipe_2_x"] - player_x
            distance_to_pipe_end = obs["pipe_1_x"] + PIPE_WIDTH - player_x
            next_pipe_top = obs["pipe_2_top"]
            current_pipe_bottom = obs["pipe_1_bottom"]
            next_pipe_bottom = obs["pipe_2_bottom"]
        else:
            distance_to_pipe_start = obs["pipe_0_x"] - player_x
            distance_to_pipe_end = obs["pipe_0_x"] + PIPE_WIDTH - player_x
            distance_to_next_pipe_start = obs["pipe_1_x"] - player_x
            next_pipe_top = obs["pipe_1_top"]
            current_pipe_bottom = obs["pipe_0_bottom"]
            next_pipe_bottom = obs["pipe_1_bottom"]
        print(obs["player_y"] + PLAYER_HEIGHT + obs["player_y_velocity"], current_pipe_bottom)
        print(obs["player_y"] - 36, current_pipe_bottom -100)

        if obs["player_y"] + PLAYER_HEIGHT + obs["player_y_velocity"] >= current_pipe_bottom:
            return FLAP
        elif  65 < distance_to_pipe_end < 70:
            if obs["player_y"] - 45 >= current_pipe_bottom -100:
                print("DECISION FLAP")
                return FLAP
            else:
                print("DECISION IDLE")
                return IDLE
        else:
            return IDLE

    obs, info = env.reset()
    while True:
        # Determine the perfect action
        action = get_action(obs)
        obs = obs.tolist() + [info["score"], action]
        states.append(obs)

        # Processing step
        obs, reward, done, terminated, info = env.step(action)



        if done or terminated:
            obs = obs.tolist() + [info["score"], action]
            states.append(obs)
            break

    env.close()
    states = pd.DataFrame(states, columns=column_names + ["score", "action"])
    states.to_csv("perfect_states.csv", index=False)

def test_play() -> None:
    """Tests a game of Flappy Bird with a perfect agent."""
    play(audio_on=False, render_mode="human", use_lidar=False)


if __name__ == "__main__":
    test_play()

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

""" Tests the simple-observations version of the Flappy Bird environment with a
"""


import gymnasium
import numpy as np
import pandas as pd
import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.constants import PLAYER_HEIGHT, BACKGROUND_WIDTH, PIPE_WIDTH, BACKGROUND_HEIGHT


def play(
    audio_on: bool = True,
    render_mode: str|None = "human", 
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
        "FlappyBird-v0", audio_on=audio_on, render_mode=render_mode, use_lidar=use_lidar, debug=True
    )
    states:list[list[float,12]] = []
    column_names = ["pipe_0_x", "pipe_0_top", "pipe_0_bottom", "pipe_1_x", "pipe_1_top", "pipe_1_bottom", "pipe_2_x", "pipe_2_top", "pipe_2_bottom", "player_y", "player_y_velocity", "player_rotation"]

    # Define the bird's fixed horizontal position (assumed based on the environment)
    BIRD_X = 50  # Adjust this value if the bird's position is different in the environment

    def get_action(obs: np.ndarray) -> int:
        """Determines the action based on the current observation to play perfectly."""
        pipe_0_x = obs[0] * BACKGROUND_WIDTH
        pipe_1_x = obs[3] * BACKGROUND_WIDTH
        pipe_2_x = obs[6] * BACKGROUND_WIDTH
        player_y = obs[9] * BACKGROUND_HEIGHT
        
        if pipe_0_x + PIPE_WIDTH < BIRD_X:
            current_pipe_top, current_pipe_bottom = obs[4] * BACKGROUND_HEIGHT, obs[5] * BACKGROUND_HEIGHT
        elif pipe_1_x + PIPE_WIDTH < BIRD_X:
            current_pipe_top, current_pipe_bottom = obs[7] * BACKGROUND_HEIGHT, obs[8] * BACKGROUND_HEIGHT
        else:
            current_pipe_top, current_pipe_bottom = obs[1] * BACKGROUND_HEIGHT, obs[2] * BACKGROUND_HEIGHT

        desired_y = current_pipe_bottom - 30
        return 1 if player_y > desired_y else 0

    obs, info = env.reset()
    while True:
        # Determine the perfect action
        action = get_action(obs)

        # Processing step
        obs, reward, done, terminated, info = env.step(action)
        states.append(obs.tolist())

        if done or terminated:
            break

    env.close()
    states = pd.DataFrame(states, columns=column_names)
    states.to_csv("perfect_states.csv",index=False)
    assert obs.shape == env.observation_space.shape
    assert info["score"] >= 0


def test_play() -> None:
    """Tests a game of Flappy Bird with a perfect agent."""
    play(audio_on=False, render_mode="human", use_lidar=False)


if __name__ == "__main__":
    test_play()

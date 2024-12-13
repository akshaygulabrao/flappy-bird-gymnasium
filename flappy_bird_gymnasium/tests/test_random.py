# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
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
random agent.
"""
from typing import Union
import gymnasium



def play(
    audio_on: bool = True,
    render_mode: Union[str, None] = "human",
    use_lidar: bool = False,
) -> None:
    """Plays a game of Flappy Bird with a random agent.

    Args:
        audio_on (bool): If `True`, the game will play audio. Defaults to `True`.
        render_mode (Union[str, None]): The render mode. Defaults to `"human"`.
        use_lidar (bool): If `True`, the game will use LIDAR observations. Defaults
            to `False`.
    """
    env = gymnasium.make(
        "FlappyBird-v0", audio_on=audio_on, render_mode=render_mode, use_lidar=use_lidar
    )
    obs = env.reset()
    while True:
        # Getting random action:
        action = env.action_space.sample()

        # Processing:
        obs, _, done, _, info = env.step(action)

        print(f"Obs: {obs}\n" f"Score: {info['score']}\n")

        if done:
            break

    env.close()
    assert obs.shape == env.observation_space.shape
    assert info["score"] == 0


def test_play() -> None:
    """Tests a game of Flappy Bird with a random agent."""
    play(audio_on=False, render_mode=None, use_lidar=False)
    play(audio_on=False, render_mode=None, use_lidar=True)


if __name__ == "__main__":
    play()

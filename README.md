# 2048 Video Game

## Examples

To play 2048 Video Game by using a random strategy, run the code shown below:

```python
from environment.Game2048 import GameEnv

env = GameEnv()
observation, action, reward, done = env.reset(), None, 0.0, False
k = 0
print("#move: {}, obs: {}, action: {}, reward: {}, score: {}, done: {}".format(k, observation, action, reward, env.score, done))
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    k += 1
    print("#move: {}, obs: {}, action: {}, reward: {}, score: {}, done: {}".format(k, observation, action, reward, env.score, done))
```

To play 2048 Video Game with graphical interface by using a random strategy, run the code shown below:

```python
from time import sleep
from environment.Game2048 import GameEnv

env = GameEnv()
env.render.start()
observation, action, reward, done = env.reset(), None, 0.0, False
k = 0
print("#move: {}, obs: {}, action: {}, reward: {}, score: {}, done: {}".format(k, observation, action, reward, env.score, done))
while not done:
    action = env.action_space.sample()
    sleep(0.5)
    observation, reward, done, _ = env.step(action)
    k += 1
    print("#move: {}, obs: {}, action: {}, reward: {}, score: {}, done: {}".format(k, observation, action, reward, env.score, done))
    env.render.refresh()
env.render.close()
```

## Requirements

This code was tested on **Python 3.8.5**.

To install the required python pakages, please run:

```python
python3 -m pip install --no-cache-dir pipenv
python3 -m pipenv sync
```

To activate the virtual environment, please run:

```python
python3 -m pipenv shell
```

To remove the virtual environment, please run:

```python
python3 -m pipenv --rm
```

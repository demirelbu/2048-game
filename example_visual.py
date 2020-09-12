from time import sleep
from Game2048 import GameEnv


def main():
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


if __name__ == "__main__":
    main()

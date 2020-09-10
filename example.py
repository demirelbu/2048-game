from Game2048 import GameEnv


def main():
    env = GameEnv()
    observation = env.reset()
    action = None
    reward = 0.0
    done = False
    k = 0
    print("#move: {}, obs: {}, action: {}, reward: {}, score: {}, done: {}".format(k, observation, action, reward, env.score, done))
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        k += 1
        print("#move: {}, obs: {}, action: {}, reward: {}, score: {}, done: {}".format(k, observation, action, reward, env.score, done))


if __name__ == "__main__":
    main()

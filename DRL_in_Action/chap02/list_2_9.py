import numpy as np
import random
import torch
import matplotlib.pyplot as plt

arms = 10
N, D_in, H, D_out = 1, arms, 100, arms


class ContextBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.init_distribution(arms)
        self.update_state()

    def init_distribution(self, arms):
        # Num states = Num Arms to keep things simple
        # 10X10 배열을 만든다 행 : State, 열 : Action
        self.bandit_matrix = np.random.rand(arms, arms)
        # each row represents a state, each column an arm

    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state(self):
        return self.state

    def update_state(self):
        # 현재 상태를 하나 고른다.
        self.state = np.random.randint(0, self.arms)

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])

    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward


model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.ReLU(),
)

loss_fn = torch.nn.MSELoss()


def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec


def softmax(av, tau=1.12):
    softm = (np.exp(av / tau) / np.sum(np.exp(av / tau)))
    return softm


def running_mean(x, N=50):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y


def train(env, epochs=5000, learning_rate=1e-2):
    # one_hot vector에서 현재의 state만 1로 만든다.
    cur_state = torch.Tensor(one_hot(arms, env.get_state()))  # A
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards = []
    for i in range(epochs):
        # model : 10->100->10 , y_pred : shape: (10, )의 vector
        y_pred = model(cur_state)  # B
        # Tensor.data : requires_grad=False=
        av_softmax = softmax(y_pred.data.numpy(), tau=2.0)  # C
        temp = av_softmax.sum()
        av_softmax /= av_softmax.sum()  # D
        # 확률 p에 따라 arm 중 하나를 고른다.
        choice = np.random.choice(arms, p=av_softmax)  # E
        # 현재 상태에서 arm을 했을 때의 reward를 리턴하고 state를 바꾼다.
        cur_reward = env.choose_arm(choice)  # F
        # y_pred의 choice부분만 cur_reward로 바꾼다.
        one_hot_reward = y_pred.data.numpy().copy()  # G
        one_hot_reward[choice] = cur_reward  # H
        reward = torch.Tensor(one_hot_reward)
        rewards.append(cur_reward)
        loss = loss_fn(y_pred, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 현재 상태를 갱신한다. arms=10
        cur_state = torch.Tensor(one_hot(arms, env.get_state()))  # I
    return np.array(rewards)


if __name__ == "__main__":
    env = ContextBandit(arms)
    rewards = train(env)
    plt.plot(running_mean(rewards, N=500))
    plt.show()

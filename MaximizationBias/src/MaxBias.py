""" This class describes Example 6.7 `Maximization Bias Example` in Sutton's
book `Reinforcement Learning: An introduction`.

In this example, there are 3 states: A, B and TRM, where A is the initial
state and TRM is the the terminal state. The corresponding actions of each
non-terminal state are as follows: In A, the agent needs to choose an action
between left and right. The right action transitions immediately to the state
TRM with a reward 0. When left is chosen, the agent arrives at B with 0 reward
and then he must play a multi-arm bandit. The reward of each arm follows a
normal distribution N(mu, sigma^2) (mu = -0.1 and sigma = 1 by default). After
that, he comes to the terminate state.

Jingyu Liu, May 22, 2023.

"""

import random
from enum import Enum
import numpy as np


class State(Enum):
    A = 0
    B = 1
    TRM = 2


class Direction(Enum):
    left = 0
    right = 1


class MaxBias:
    def __init__(self, arm_number=10, mu=-0.1, sigma=1.0):
        # Constructor.

        # Agent setting.
        self.state_ = State.A.value  # Current state.

        # Environment setting.
        self.arm_number_ = arm_number  # Number of arms.
        self.mu_ = mu  # Expectation of each arm.
        self.sigma_ = sigma  # Standard deviation of each arm.

        # Value function:
        # [V(A), V(B), V(TRM)].
        self.Vfun_ = [0.0, 0.0, 0.0]
        # Q function:
        # q(A, a) where a = left or right,
        # q(B, a) where a = 0, 1, ..., (arm_number - 1),
        # q(TRM, a) = 0.
        self.q_ = []
        self.q1_ = []
        self.q2_ = []

    def QLearning(self, epsilon=0.1, step_size=0.1, discount=1.0,
                  episode_number=300, eta=1.0):
        # Q-Learning according to the epsilon-greedy policy.

        left_action_array = np.zeros(episode_number)
        self.InitializeQ()
        for i in range(episode_number):
            self.InitializeState()
            while self.state_ != State.TRM.value:
                a = self.GenerateQAction(epsilon)
                if self.state_ == State.A.value and a == Direction.left.value:
                    left_action_array[i] = 1
                s, r = self.TakeAction(a)
                self.UpdateQ(a, s, r, step_size, discount)
                self.UpdateState(s)
            if (i + 1) % 100 == 0:
                epsilon *= eta
        return left_action_array

    def InitializeQ(self):
        # Initialize Q list.

        self.q_ = [np.zeros(2), np.zeros(self.arm_number_), np.zeros(1)]

    def InitializeState(self):
        # Initialize state.

        self.state_ = State.A.value

    def GenerateQAction(self, epsilon):
        # Generate action in Q-learning. The return is a number. For state A, it
        # represents left (0) or right (1). For state B, it is which arm the
        # agent chooses.

        if self.state_ == State.A.value:
            # Epsilon-greedy policy.
            p = random.random()
            if p <= epsilon:
                # Random action.
                return random.choice(
                    [Direction.left.value, Direction.right.value])
            else:
                # Choose max reward action.
                if (self.q_[State.A.value][Direction.left.value]
                        > self.q_[State.A.value][Direction.right.value]):
                    return Direction.left.value
                elif (self.q_[State.A.value][Direction.left.value]
                      < self.q_[State.A.value][Direction.right.value]):
                    return Direction.right.value
                else:
                    return random.choice(
                        [Direction.left.value, Direction.right.value])
        elif self.state_ == State.B.value:
            p = random.random()
            if p <= epsilon:
                # Random action.
                return random.randint(0, self.arm_number_ - 1)
            else:
                arm_candidate = np.where(
                    self.q_[State.B.value] == np.max(self.q_[State.B.value]))
                arm_candidate = arm_candidate[0]
                return random.choice(arm_candidate)
        else:
            raise Exception("State error when generating an action!")

    def TakeAction(self, a):
        # Take action. The returns are the next state and the reward after
        # taking action a.

        if self.state_ == State.A.value:
            if a == Direction.left.value:
                return State.B.value, 0.0
            elif a == Direction.right.value:
                return State.TRM.value, 0.0
            else:
                raise Exception(
                    "Action error at state A when taking an action!")
        elif self.state_ == State.B.value:
            if 0 <= a < self.arm_number_:
                reward = random.gauss(self.mu_, self.sigma_)
                return State.TRM.value, reward
            else:
                raise Exception(
                    "Action error at state B when taking an action!")
        else:
            raise Exception("State error when taking an action!")

    def UpdateQ(self, a, s, r, step_size, discount):
        # Update Q list.

        # self.q_[self.state_][a] +=
        # step_size * (r + discount * max(self.q_[s]) - self.q_[self.state_][a])
        self.q_[self.state_][a] += (
                step_size *
                (r + discount * np.max(self.q_[s]) - self.q_[self.state_][a]))

    def UpdateState(self, s):
        # Update state.

        self.state_ = s

    def DoubleQLearning(self, epsilon=0.1, step_size=0.1, discount=1.0,
                        episode_number=300, eta=1.0):
        # Double Q-Learning according to the epsilon-greedy policy.

        left_action_array = np.zeros(episode_number)
        self.InitializeDoubleQ()
        for i in range(episode_number):
            self.InitializeState()
            while self.state_ != State.TRM.value:
                a = self.GenerateDoubleQAction(epsilon)
                if self.state_ == State.A.value and a == Direction.left.value:
                    left_action_array[i] = 1
                s, r = self.TakeAction(a)
                self.UpdateDoubleQ(a, s, r, step_size, discount)
                self.UpdateState(s)
            if (i + 1) % 100 == 0:
                epsilon *= eta
        return left_action_array

    def InitializeDoubleQ(self):
        # Initialize double Q list.

        self.q2_ = [np.zeros(2), np.zeros(self.arm_number_), np.zeros(1)]
        self.q1_ = [np.zeros(2), np.zeros(self.arm_number_), np.zeros(1)]

    def GenerateDoubleQAction(self, epsilon):
        # Generate action in double Q-learning. The return is a number. For
        # state A, it represents left (0) or right (1). For state B, it is which
        # arm the agent chooses.

        if self.state_ == State.A.value:
            # Epsilon-greedy policy.
            p = random.random()
            if p <= epsilon:
                # Random action.
                return random.choice(
                    [Direction.left.value, Direction.right.value])
            else:
                # Choose max reward action.
                if (self.q1_[State.A.value][Direction.left.value] +
                        self.q2_[State.A.value][Direction.left.value] >
                        self.q1_[State.A.value][Direction.right.value] +
                        self.q2_[State.A.value][Direction.right.value]):
                    return Direction.left.value
                elif (self.q1_[State.A.value][Direction.left.value] +
                      self.q2_[State.A.value][Direction.left.value] <
                      self.q1_[State.A.value][Direction.right.value] +
                      self.q2_[State.A.value][Direction.right.value]):
                    return Direction.right.value
                else:
                    return random.choice(
                        [Direction.left.value, Direction.right.value])
        elif self.state_ == State.B.value:
            p = random.random()
            if p <= epsilon:
                # Random action.
                return random.randint(0, self.arm_number_ - 1)
            else:
                total_q_B = np.add(
                    self.q1_[State.B.value], self.q2_[State.B.value])
                arm_candidate = np.where(total_q_B == np.max(total_q_B))
                arm_candidate = arm_candidate[0]
                return random.choice(arm_candidate)
        else:
            raise Exception("State error when generating an action!")

    def UpdateDoubleQ(self, a, s, r, step_size, discount):
        # Update double Q list.

        p = random.random()
        if p <= 0.5:
            action_candidate = (
                [i for i, v in enumerate(self.q1_[s]) if v == max(self.q1_[s])])
            self.q1_[self.state_][a] += (
                    step_size *
                    (r + discount * self.q2_[s][random.choice(action_candidate)]
                     - self.q1_[self.state_][a]))
        else:
            action_candidate = (
                [i for i, v in enumerate(self.q2_[s]) if v == max(self.q2_[s])])
            self.q2_[self.state_][a] += (
                    step_size *
                    (r + discount * self.q1_[s][random.choice(action_candidate)]
                     - self.q2_[self.state_][a]))

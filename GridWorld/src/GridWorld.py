""" This class describes Example 3.5 and Example 3.8 `GridWorld` in Sutton's
book `Reinforcement Learning: An introduction`. We test Q-learning and E-SARSA
on it.

Jingyu Liu, June 21, 2023.

"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
import copy

#   direction    order    vector
#   east           0      [1, 0]
#   west           1      [-1, 0]
#   north          2      [0, 1]
#   south          3      [0, -1]
DIRECTION_VEC = np.array([[1, 0],
                          [-1, 0],
                          [0, 1],
                          [0, -1]])

# Transition state:
# A --> A', B --> B'.
A = np.array([1, 4])
A_PRIME = np.array([1, 0])
B = np.array([3, 4])
B_PRIME = np.array([3, 2])

# Rewards.
REWARD_A_TO_A_PRIME = 10
REWARD_B_TO_B_PRIME = 5
REWARD_OUT_OF_BOUND = -1
REWARD_MOVE = 0


class GridWorld:
    def __init__(self):
        # Constructor.

        # Agent setting.
        self.state_ = np.empty(2)

        # Value function and Q function.
        self.v_fun_ = np.empty((5, 5))
        self.q_fun_ = np.empty((5, 5, 4))
        self.state_update_number_ = np.empty((5, 5))

        # Policy.
        self.policy_ = [[None for j in range(5)] for i in range(5)]

    def OptimalV(self, discount=0.9, tol=1e-4):
        # Compute optimal value function.

        self.InitializeV()
        sweep_iter_number = 0
        while True:
            v_fun_old = copy.deepcopy(self.v_fun_)
            for i in range(5):
                for j in range(5):
                    self.state_ = np.array([i, j])
                    v_update = np.zeros(4)
                    for a in range(4):
                        s, r = self.TakeAction(a)
                        v_update[a] = r + discount * self.v_fun_[s[0]][s[1]]
                    self.v_fun_[i][j] = np.max(v_update)
            sweep_iter_number += 1
            if (np.max(np.abs(self.v_fun_ - v_fun_old))
                    / max(1, np.max(np.abs(v_fun_old))) < tol):
                break
        return sweep_iter_number

    def InitializeV(self):
        # Initialize value function.

        self.v_fun_ = np.zeros((5, 5))

    def OptimalQ(self, discount=0.9, tol=1e-4):
        # Compute optimal Q function.

        self.InitializeQ()
        sweep_iter_number = 0
        while True:
            q_fun_old = copy.deepcopy(self.q_fun_)
            for i in range(5):
                for j in range(5):
                    self.state_ = np.array([i, j])
                    for a in range(4):
                        s, r = self.TakeAction(a)
                        self.q_fun_[i][j][a] = (
                                r + discount * np.max(self.q_fun_[s[0]][s[1]]))
            sweep_iter_number += 1
            if (np.max(np.abs(self.q_fun_ - q_fun_old))
                    / max(1, np.max(np.abs(q_fun_old))) < tol):
                break

        self.FindPolicy()

        return sweep_iter_number

    def InitializeQ(self):
        # Initialize Q function.

        self.q_fun_ = np.zeros((5, 5, 4))

    def FindPolicy(self):
        # Give best action in each state.

        self.InitializePolicy()
        for i in range(5):
            for j in range(5):
                policy_ij = (np.where(
                    self.q_fun_[i][j] == np.max(self.q_fun_[i][j])))
                policy_ij = policy_ij[0]
                self.policy_[i][j] = policy_ij

    def InitializePolicy(self):
        # Initialize best action list.

        self.policy_ = [[None for j in range(5)] for i in range(5)]

    def ShowCurrentV(self, save_file=None):
        # Plot the value in grid world.

        square_length = 50
        total_length = 5 * square_length
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim([0, total_length])
        ax.set_ylim([0, total_length])
        ax.set_xticks([])
        ax.set_yticks([])

        for i in range(5):
            for j in range(5):
                # Bottom left corner of the square.
                bl_corner_x = i * square_length
                bl_corner_y = j * square_length

                # Plot square.
                rect = plt.Rectangle((bl_corner_x, bl_corner_y),
                                     square_length, square_length,
                                     edgecolor='black',
                                     facecolor='none')
                ax.add_patch(rect)

                # Plot arrow
                center_x = bl_corner_x + square_length / 2
                center_y = bl_corner_y + square_length / 2
                plt.text(center_x - square_length / 5,
                         center_y - square_length / 10,
                         round(self.v_fun_[i][j], 1),
                         color="black")

        plt.show()
        if save_file is not None:
            fig.savefig(save_file)

    def ShowCurrentPolicy(self, save_file=None):
        # Plot the policy in grid world.

        square_length = 50
        total_length = 5 * square_length
        arrow_length = square_length / 3
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim([0, total_length])
        ax.set_ylim([0, total_length])
        ax.set_xticks([])
        ax.set_yticks([])

        for i in range(5):
            for j in range(5):
                # Bottom left corner of the square.
                bl_corner_x = i * square_length
                bl_corner_y = j * square_length

                # Plot square.
                rect = plt.Rectangle((bl_corner_x, bl_corner_y),
                                     square_length, square_length,
                                     edgecolor='black',
                                     facecolor='none')
                ax.add_patch(rect)

                # Plot arrow
                center_x = bl_corner_x + square_length / 2
                center_y = bl_corner_y + square_length / 2
                for direction in self.policy_[i][j]:
                    arrow = Arrow(center_x, center_y,
                                  DIRECTION_VEC[direction][0] * arrow_length,
                                  DIRECTION_VEC[direction][1] * arrow_length,
                                  width=(square_length / 5),
                                  color='black')
                    ax.add_patch(arrow)

        plt.show()
        if save_file is not None:
            fig.savefig(save_file)

    def QLearning(self, epsilon=0.1, step_size=0.1, discount=0.9,
                  moving_step_number=10000,
                  target_policy=None,
                  start_state=None):
        # Q-Learning according to the epsilon-greedy policy.

        if target_policy is not None:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            if start_state is None:
                self.InitializeState()
            else:
                self.UpdateState(start_state)
            print(f"Initial state: {self.state_}")
            update_iter_number = 0
            while not self.PolicySubset(target_policy):
                a = self.GenerateQAction(epsilon)
                s_prime, r = self.TakeAction(a)
                self.UpdateQ(a, s_prime, r, step_size, discount)
                self.UpdateV()
                self.UpdatePolicy()
                self.state_update_number_[self.state_[0]][self.state_[1]] += 1
                update_iter_number += 1
                self.UpdateState(s_prime)
        else:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            if start_state is None:
                self.InitializeState()
            else:
                self.UpdateState(start_state)
            print(f"Initial state: {self.state_}")
            update_iter_number = 0
            for it in range(moving_step_number):
                a = self.GenerateQAction(epsilon)
                s_prime, r = self.TakeAction(a)
                self.UpdateQ(a, s_prime, r, step_size, discount)
                self.UpdateV()
                self.UpdatePolicy()
                self.state_update_number_[self.state_[0]][self.state_[1]] += 1
                update_iter_number += 1
                self.UpdateState(s_prime)

        return update_iter_number

    def InitializeStateUpdateNumber(self):
        # Initialize state update number.

        self.state_update_number_ = np.zeros((5, 5))

    def InitializeState(self):
        # Initialize state.

        self.state_ = np.random.randint(0, 5, 2)

    def GenerateQAction(self, epsilon, s=None):
        # Generate action in Q-learning according to epsilon-greedy. The return
        # is an integer a representing the direction the agent will move to.

        if s is None:
            p = random.random()
            if p <= epsilon:
                return random.choice([0, 1, 2, 3])
            else:
                direction_candidate = (np.where(
                    self.q_fun_[self.state_[0]][self.state_[1]]
                    == np.max(self.q_fun_[self.state_[0]][self.state_[1]])))
                direction_candidate = direction_candidate[0]
                return random.choice(direction_candidate)
        else:
            p = random.random()
            if p <= epsilon:
                return random.choice([0, 1, 2, 3])
            else:
                direction_candidate = (np.where(
                    self.q_fun_[s[0]][s[1]] == np.max(self.q_fun_[s[0]][s[1]])))
                direction_candidate = direction_candidate[0]
                return random.choice(direction_candidate)

    def TakeAction(self, a):
        # Take action a. The returns are the next state s and the reward r after
        # taking action a.

        if (self.state_ == A).all():
            return A_PRIME, REWARD_A_TO_A_PRIME
        elif (self.state_ == B).all():
            return B_PRIME, REWARD_B_TO_B_PRIME

        new_state = np.add(self.state_, DIRECTION_VEC[a])
        if (new_state[0] == -1 or new_state[0] == 5
                or new_state[1] == -1 or new_state[1] == 5):
            return self.state_, REWARD_OUT_OF_BOUND
        else:
            return new_state, REWARD_MOVE

    def UpdateQ(self, a, s_prime, r, step_size, discount):
        # Update Q function in Q-learning.

        incr = (r + discount * np.max(self.q_fun_[s_prime[0]][s_prime[1]])
                - self.q_fun_[self.state_[0]][self.state_[1]][a])
        self.q_fun_[self.state_[0]][self.state_[1]][a] += (step_size * incr)

    def UpdateV(self):
        # Update value function.

        self.v_fun_[self.state_[0]][self.state_[1]] = (
            max(self.q_fun_[self.state_[0]][self.state_[1]]))

    def UpdatePolicy(self):
        # Update policy.

        policy_s = (np.where(
            self.q_fun_[self.state_[0]][self.state_[1]]
            == np.max(self.q_fun_[self.state_[0]][self.state_[1]])))
        policy_s = policy_s[0]
        self.policy_[self.state_[0]][self.state_[1]] = policy_s

    def UpdateState(self, s):
        # Update state.

        self.state_ = s

    def PolicySubset(self, given_policy):
        # If policy belongs to the given policy.

        for i in range(5):
            for j in range(5):
                if not (set(self.policy_[i][j]) <= set(given_policy[i][j])):
                    return False
        return True

    def ESARSA(self, epsilon=0.1, step_size=0.1, discount=0.9,
               moving_step_number=10000,
               target_policy=None,
               start_state=None):
        # E-SARSA according to the epsilon-greedy policy.

        if target_policy is not None:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            if start_state is None:
                self.InitializeState()
            else:
                self.UpdateState(start_state)
            print(f"Initial state: {self.state_}")
            update_iter_number = 0
            while not self.PolicySubset(target_policy):
                a = self.GenerateQAction(epsilon)
                s_prime, r = self.TakeAction(a)
                self.UpdateESARSA(a, s_prime, r, step_size, discount, epsilon)
                self.UpdateV()
                self.UpdatePolicy()
                self.state_update_number_[self.state_[0]][self.state_[1]] += 1
                update_iter_number += 1
                self.UpdateState(s_prime)
        else:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            if start_state is None:
                self.InitializeState()
            else:
                self.UpdateState(start_state)
            print(f"Initial state: {self.state_}")
            update_iter_number = 0
            for it in range(moving_step_number):
                a = self.GenerateQAction(epsilon)
                s_prime, r = self.TakeAction(a)
                self.UpdateESARSA(a, s_prime, r, step_size, discount, epsilon)
                self.UpdateV()
                self.UpdatePolicy()
                self.state_update_number_[self.state_[0]][self.state_[1]] += 1
                update_iter_number += 1
                self.UpdateState(s_prime)

        return update_iter_number

    def UpdateESARSA(self, a, s_prime, r, step_size, discount, epsilon):
        # Update Q function in E-SARSA.

        expectation_s = 0
        direction_candidate = (np.where(
            self.q_fun_[s_prime[0]][s_prime[1]]
            == np.max(self.q_fun_[s_prime[0]][s_prime[1]])))
        direction_candidate = direction_candidate[0]
        for i in range(4):
            expectation_s += (epsilon / 4
                              * self.q_fun_[s_prime[0]][s_prime[1]][i])
        for i in direction_candidate:
            expectation_s += ((1 - epsilon) / len(direction_candidate)
                              * self.q_fun_[s_prime[0]][s_prime[1]][i])
        incr = (r + discount * expectation_s
                - self.q_fun_[self.state_[0]][self.state_[1]][a])
        self.q_fun_[self.state_[0]][self.state_[1]][a] += (step_size * incr)

    def SARSA(self, epsilon=0.1, step_size=0.1, discount=0.9,
              moving_step_number=10000,
              target_policy=None,
              start_state=None):
        # SARSA according to the epsilon-greedy policy.

        if target_policy is not None:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            if start_state is None:
                self.InitializeState()
            else:
                self.UpdateState(start_state)
            print(f"Initial state: {self.state_}")
            a = self.GenerateQAction(epsilon)
            update_iter_number = 0
            while not self.PolicySubset(target_policy):
                s_prime, r = self.TakeAction(a)
                a_prime = self.GenerateQAction(epsilon, s_prime)
                self.UpdateSARSA(a, s_prime, r, a_prime, step_size, discount)
                self.UpdateV()
                self.UpdatePolicy()
                self.state_update_number_[self.state_[0]][self.state_[1]] += 1
                update_iter_number += 1
                self.UpdateState(s_prime)
                a = a_prime
        else:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            if start_state is None:
                self.InitializeState()
            else:
                self.UpdateState(start_state)
            print(f"Initial state: {self.state_}")
            a = self.GenerateQAction(epsilon)
            update_iter_number = 0
            for it in range(moving_step_number):
                s_prime, r = self.TakeAction(a)
                a_prime = self.GenerateQAction(epsilon, s_prime)
                self.UpdateSARSA(a, s_prime, r, a_prime, step_size, discount)
                self.UpdateV()
                self.UpdatePolicy()
                self.state_update_number_[self.state_[0]][self.state_[1]] += 1
                update_iter_number += 1
                self.UpdateState(s_prime)
                a = a_prime

        return update_iter_number

    def UpdateSARSA(self, a, s_prime, r, a_prime, step_size, discount):
        # Update Q function in SARSA.

        incr = (r + discount * self.q_fun_[s_prime[0]][s_prime[1]][a_prime]
                - self.q_fun_[self.state_[0]][self.state_[1]][a])
        self.q_fun_[self.state_[0]][self.state_[1]][a] += (step_size * incr)

    def ShowCurrentStateUpdateNumber(self, save_file=None):
        # Plot the state update number in grid world.

        square_length = 50
        total_length = 5 * square_length
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim([0, total_length])
        ax.set_ylim([0, total_length])
        ax.set_xticks([])
        ax.set_yticks([])

        for i in range(5):
            for j in range(5):
                # Bottom left corner of the square.
                bl_corner_x = i * square_length
                bl_corner_y = j * square_length

                # Plot square.
                rect = plt.Rectangle((bl_corner_x, bl_corner_y),
                                     square_length, square_length,
                                     edgecolor='black',
                                     facecolor='none')
                ax.add_patch(rect)

                # Plot arrow
                center_x = bl_corner_x + square_length / 2
                center_y = bl_corner_y + square_length / 2
                update_number_ij = (
                    str(int(self.state_update_number_[i][j])))
                plt.text(center_x - square_length / 15 * len(update_number_ij),
                         center_y - square_length / 10,
                         update_number_ij,
                         color="black")

        plt.show()
        if save_file is not None:
            fig.savefig(save_file)

    def QLearningMaxWalk(self, epsilon=0.1, step_size=0.1, discount=0.9,
                         episode_number=1000,
                         maxwalk=10,
                         target_policy=None):
        # Maximum walk Q-Learning according to the epsilon-greedy policy.

        if target_policy is not None:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            update_iter_number = 0
            episode_iter_number = 0
            while not self.PolicySubset(target_policy):
                self.InitializeState()
                episode_iter_number += 1

                for walk in range(maxwalk):
                    a = self.GenerateQAction(epsilon)
                    s_prime, r = self.TakeAction(a)
                    self.UpdateQ(a, s_prime, r, step_size, discount)
                    self.UpdateV()
                    self.UpdatePolicy()
                    self.state_update_number_[self.state_[0]][
                        self.state_[1]] += 1
                    update_iter_number += 1
                    self.UpdateState(s_prime)
        else:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            update_iter_number = 0
            episode_iter_number = 0
            for it in range(episode_number):
                self.InitializeState()
                episode_iter_number += 1

                for walk in range(maxwalk):
                    a = self.GenerateQAction(epsilon)
                    s_prime, r = self.TakeAction(a)
                    self.UpdateQ(a, s_prime, r, step_size, discount)
                    self.UpdateV()
                    self.UpdatePolicy()
                    self.state_update_number_[self.state_[0]][
                        self.state_[1]] += 1
                    update_iter_number += 1
                    self.UpdateState(s_prime)

        return update_iter_number, episode_iter_number

    def ESARSAMaxWalk(self, epsilon=0.1, step_size=0.1, discount=0.9,
                      episode_number=1000,
                      maxwalk=10,
                      target_policy=None):
        # Maximum walk E-SARSA according to the epsilon-greedy policy.

        if target_policy is not None:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            update_iter_number = 0
            episode_iter_number = 0
            while not self.PolicySubset(target_policy):
                self.InitializeState()
                episode_iter_number += 1

                for walk in range(maxwalk):
                    a = self.GenerateQAction(epsilon)
                    s_prime, r = self.TakeAction(a)
                    self.UpdateESARSA(a, s_prime, r,
                                      step_size, discount, epsilon)
                    self.UpdateV()
                    self.UpdatePolicy()
                    self.state_update_number_[self.state_[0]][
                        self.state_[1]] += 1
                    update_iter_number += 1
                    self.UpdateState(s_prime)
        else:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            update_iter_number = 0
            episode_iter_number = 0
            for it in range(episode_number):
                self.InitializeState()
                episode_iter_number += 1

                for walk in range(maxwalk):
                    a = self.GenerateQAction(epsilon)
                    s_prime, r = self.TakeAction(a)
                    self.UpdateESARSA(a, s_prime, r,
                                      step_size, discount, epsilon)
                    self.UpdateV()
                    self.UpdatePolicy()
                    self.state_update_number_[self.state_[0]][
                        self.state_[1]] += 1
                    update_iter_number += 1
                    self.UpdateState(s_prime)

        return update_iter_number, episode_iter_number

    def SARSAMaxWalk(self, epsilon=0.1, step_size=0.1, discount=0.9,
                     episode_number=1000,
                     maxwalk=10,
                     target_policy=None):
        # Maximum walk SARSA according to the epsilon-greedy policy.

        if target_policy is not None:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            update_iter_number = 0
            episode_iter_number = 0
            while not self.PolicySubset(target_policy):
                self.InitializeState()
                a = self.GenerateQAction(epsilon)
                episode_iter_number += 1

                for walk in range(maxwalk):
                    s_prime, r = self.TakeAction(a)
                    a_prime = self.GenerateQAction(epsilon, s_prime)
                    self.UpdateSARSA(a, s_prime, r, a_prime,
                                     step_size, discount)
                    self.UpdateV()
                    self.UpdatePolicy()
                    self.state_update_number_[self.state_[0]][
                        self.state_[1]] += 1
                    update_iter_number += 1
                    self.UpdateState(s_prime)
                    a = a_prime
        else:
            self.InitializeV()
            self.InitializeQ()
            self.InitializeStateUpdateNumber()

            self.FindPolicy()

            update_iter_number = 0
            episode_iter_number = 0
            for it in range(episode_number):
                self.InitializeState()
                a = self.GenerateQAction(epsilon)
                episode_iter_number += 1

                for walk in range(maxwalk):
                    s_prime, r = self.TakeAction(a)
                    a_prime = self.GenerateQAction(epsilon, s_prime)
                    self.UpdateSARSA(a, s_prime, r, a_prime,
                                     step_size, discount)
                    self.UpdateV()
                    self.UpdatePolicy()
                    self.state_update_number_[self.state_[0]][
                        self.state_[1]] += 1
                    update_iter_number += 1
                    self.UpdateState(s_prime)
                    a = a_prime

        return update_iter_number, episode_iter_number

    def TargetPolicyProportion(self, target_policy):
        # Proportion in target policy.

        self.FindPolicy()
        number_count = 0
        for i in range(5):
            for j in range(5):
                if set(self.policy_[i][j]) <= set(target_policy[i][j]):
                    number_count += 1
        return number_count / 25.0

    def QLearningObserve(self, target_policy,
                         epsilon=0.1, step_size=0.1, discount=0.9,
                         moving_step_number=10000,
                         observe_interval=1000):
        # Observe Q-Learning according to the epsilon-greedy policy.

        optimal_action_proportion_array = np.zeros(
            int(moving_step_number / observe_interval) + 1)

        self.InitializeV()
        self.InitializeQ()
        self.InitializeStateUpdateNumber()

        self.FindPolicy()

        self.InitializeState()
        update_iter_number = 0
        cnt = 0
        for it in range(moving_step_number + 1):
            if update_iter_number % observe_interval == 0:
                optimal_action_proportion = self.TargetPolicyProportion(
                    target_policy=target_policy)
                optimal_action_proportion_array[cnt] = (
                    optimal_action_proportion)
                cnt += 1
            a = self.GenerateQAction(epsilon)
            s_prime, r = self.TakeAction(a)
            self.UpdateQ(a, s_prime, r, step_size, discount)
            self.UpdateV()
            self.UpdatePolicy()
            self.state_update_number_[self.state_[0]][self.state_[1]] += 1
            update_iter_number += 1
            self.UpdateState(s_prime)

        return optimal_action_proportion_array

    def ESARSAObserve(self, target_policy,
                      epsilon=0.1, step_size=0.1, discount=0.9,
                      moving_step_number=10000,
                      observe_interval=1000):
        # Observe E-SARSA according to the epsilon-greedy policy.

        optimal_action_proportion_array = np.zeros(
            int(moving_step_number / observe_interval) + 1)

        self.InitializeV()
        self.InitializeQ()
        self.InitializeStateUpdateNumber()

        self.FindPolicy()

        self.InitializeState()
        update_iter_number = 0
        cnt = 0
        for it in range(moving_step_number + 1):
            if update_iter_number % observe_interval == 0:
                optimal_action_proportion = self.TargetPolicyProportion(
                    target_policy=target_policy)
                optimal_action_proportion_array[cnt] = (
                    optimal_action_proportion)
                cnt += 1
            a = self.GenerateQAction(epsilon)
            s_prime, r = self.TakeAction(a)
            self.UpdateESARSA(a, s_prime, r, step_size, discount, epsilon)
            self.UpdateV()
            self.UpdatePolicy()
            self.state_update_number_[self.state_[0]][self.state_[1]] += 1
            update_iter_number += 1
            self.UpdateState(s_prime)

        return optimal_action_proportion_array

    def SARSAObserve(self, target_policy,
                     epsilon=0.1, step_size=0.1, discount=0.9,
                     moving_step_number=10000,
                     observe_interval=1000):
        # Observe SARSA according to the epsilon-greedy policy.

        optimal_action_proportion_array = np.zeros(
            int(moving_step_number / observe_interval) + 1)

        self.InitializeV()
        self.InitializeQ()
        self.InitializeStateUpdateNumber()

        self.FindPolicy()

        self.InitializeState()
        a = self.GenerateQAction(epsilon)
        update_iter_number = 0
        cnt = 0
        for it in range(moving_step_number + 1):
            if update_iter_number % observe_interval == 0:
                optimal_action_proportion = self.TargetPolicyProportion(
                    target_policy=target_policy)
                optimal_action_proportion_array[cnt] = (
                    optimal_action_proportion)
                cnt += 1
            s_prime, r = self.TakeAction(a)
            a_prime = self.GenerateQAction(epsilon, s_prime)
            self.UpdateSARSA(a, s_prime, r, a_prime, step_size, discount)
            self.UpdateV()
            self.UpdatePolicy()
            self.state_update_number_[self.state_[0]][self.state_[1]] += 1
            update_iter_number += 1
            self.UpdateState(s_prime)
            a = a_prime

        return optimal_action_proportion_array

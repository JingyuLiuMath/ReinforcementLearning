# GridWorld

这是关于 Sutton 的书 `Reinforcement Learning: An introduction` 中 `Example 3.5` 和 `Example 3.8` `GridWorld` 的一些讨论.

- `./doc` 是相关的文档.
- `./images` 是文档中相关的图片.
- `./src` 目录下是我们的代码, 主要是一个名叫 `GridWorld` 的类.
- `./test` 目录下有 5 个文件夹:
  - `optimal_value_and_policy` 是计算最佳值函数和最佳策略并绘图的代码.
  - `RL_optimal` 是 3 种强化学习方法 (SARSA, Q-learning, E-SARSA) 的表现, 这里我们让其收敛到最优策略, 并给出此时的策略, 值函数和更新次数.
  - `RL_in_100w_updates` 是 3 种强化学习方法更新 100 万次后的结果, 我们同样给出此时的策略和值函数.
  - `RL_maxwalk_optimal` 是 3 种强化学习方法在最大游走次数 \(k = 10\) 时的表现.
  - `RL_maxwalk_k`是 3 种强化学习方法的更新次数与最大游走次数 \(k\) 的关系. 值得注意的是, 这里的代码运行时间很长 (约 1 天).

刘经宇, 2023 年 6 月 25 日.

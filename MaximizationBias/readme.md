# MaximizationBias

这是关于 Sutton 的书 `Reinforcement Learning: An introduction` 中 `Example 6.7` 的结果的复现和一些进一步的讨论.

- `./doc` 是相关的文档.
- `./src` 目录下是我们的代码, 主要是一个名叫 `MaxBias` 的类.
- `./test` 目录下有 4 个文件:
  - `demo.py` 展示了如何使用我们的类.
  - `reproduce.py` 是对于书中结果的复现.
  - `long_episode.py` 展现了这个例子在更长的 episode 上的结果.
  - `refinement.py` 是对算法的改进. 我们主要调整了学习中的 $\varepsilon$-greedy, 这保证了结果能收敛到理论上的最优值.
- `./images` 是文档中相关的图片.

刘经宇, 2023 年 5 月 22 日.

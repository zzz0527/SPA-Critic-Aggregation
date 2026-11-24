# Third-party sources

| Local path | Source basis | Upstream source |
| --- | --- | --- |
| `methods/iql_aggregation/`, `baselines/iql/` | Adapted from the official IQL implementation | [ikostrikov/implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning) |
| `methods/idql_aggregation/`, `baselines/idql/` | Adapted from the official IDQL implementation | [philippe-eecs/IDQL](https://github.com/philippe-eecs/IDQL) |
| `baselines/active/` | Adapted from the code released with ACTIVE | [ICLR 2025 supplemental code](https://proceedings.iclr.cc/paper_files/paper/2025/file/c06f788963f0ce069f5b2dbf83fe7822-Supplemental-Conference.zip) |
| `baselines/proj_iql/` | Reproduction built on the repository's IQL baseline; no official Projective-IQL code was used | [Projective Implicit Q-Learning paper](https://arxiv.org/abs/2501.08907) |
| `baselines/uniq/` | Reproduction built on the repository's IQL baseline; no official UNIQ code was used | [UNIQ paper](https://arxiv.org/abs/2606.07592) |
| `requirements.txt` | D4RL dependency pinned to the source revision used by this package | [rail-berkeley/d4rl at `89141a6`](https://github.com/rail-berkeley/d4rl/tree/89141a689b0353b0dac3da5cba60da4b1b16254d) |

The SPA routing, aggregation, and stagewise training logic in `methods/` are the contributions of this package.

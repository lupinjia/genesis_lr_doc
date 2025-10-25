# genesis_lr

[![GitHub Repo stars](https://img.shields.io/github/stars/lupinjia/genesis_lr)](https://github.com/lupinjia/genesis_lr)

## Introduction

genesis_lr is a training framework for robot control based on RL (Reinforcement Learning). It's based on [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl) from RSL in ETHZ. 

## Features

- 🤖**Integration of IsaacGym and Genesis** in one framework. You only need to create two conda environments to use different simulators.
- 💡**Incoporation of various published papers of RL**. We implemented various methods published in papers and provided material to compare and differentiate them.
    
    | Method | Paper Link | Location | Materials |
    |--------|------------|----------|-----------|
    | Walk These Ways | [Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://arxiv.org/abs/2212.03238) | [go2_wtw](https://github.com/lupinjia/genesis_lr/blob/main/legged_gym/envs/go2/go2_wtw/) | [walk_these_ways](./user_guide/blind_locomotion/walk_these_ways.md) |
    | System Identification | [Learning Agile Bipedal Motions on a Quadrupedal Robot](https://arxiv.org/abs/2311.05818) | [go2_sysid](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_sysid) | |
    | One-Stage Teacher-Student | [Rapid Locomotion via Reinforcement Learning](https://agility.csail.mit.edu/) | [go2_ts](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ts) | [teacher_student](./user_guide/blind_locomotion/teacher_student.md) |
    | EstimatorNet | [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481) | [go2_ee](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ee) | [explicit_estimator](./user_guide/blind_locomotion/explicit_estimator.md) |
    | Constraints as Terminations | [CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning](https://constraints-as-terminations.github.io/) | [go2_cat](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_cat) | |

## Implemented Demos

| Name  | Gensis | IsaacGym | Real Robot | Video |
| ----- | ------ | -------- | ---------  | ----- |
| Walk These Ways (Go2) | ✅ | ✅ | ✅ | [video_link](https://www.bilibili.com/video/BV1FPedzZEdi/) |
| One-Stage Teacher-Student | ✅ | ✅ |
| |

```{toctree}
:maxdepth: 1

user_guide/index

```
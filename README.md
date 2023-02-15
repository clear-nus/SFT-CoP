# Safety-Constrained Policy Transfer with Successor Features
[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2211.05361&color=B31B1B)](https://arxiv.org/abs/2211.05361)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Coming soon! We are currently cleaning up our code for release.

This repository contains the code for our paper [Safety-Constrained Policy Transfer with Successor Features](https://arxiv.org/abs/2211.05361) (ICRA-23).

## Introduction

In this work, we focus on the problem of safe policy transfer in reinforcement learning: we seek to leverage existing policies when learning a new task with specified constraints. This problem is important for safety-critical applications where interactions are costly and unconstrained policies can lead to undesirable or dangerous outcomes, e.g., with physical robots that interact with humans. We propose a Constrained Markov Decision Process (CMDP) formulation that simultaneously enables the transfer of policies and adherence to safety constraints. Our formulation cleanly separates task goals from safety considerations and permits the specification of a wide variety of constraints. Our approach relies on a novel extension of generalized policy improvement to constrained settings via a Lagrangian formulation. We devise a dual optimization algorithm that estimates the optimal dual variable of a target task, thus enabling safe transfer of policies derived from successor features learned on source tasks. Our experiments in simulated domains show that our approach is effective; it visits unsafe states less frequently and outperforms alternative state-of-the-art methods when taking safety constraints into account.



## License

[MIT](LICENSE)


## Acknowledgement
This repo contains code that's based on the following code: [RaSF](https://openreview.net/forum?id=a_f_NR8mMr9).


## BibTeX

If you find this repository or the ideas presented in our paper useful for your research, please consider citing our paper.


## Contact us

Feel free to contact <a href="mailto:zeyu@comp.nus.edu.sg">Zeyu Feng</a>, <a href="mailto:bowenzhang@comp.nus.edu.sg">Bowen Zhang</a>, <a href="mailto:jianxin.bi@comp.nus.edu.sg">Jianxin Bi</a> or <a href="mailto:harold@comp.nus.edu.sg">Harold Soh</a> for any questions regarding the code or the paper. Please visit our website for more information: [CLeAR website](https://clear-nus.github.io/).

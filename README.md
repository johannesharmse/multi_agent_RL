# Multi Agent Reinforecement Learning

**Submission for Move 37 final assignment**

A Deep Deterministic Policy Gradients(DDPG) algorithm implementation for a multi-agent particle environment. 

**Credit:** The code in this repo has been adapted from Rohan Sawhney's Multi-agent RL [repo](https://github.com/rohan-sawhney/multi-agent-rl).

## Overview

The multi-agent environment has two agents:

* One good agent: **Pacman**

* One adversary: **Blue Ghost**

***Note:** In the code, Pacman is actually viewed as the adversary, but we all know the ghost is the real enemy*

Each of the two agents has their own objectives/ Pacman tries to minimise the distance between itself and the ghost, with the ultimate goal being to collide with the ghost. The blue ghost tries to maximise the distance between itself and Pacman, and tries to escape collisions at all costs.

The game only ends once any of the two players exit the boundaries of frame.

As learning progress, both agents get stronger. The ghost gets better at escaping Pacman, and Pacman gets better at catching the blue ghost.

***Note:** Pacman is slightly slower than the ghost. Life is more fair this way.*

DDPG is an extension of actor-critic reinforcement learning. The actor/agent wants to learn the best policy (how to move give a specific state). The critic helps the actor reach a more stable policy by predicting the value of a state and critiqueing the actor's actions. This prevents the actor from following a policy based on a stroke of luck.

## Additional Resources

[School of AI - Move 37 Course](https://www.theschool.ai/courses/move-37-course/)

[OpenAI - MADDPG](https://blog.openai.com/learning-to-cooperate-compete-and-communicate/)

[Arxiv - Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)


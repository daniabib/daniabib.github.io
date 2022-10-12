---
title: PAC Learning
date: 2022-09-28 12:00:00 -300
categories: [deep-learning, computer-science]
tags: [ai, ml]
---


### FROM: FOUNDATIONS OF ML Ch. 2

"Machine learning is fundamentally about generalization", p. 7.

"The problem is typically formulated as that of selecting a function out of a hypothesis set, that is a subset of the family of all functions."

How should we define the complexity of a hypothesis set?

"Several fundamental questions arise when designing and analyzing algorithms that learn from examples: What can be learned efficiently? What is inherently hard to learn? How many examples are needed to learn successfully? Is there a general model of learning?", p. 9.

## Probably Approximately Correct (PAC)
"The PAC framework helps define the class of learnable concepts in terms of the number of sample points needed to achieve an approximate solution, sample complexity, and the time and space complexity of the learning algorithm, which depends on the cost of the computational representation of the concepts."

The concept $c(x)$ is the "true" concept we want to learn.

**DEFINITION PAC-Learning:** A concept class $\mathcal{C}$ is said to be PAC-learnable if there exists an algorithm $\mathcal{A}$ and a polynomial function $poly(·, ·, ·, ·)$ such that for any $\epsilon > 0$ and $\delta > 0$, for all distributions $\mathcal{D}$ on $\mathcal{X}$ and for any target concept $c \in \mathcal{C}$, the following holds for any sample size $m \geq poly(1/\epsilon, 1/\delta, n, size(c))$:






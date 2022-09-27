---
title: First Test
date: 2022-07-25 12:00:00 -300
categories: [philosophy, computer-science]
tags: [ai, ml]
---

# Heading 1

Today, the most powerful artificial intelligence systems employ a type of machine learning called deep learning. Their algorithms learn by processing massive amounts of data through hidden layers of interconnected nodes, referred to as deep neural networks. As their name suggests, deep neural networks were inspired by the real neural networks in the brain, with the nodes modeled after real neurons — or, at least, after what neuroscientists knew about neurons back in the 1950s, when an influential neuron model called the perceptron was born. Since then, our understanding of the computational complexity of single neurons has dramatically expanded, so biological neurons are known to be more complex than artificial ones. But by how much?

To find out, David Beniaguev, Idan Segev and Michael London, all at the Hebrew University of Jerusalem, trained an artificial deep neural network to mimic the computations of a simulated biological neuron. They showed that a deep neural network requires between five and eight layers of interconnected “neurons” to represent the complexity of one single biological neuron.

But the study’s authors caution that it’s not a straightforward correspondence yet. “The relationship between how many layers you have in a neural network and the complexity of the network is not obvious,” said London. So we can’t really say how much more complexity is gained by moving from, say, four layers to five. Nor can we say that the need for 1,000 artificial neurons means that a biological neuron is exactly 1,000 times as complex. Ultimately, it’s possible that using exponentially more artificial neurons within each layer would eventually lead to a deep neural network with one single layer — but it would likely require much more data and time for the algorithm to learn.

```python
class Test(Enum):
    a: int

    def __init__(self):
        self.a = 1
```

“We tried many, many architectures with many depths and many things, and mostly failed,” said London. The authors have shared their code to encourage other researchers to find a clever solution with fewer layers. But, given how difficult it was to find a deep neural network that could imitate the neuron with 99% accuracy, the authors are confident that their result does provide a meaningful comparison for further research. Lillicrap suggested it might offer a new way to relate image classification networks, which often require upward of 50 layers, to the brain. If each biological neuron is like a five-layer artificial neural network, then perhaps an image classification network with 50 layers is equivalent to 10 real neurons in a biological network.

## Heading 2: Some stuff
The authors also hope that their result will change the present state-of-the-art deep network architecture in AI. “We call for the replacement of the deep network technology to make it closer to how the brain works by replacing each simple unit in the deep network today with a unit that represents a neuron, which is already — on its own — deep,” said Segev. In this replacement scenario, AI researchers and engineers could plug in a five-layer deep network as a “mini network” to replace every artificial neuron.

But some wonder whether this would really benefit AI. “I think that’s an open question, whether there’s an actual computational advantage,” said Anthony Zador, a neuroscientist at Cold Spring Harbor Laboratory. “This [work] lays the foundation for testing that.”
# Yann LeCun EBM
clamping the value of observed variables and finding configuration of remaining variables that minimize the energy


The Nature of Life

In accordance with the second law of thermodynamics, the universe naturally tends towards disorder. However, 
life stands as an exception. Life represents a self-organizing system capable of maintaining order and complexity. 
Its ability to do so hinges on the capacity to extract energy from the environment and utilize it to sustain order.

Thus, the fundamental motivation of life is the maintenance of homeostasis in a universe that inherently gravitates 
towards disorder. Entropy, often defined as the measure of disorder in a system, reflects the extent of free energy 
within that system.

Consider a wood log: it contains significant energy stored in its chemical bonds. When burned, this energy is liberated, 
transforming the log into ash. As per the first law of thermodynamics, energy remains conserved. However, it shifts 
from a bound state within the log to a "free" form of energy that cannot be reclaimed to reconstruct the log. 
Consequently, the energy density within the system decreases.

In order to counteract the universe's inclination towards disorder, life must intake high-density energy and emit 
low-density energy, utilizing these inputs as resources to sustain order.

From this perspective, the intrinsic behaviors of single-celled organisms, such as recognizing advantages and avoiding 
disadvantages, become comprehensible. However, we do not typically ascribe intelligence to single-celled organisms.

# Nature of Life
Based on the second law of thermodynamics, the universe tends to disorder. 
However, life is an exception. Life is a self-organizing system that can maintain order and complexity.
The key to life's ability to maintain order is the ability to extract energy from the environment and use it to maintain order.

So motivation of life is to maintain homeostasis in a world that tends to disorder.

The concept the entropy is described as the measure of disorder in a system. In other words, entropy is a measure  of the extent
of the free energy in a system. 

For example, a wood log has lots of energy that been constrained in the chemical bonds. When the wood log is burned, the energy is released and the wood log is turned into ash.
based on the first law of thermodynamics, energy is conserved. The energy in the wood log converted into a "free" form of energy, which one can't gather back into the wood log.
In this process, the density of energy in the system is increased.

In order to maintain the disorder tendency of the universe, life needs to intake high-density energy and release low-density energy, use those intake as scapegoat to maintain order.

Under this perspective, single cell organism's intrinsic or behavior such as Seeing advantages and avoiding disadvantages  could be easily understood. 
But we won't call a single cell organism as intelligent.

# Nature of Intelligence
Although single cell organism could be understood as a self-organizing system, we won't call it intelligent.
The reason is we believe when an amoebas moving away a high concentration of salt, it is not making a decision, it is just a chemical reaction.
From a high-level perspective, the amoebas doesn't have the "mind" to build model the world, understand what is salt, why is harmful, and what is the consequence of moving toward the salt.

By this example, we could have few hints about the nature of intelligence.
The environment is complex and dynamic, due to the disorder tendency of the universe, the environment is always changing.
In order not only to maintain order, but also win the competition between other self-organizing systems, some elite self-organizing systems are
required to predict the future, given the current state and historical state of the environment.

But where does those states come from? In other words, how those life collect the information about the environment?
The answer is the sensory system. The sensory system is the interface between the environment and the self-organizing system.
Taking human vision as an example, vision system is a system for human to perceive the environment by light's intensity(brightness) and wavelength(color).

Although human vision system is quite advanced, but it could never have told you what is the "real" world looks like.
electromagnetic spectrum is a continuous spectrum, but human vision system only perceive a small portion of it, so called "visible light".
Compared the amount of information in the environment, the information collected by the sensory system is quite limited, which is determined by the physical limitation of the sensory system, 
decide by phenotype of the self-organizing system( it's also dynamic, cause by the evolution nature selection, also serve as prediction of the environment).

Consider all the possible states of real word, the possible states of the sensory system is only a minuscule subset of it.
Given the limited information collected by the sensory system, to perdict the changing environment then make the most optimal decision, 
the self-organizing are required to use the limited information to deduce the most likely states of the environment, then make the decision based on the deduced states.

The mapping from the sensory system to the deduced states is called the model of the world.
The ability to build the model of the world is called intelligence.
This point of view explains the vagueness of the concept of intelligence, such as should we call cat or dog intelligent agent?
Also explain why human tend to find a way explain the world, such as science, religion, philosophy, etc.



# Nature of Intelligence
While single-cell organisms exhibit self-organizing behavior, we refrain from labeling them as intelligent. 
When an amoeba moves away from a high concentration of salt, it's not making a decision but rather reacting chemically. 
From a higher perspective, the amoeba lacks the cognitive capacity to model the world, comprehend the concept of salt, 
understand its harmful effects, or anticipate the consequences of its actions.

This example sheds light on the essence of intelligence. The environment is intricate and ever-changing due to the 
universe's tendency toward disorder. To not only sustain order but also thrive amidst competition from other 
self-organizing systems, some elite systems must forecast the future based on the current and historical states of 
the environment.

But how do organisms gather information about the environment? The answer lies in the sensory system, the interface 
between the environment and the self-organizing system. Take human vision, for instance; it perceives the environment
through light intensity (brightness) and wavelength (color).

Despite the sophistication of human vision, it can't depict the "true" appearance of the world. The electromagnetic 
spectrum is continuous, yet human vision only detects a fraction of it known as "visible light." The information 
collected by the sensory system is limited, dictated by the physical constraints of the system and shaped by the 
self-organizing system's phenotype (which evolves through natural selection and serves as a prediction of the 
environment).

Considering the myriad possible states of the real world, the potential states accessible to the sensory system are 
minuscule. Given this limited information, self-organizing systems must use deduction to predict the most likely states 
of the environment and make decisions accordingly.

The process of mapping from the sensory system to deduced states and mapping from sensory system to prediction of the 
environment states is termed the "model of the world," and the ability to construct this model represents intelligence. 
This perspective elucidates the ambiguity surrounding the concept of intelligence, such as whether we should consider 
cats or dogs as intelligent agents. It also clarifies why humans are driven to understand the world through avenues like science, religion, and philosophy.

# Adaptation and Evolution of Intelligence



# background
On the path of stimulating the intelligence using computer, brain is the perfect place to look for inspiration. 
In the past few decades, computer scientists colaborated with neuroscientists, mathematicians and etc, have exploeded 
mimicking the neural network structure of the brain to solve real-world problems.

In the past few decades, computer scientists have solved tremendous real-world problems in the field of vision, speech, 
and language processing via simulating the neural network structure of the brain.
Although these achievements are remarkable, this oversimplified model of the brain is far from the capabilities of the
human brain. Compared to the human brain, the breakthrough neural network model cosumes a few orders of magnitude more
energy and data to achieve the same level of performance and it requires massive amount of data to train.

On the other hand, even in vitro human brain could achieve the same level of performance with much less energy and data.

A main reason for the difference is the motivation of learning or the framework of learning.

Learning could be viewed as a process of minimizing the difference between two distributions: the real-world distribution
and the model distribution. 
The real-world distribution is the distribution of the environment, the prob of the state of the environment. 
The model distribution is the distribution of the model, the prob of the state predicted by the model.

The most nature way of learning is to minimize the difference between the two distributions is to minimize the energy of the system using 
Gibbs distribution.
but the partition function of the Gibbs distribution is intractable, so the learning process is intractable.

Artificial neural network model simplifies this by surrogating the energy function with an activation function,
which is a smooth function that could be easily optimized using gradient descent. 

step back, the human brian is not computing partition, rather than computing the probability of all possible states of the environment
the brain use heuristic to approximate the most likely states of the environment, then make the decision based on the approximation.











Most of groundbreaking achievements should be credited to certain learning algorithms, such as supervised learning,
Reinforcement learning, and self-supervised learning. 



trained on massive amount of data, 




1. efficiency
2. motivation















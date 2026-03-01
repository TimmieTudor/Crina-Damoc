# Normalization in Crina

Hello, Dr. Timmie here. I'm writing this note to document the normalization strategy in Crina.

## Why Normalization?

Normalization is a technique used to scale the activations of neurons in a neural network to a standard range. This helps to prevent the activations from becoming too large or too small, which can lead to a number of problems, including:

- Vanishing gradients
- Exploding gradients
- Slow convergence
- Poor generalization

## Types of Normalization

There are several types of normalization that can be used in neural networks, including:

- Batch normalization
- Layer normalization
- RMS normalization
- Instance normalization

## Normalization in Crina

In Crina, we use RMS normalization to scale the activations of neurons in the network. RMS normalization is a type of normalization that is similar to layer normalization, but it is computed over the root mean square of the activations rather than the mean and variance. Its performance is comparable to layer normalization, but it is computationally more efficient. It is also known as Root Mean Square Layer Normalization (RMSNorm).

## Implementation Details

In Crina, specifically the tree attention mechanism, we didn't use any normalization layers initially. However, we found that the model was not converging properly, gradients were really unstable, and the loss was quite high. So we started with a small step: adding RMSNorm at the end of the leaf mixing layer. This helped to stabilize the gradients and improve the model's performance a bit. This experiment is still ongoing, but the results are promising. We also added RMSNorm to the up-sweep and down-sweep layers, and we're seeing similar performance improvements, but with more stable gradients. 
# McCulloch-Pitts Neuron

This module implements the McCulloch-Pitts neuron, the earliest
mathematical model of a biological neuron.

## Model Definition

The neuron computes:

y = 1 if Σ w_i x_i ≥ θ  
    0 otherwise

This makes it a linear threshold unit.

## Capabilities

- Implements AND, OR, NAND logic gates
- Deterministic, no learning

## Limitations

- Cannot represent XOR
- Only linearly separable functions are computable

## Significance

The MP neuron laid the foundation for:
- Perceptrons
- Artificial neural networks
- Modern deep learning

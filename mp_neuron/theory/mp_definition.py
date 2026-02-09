"""
McCulloch-Pitts Neuron (1943)

y = 1 if Σ w_i x_i >= θ
    0 otherwise

This is a binary threshold unit.
"""

def mp_equation(inputs, weights, threshold):
    """
    Pure mathematical definition of MP neuron.
    """
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    return 1 if weighted_sum >= threshold else 0

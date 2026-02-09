class MPNeuron:
    """
    McCulloch-Pitts Neuron implementation
    """

    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def forward(self, inputs):
        """
        Computes neuron output
        """
        weighted_sum = 0
        for x, w in zip(inputs, self.weights):
            weighted_sum += x * w

        return 1 if weighted_sum >= self.threshold else 0

from mp_neuron.model.mp_neuron import MPNeuron

def test_xor():
    """
    XOR cannot be implemented by a single MP neuron
    """
    neuron = MPNeuron(weights=[1, 1], threshold=1)

    inputs = [(0,0), (0,1), (1,0), (1,1)]
    print("XOR Attempt")
    for x in inputs:
        print(x, "->", neuron.forward(x))

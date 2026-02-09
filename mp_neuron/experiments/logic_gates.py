from mp_neuron.model.mp_neuron import MPNeuron

def test_and():
    neuron = MPNeuron(weights=[1, 1], threshold=2)
    inputs = [(0,0), (0,1), (1,0), (1,1)]

    print("AND Gate")
    for x in inputs:
        print(x, "->", neuron.forward(x))


def test_or():
    neuron = MPNeuron(weights=[1, 1], threshold=1)
    inputs = [(0,0), (0,1), (1,0), (1,1)]

    print("\nOR Gate")
    for x in inputs:
        print(x, "->", neuron.forward(x))


def test_nand():
    neuron = MPNeuron(weights=[-1, -1], threshold=-1)
    inputs = [(0,0), (0,1), (1,0), (1,1)]

    print("\nNAND Gate")
    for x in inputs:
        print(x, "->", neuron.forward(x))

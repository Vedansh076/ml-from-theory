from mp_neuron.experiments.logic_gates import (
    test_and,
    test_or,
    test_nand
)
from mp_neuron.experiments.limitations import test_xor


def main():
    test_and()
    test_or()
    test_nand()
    test_xor()


if __name__ == "__main__":
    main()

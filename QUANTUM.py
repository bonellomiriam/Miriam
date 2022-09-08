from pytket import circuit
import numpy as np 


def build_ghz_circuit(x):
    circ = Circuit(x)
    circ.H(0)
    for i in range(x-1):
        i+=1
        circ.cs(0,i)
    print(circ.get_commands())

input =int(input('The number of qubits:'))
build_ghz_circuit(input)

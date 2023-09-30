# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the schmidt.py module.
"""

from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit
from qclib.compression import SchmidtCompressor
from qclib.state_preparation import LowRankInitialize
from qclib.util import get_state

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestSchmidt(TestCase):

    def test_exact(self):
        n_qubits = 6
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        initializer = LowRankInitialize(state_vector)

        partition = [0,2,4]
        compressor = SchmidtCompressor(state_vector, opt_params={'partition': partition})
        decompressor = compressor.inverse()

        circuit = QuantumCircuit(n_qubits)
        circuit.append(initializer.definition, list(range(n_qubits)))
        circuit.append(compressor.definition, list(range(n_qubits)))
        circuit.reset(compressor.reset_qubits)
        circuit.append(decompressor.definition, list(range(n_qubits)))

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

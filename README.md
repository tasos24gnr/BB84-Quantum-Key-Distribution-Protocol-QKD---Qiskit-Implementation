# BB84-Quantum-Key-Distribution-Protocol-QKD---Qiskit-Implementation

This repository provides a fully functional implementation of the BB84 quantum key distribution protocol using [Qiskit](https://qiskit.org/). The BB84 protocol allows two parties, Alice and Bob, to securely generate a shared cryptographic key over a quantum channel, leveraging the fundamental principles of quantum mechanics.

---

## 🌐 Project Scope
This project supports execution on:
- **Ideal (noiseless) quantum simulators**
- **Noise-augmented simulators** based on real IBM hardware (e.g., FakeManila, FakeBelem)
- **Real IBM Quantum hardware** using Qiskit Runtime

The implementation demonstrates:
- Key generation through quantum state preparation and measurement
- Basis reconciliation and sifting
- Quantum Bit Error Rate (QBER) estimation

> ❗ This implementation does **not** include post-processing (error correction or privacy amplification), as the focus is on protocol execution and behavior under varying noise conditions.

---

## 🔍 Features
- Modular and reproducible protocol logic
- Clear random seeding for testing
- Three execution modes: ideal, noisy, real
- Histogram visualizations of outcomes (multi-shot mode)
- Full circuit generation and display
- Optional backend switching via configuration
- Automatic QBER calculation and key agreement validation

---

## 📂 Project Structure


 ## Protocol Workflow
The script implements the full BB84 workflow:
1. **Random Bit Generation**: Alice and Bob generate random bits and basis choices.
2. **Qubit Encoding**: Alice encodes her bits using either Z- or X-basis.
3. **Quantum Transmission**: Bob measures the incoming qubits using his basis choices.
4. **Basis Reconciliation**: Alice and Bob retain only bits where their bases matched.
5. **Key Sifting**: Sifted keys are compared, and QBER is calculated.

## Code Components
### Encoding and Measurement
- `encode_message(bits, bases)`: Encodes bits into quantum states using basis-specific transformations.
- `measure_message(circuits, bases)`: Adds measurement operations based on Bob’s bases, transforming X-basis into Z-basis when necessary.

### Simulation and Execution
- `bb84_protocol(n_bits, seed, use_simulator)`: Orchestrates the full protocol and routes execution to the appropriate backend.
- `run_with_noise_model(n_bits, seed, backend_name)`: Uses predefined noise models from fake backends to simulate realistic quantum behavior.

### Post-Processing
- `remove_garbage(a_bases, b_bases, bits)`: Sifts bits based on basis agreement.
- `create_full_circuit(...)`: Builds a comprehensive circuit representation.
- `analyze_results(results)`: Computes QBER, compares keys, and prints detailed diagnostics.

## Execution Modes
By modifying the `__main__` block, users can easily toggle between simulation modes and real hardware execution. The script includes safeguards and job status prompts for remote runs.


---

## 🧪 How to Run
```bash
python bb84.py
```
You can toggle between backends (simulator vs real hardware) inside the script by changing the `run_on_hardware` flag.

> **NOTE**: To run on IBM Quantum hardware, you must:
> - Get an API token from [quantum.ibm.com](https://quantum.ibm.com/)
> - Save your token in the script (`QiskitRuntimeService.save_account(...)`)

---

## 📊 Example Experiments
- **Experiment 1**: 128 qubits, 1 shot → Evaluate noise impact via QBER
- **Experiment 2**: 8 qubits, 100 shots → Visualize statistical behavior of quantum measurement

Results include histograms, sifted keys, and circuit diagrams. See `results/` for output figures.

---

## 📘 Report & Documentation
This repository accompanies a formal report that discusses:
- Theoretical background of BB84
- Execution methodology
- Analysis of results from simulators and real hardware
- Interpretation of QBER and key agreement

---

## 📚 References
- Bennett, C. H., & Brassard, G. (1984). Quantum cryptography: Public key distribution and coin tossing.
- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*.
- Scarani, V., et al. (2009). The security of practical quantum key distribution. *Reviews of Modern Physics*.
- Preskill, J. (1998). Lecture Notes on Quantum Computation.
- [Qiskit Documentation](https://qiskit.org/documentation/)

---

## 💡 License
GNU GPLv3
---

## 🤝 Contributions
Pull requests are welcome. Feel free to open issues or suggest improvements!

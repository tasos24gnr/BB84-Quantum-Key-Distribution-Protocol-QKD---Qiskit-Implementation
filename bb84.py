"""
BB84 Quantum Key Distribution Protocol implemented using Qiskit.

Overview:
This script provides a complete implementation of the BB84 QKD protocol,
enabling two parties (Alice and Bob) to securely establish a shared secret key
based on the principles of quantum mechanics. The implementation supports three
execution modes: an ideal simulator, a noise-augmented simulator based on real
quantum hardware models, and actual quantum hardware via IBM Quantum.

The protocol consists of several stages: Alice prepares quantum states
according to randomly chosen bits and bases; Bob measures those states using
his own randomly chosen bases. They then publicly compare bases and retain only
the outcomes where both bases matched. This subset forms the shared secret key.


Inputs:
- n_bits: The number of qubits (bits) to exchange.
- seed: A seed for the random number generator to ensure reproducibility.
- use_simulator: Boolean flag indicating whether to use a local simulator or
  a real quantum backend.

Outputs:
- A dictionary containing all relevant data: Alice's and Bob's bit strings and
  bases, Bob's measurement results, the sifted keys, the job ID (or a label
  for simulation), and the full quantum circuit representation.

Note:
This script does not include error correction or privacy amplification. It is
intended as a  demonstration of BB84 core concepts with optional
realistic noise simulation.
"""


from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
import numpy as np
from qiskit_aer.noise import NoiseModel
from qiskit.visualization import plot_histogram
import os

# Attempt to import fake backends based on the Qiskit version
try:
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeBelemV2
except ImportError:
    try:
        from qiskit.providers.fake_provider import FakeManilaV2, FakeBelemV2
    except ImportError:
        from qiskit.providers.fake_provider import FakeManila, FakeBelem
        FakeManilaV2 = FakeManila
        FakeBelemV2 = FakeBelem


# ---------------------------------------------------------------------------
# IBM Quantum account configuration:
# ---------------------------------------------------------------------------

# FOR SIMPLICITY I HARDCOPY MY TOKEN HERE , IT IS NOT A GOOD PRACTICE
QiskitRuntimeService.save_account(
    token="mytoken",
    channel="ibm_quantum",
    overwrite=True
)



def encode_message(bits, bases):
    """
    Prepare individual one-qubit circuits encoding Alice's bits in chosen bases.

    Quantum encoding in BB84 protocol:
    - Z-basis (basis=0): |0⟩ and |1⟩ states encode bits 0 and 1 respectively.
      Implementation: If bit is 1, apply an X gate to flip |0⟩ to |1⟩.
    - X-basis (basis=1): |+⟩ and |−⟩ states encode bits 0 and 1 respectively.
      Implementation: Start with |0⟩, apply H gate for |+⟩ (bit=0).
      For bit=1, apply X then H to get |−⟩.

    Args:
        bits (list or np.array): Alice's randomly generated bit string (0s and 1s).
        bases (list or np.array): Alice's randomly chosen bases (0 for Z, 1 for X).

    Returns:
        List of QuantumCircuit objects each representing one encoded qubit.
    """
    circuits = []
    for bit, basis in zip(bits, bases):
        # Create a single qubit circuit with one classical bit for measurement
        qc = QuantumCircuit(1, 1)

        if basis == 0:
            # Z-basis encoding: |0> or |1>
            if bit == 1:
                qc.x(0)  # Flip qubit to |1> if bit is 1
        else:
            # X-basis encoding: |+> or |->
            if bit == 0:
                qc.h(0)  # |+> = H|0>
            else:
                qc.x(0)  # Flip to |1>
                qc.h(0)  # Then apply H to get |->

        circuits.append(qc)

    return circuits


def measure_message(circuits, bases):
    """
    Append measurement operations to the prepared qubit circuits in Bob's chosen bases.

    Measurement in BB84:
    - Bob chooses a basis (Z=0 or X=1) randomly for each qubit he receives.
    - If X-basis measurement is chosen, apply Hadamard gate before measuring to rotate
      the qubit back to Z-basis measurement frame.
    - Measure the qubit in the computational basis.

    Args:
        circuits (list of QuantumCircuit): Circuits prepared by Alice.
        bases (list or np.array): Bob's randomly chosen measurement bases.

    Returns:
        List of QuantumCircuit objects with measurement instructions added.
    """
    measured_circuits = []

    for qc, basis in zip(circuits, bases):
        # Make a copy to avoid modifying original Alice's circuits
        m_qc = qc.copy()

        if basis == 1:
            # To measure in X-basis, rotate qubit state with H gate
            m_qc.h(0)

        # Measure the qubit into classical bit 0
        m_qc.measure(0, 0)
        measured_circuits.append(m_qc)

    return measured_circuits


def remove_garbage(a_bases, b_bases, bits):
    """
    Perform the sifting step to retain only bits where Alice's and Bob's bases agree.

    The key idea in BB84:
    - Only those bits where Alice and Bob chose the same measurement basis
      are kept as part of the secret key.
    - Bits measured in mismatched bases are discarded as they carry no useful information.

    Args:
        a_bases (list or np.array): Alice's bases.
        b_bases (list or np.array): Bob's bases.
        bits (list or np.array): Corresponding bits (Alice's or Bob's measured bits).

    Returns:
        List of sifted bits where bases match.
    """
    return [bit for i, bit in enumerate(bits) if a_bases[i] == b_bases[i]]


def create_full_circuit(alice_bits, alice_bases, bob_bases):
    """
    Constructs a single QuantumCircuit that encodes all qubits on separate wires,
    then applies measurement according to Bob's bases.

    This circuit provides a compact representation of the entire BB84 process for n bits:
    - For each qubit:
        - Encode bit with Alice's basis.
        - Apply Bob's measurement basis transformations.
        - Measure qubit.
    - Uses one qubit and one classical bit per position.

    Args:
        alice_bits (list or np.array): Alice's bit string.
        alice_bases (list or np.array): Alice's bases.
        bob_bases (list or np.array): Bob's bases.

    Returns:
        QuantumCircuit object representing full BB84 exchange for n bits.
    """
    n = len(alice_bits)
    qc = QuantumCircuit(n, n)

    # Alice's encoding on each qubit line
    for i in range(n):
        if alice_bases[i] == 0:
            # Z-basis encoding
            if alice_bits[i] == 1:
                qc.x(i)
        else:
            # X-basis encoding
            if alice_bits[i] == 0:
                qc.h(i)
            else:
                qc.x(i)
                qc.h(i)

    qc.barrier()  # Visual barrier separating encoding and measurement phases

    # Bob's measurement basis transformation
    for i in range(n):
        if bob_bases[i] == 1:
            qc.h(i)  # Rotate X-basis measurements back to Z-basis

    qc.measure(range(n), range(n))  # Measure all qubits to classical bits

    return qc


def bb84_protocol(n_bits, seed, use_simulator=True):
    """
       Runs the entire BB84 protocol simulation or execution on real quantum hardware.

    Workflow:
    1. Generate random bits and bases for Alice.
    2. Encode qubits according to Alice's bits and bases.
    3. Generate random bases for Bob.
    4. Apply Bob's measurement bases to the prepared qubits.
    5. Run the circuits on a backend (simulator or real quantum device).
    6. Collect results and sift keys based on matching bases.

    Args:
        n_bits (int): Number of qubits to exchange.
        seed (int or None): Random seed for reproducibility. If None, uses random seed each time.
        use_simulator (bool): True to run locally on AerSimulator,
                              False to run on IBM Quantum backend.

    Returns:
        dict: Contains Alice's and Bob's bits and bases, Bob's measurement results,
              sifted keys, job identifier, and full combined circuit.
    """
    # Set random seed - if None, numpy will use current time as seed
    if seed is not None:
        np.random.seed(seed)
    # If seed is None, numpy uses a random seed based on current time

    # Step 1: Alice chooses random bits and bases
    alice_bits = np.random.randint(2, size=n_bits)  # Random 0/1 bits
    alice_bases = np.random.randint(2, size=n_bits)  # Random basis choice (0=Z,1=X)

    # Step 2: Prepare qubit circuits for Alice's bits/bases
    alice_prepared_circuits = encode_message(alice_bits, alice_bases)

    # Step 3: Bob chooses random measurement bases
    bob_bases = np.random.randint(2, size=n_bits)

    # Step 4: Add measurement in Bob's bases
    measured_circuits = measure_message(alice_prepared_circuits, bob_bases)

    # Step 5: Execute circuits on chosen backend
    if use_simulator:
        # Use AerSimulator locally for fast execution and debugging
        simulator = AerSimulator()
        # Compile circuits for the simulator
        transpiled_circuits = transpile(measured_circuits, simulator)
        job = simulator.run(transpiled_circuits, shots=1)  # One shot since only one bit needed
        result = job.result()
        # Retrieve counts (measurement results) for each circuit
        counts_list = [result.get_counts(circ) for circ in transpiled_circuits]
        job_id = "simulator"
    else:
        # Use IBM Quantum real device backend
        try:
            service = QiskitRuntimeService(channel="ibm_quantum")

            # List available backends and choose one
            backends = service.backends()
            print("Available backends:")
            for backend in backends:
                print(f"  - {backend.name}: {backend.num_qubits} qubits, Status: {backend.status().operational}")

            # Choose a backend
            backend_name = "ibm_brisbane"
            backend = service.backend(backend_name)

            print(f"\nUsing backend: {backend_name}")
            print(f"Queue length: {backend.status().pending_jobs}")

            # Create Sampler for the backend
            sampler = Sampler(backend)

            # Compile circuits for the backend device
            transpiled_circuits = transpile(measured_circuits, backend)

            # Run with more shots for real hardware (noise requires statistics)
            print("Submitting job to quantum computer...")
            job = sampler.run(transpiled_circuits, shots=1024)  # Use more shots for real hardware

            print(f"Job submitted with ID: {job.job_id()}")
            print("Waiting for results... (this may take several minutes)")

            result = job.result()

            # For real hardware with multiple shots, take the most frequent result
            counts_list = []
            for res in result:
                counts = res.data.c.get_counts()
                # Get the most frequent measurement result
                counts = res.data.c.get_counts()
                most_frequent = max(counts, key=counts.get)
                counts_list.append({most_frequent: counts[most_frequent]})


            job_id = job.job_id()

        except Exception as e:
            print(f"Error connecting to IBM Quantum: {e}")
            print("Make sure you have:")
            print("1. Valid IBM Quantum account")
            print("2. Correct API token saved")
            print("3. Internet connection")
            raise

    # Step 6: Decode Bob's measurement results
    # Extract the bit (0 or 1) from the counts dictionary keys (bitstrings)
    bob_results = [int(list(counts.keys())[0], 2) for counts in counts_list]



    # Save histogram to file
    all_counts = {}
    for i, counts in enumerate(counts_list):
        for key, val in counts.items():
            label = f"q{i}:{key}"
            all_counts[label] = val

    fig = plot_histogram(all_counts, title="IBM Hardware Measurement Results")
    filename = f"ibm_{n_bits}bits_histogram.png"
    save_path = os.path.join(os.getcwd(), filename)
    fig.savefig(save_path)
    print(f"Histogram saved to: {save_path}")

    # Step 7: Sift keys - keep bits where Alice and Bob's bases matched
    alice_key = remove_garbage(alice_bases, bob_bases, alice_bits)
    bob_key = remove_garbage(alice_bases, bob_bases, bob_results)

    # Step 8: Build a full circuit representing the entire BB84 exchange
    full_circuit = create_full_circuit(alice_bits, alice_bases, bob_bases)

    # Return all relevant data for analysis
    return {
        'alice_bits': alice_bits.tolist(),
        'alice_bases': alice_bases.tolist(),
        'bob_bases': bob_bases.tolist(),
        'bob_results': bob_results,
        'alice_key': alice_key,
        'bob_key': bob_key,
        'job_id': job_id,
        'full_circuit': full_circuit
    }


def run_with_noise_model(n_bits, seed, backend_name='manila'):


    if backend_name.lower() == 'manila':
        try:
            fake_backend = FakeManilaV2()
            backend_label = 'FakeManilaV2'
        except NameError:
            fake_backend = FakeManila()
            backend_label = 'FakeManila'
    elif backend_name.lower() == 'belem':
        try:
            fake_backend = FakeBelemV2()
            backend_label = 'FakeBelemV2'
        except NameError:
            fake_backend = FakeBelem()
            backend_label = 'FakeBelem'
    else:
        try:
            fake_backend = FakeManilaV2()
            backend_label = 'FakeManilaV2 (default)'
        except NameError:
            fake_backend = FakeManila()
            backend_label = 'FakeManila (default)'

    noise_model = NoiseModel.from_backend(fake_backend)
    noisy_sim = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)

    if seed is not None:
        np.random.seed(seed)
    alice_bits = np.random.randint(2, size=n_bits)
    alice_bases = np.random.randint(2, size=n_bits)
    bob_bases = np.random.randint(2, size=n_bits)

    circuits = encode_message(alice_bits, alice_bases)
    measured_circuits = measure_message(circuits, bob_bases)
    transpiled = transpile(measured_circuits, noisy_sim)

    job = noisy_sim.run(transpiled, shots=1024)
    result = job.result()

    counts_list = [result.get_counts(circ) for circ in transpiled]
    bob_results = [int(max(c, key=c.get), 2) for c in counts_list]


    # Save histogram to file
    all_counts = {}
    for i, counts in enumerate(counts_list):
        for key, val in counts.items():
            label = f"q{i}:{key}"
            all_counts[label] = val

    fig = plot_histogram(all_counts, title="Noisy Simulation Measurement Results")
    filename = f"{backend_label}_{n_bits}bits_histogram.png"
    save_path = os.path.join(os.getcwd(), filename)
    fig.savefig(save_path)
    print(f"Histogram saved to: {save_path}")

    alice_key = remove_garbage(alice_bases, bob_bases, alice_bits)
    bob_key = remove_garbage(alice_bases, bob_bases, bob_results)

    full_circuit = create_full_circuit(alice_bits, alice_bases, bob_bases)

    return {
        'alice_bits': alice_bits.tolist(),
        'alice_bases': alice_bases.tolist(),
        'bob_bases': bob_bases.tolist(),
        'bob_results': bob_results,
        'alice_key': alice_key,
        'bob_key': bob_key,
        'job_id': f'noisy_simulator ({backend_label})',
        'full_circuit': full_circuit
    }


def analyze_results(results):
    """
     Print detailed intermediate values and analyze final keys for agreement.

    Actions:
    - Print Alice's bits and bases.
    - Print Bob's bases and measurement results.
    - Show sifted keys after basis reconciliation.
    - Check if keys match exactly.
    - If keys mismatch, calculate Quantum Bit Error Rate (QBER).
    - Display job id and full circuit for reference.

    Args:
        results (dict): Output from bb84_protocol containing all intermediate and final data.
    """
    print("Alice's bits:   ", results['alice_bits'])
    print("Alice's bases:  ", results['alice_bases'])
    print("Bob's bases:    ", results['bob_bases'])
    print("Bob's results:  ", results['bob_results'])
    print("\nSifted keys:")
    print("  Alice's key:", results['alice_key'])
    print("  Bob's key:  ", results['bob_key'])

    if results['alice_key'] == results['bob_key']:
        print("\n✅ Success: Keys match exactly!")
    else:
        print("\n⚠️  Warning: Keys do not match.")
        # Find indices and differing bit values in keys
        mismatches = [
            (i, a, b) for i, (a, b) in
            enumerate(zip(results['alice_key'], results['bob_key'])) if a != b
        ]
        for i, a, b in mismatches:
            print(f"  Index {i}: Alice={a}, Bob={b}")
        # Calculate QBER = fraction of differing bits in sifted key
        qber = len(mismatches) / len(results['alice_key']) if results['alice_key'] else 0
        print(f"Quantum Bit Error Rate: {qber:.2%}")

    print(f"\nJob ID: {results['job_id']}")
    print("\nFull circuit:")
    print(results['full_circuit'])


# -----------------------------------------------------------------------------
# Script entry point to run protocol tests on simulator and real hardware.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("🔬 Running BB84 with 8 bits on ideal simulator (random seed)...\n")
    sim_results = bb84_protocol(n_bits=8, seed=None, use_simulator=True)
    analyze_results(sim_results)

    print("\n" + "=" * 70)
    print("🧪 Running BB84 with 8 bits on noisy simulator (random seed)...\n")
    try:
        noisy_results = run_with_noise_model(n_bits=8, seed=None, backend_name='manila')
        analyze_results(noisy_results)
    except Exception as e:
        print(f"Error with Manila backend: {e}")
        print("Trying Belem backend instead...")
        try:
            noisy_results = run_with_noise_model(n_bits=8, seed=None, backend_name='belem')
            analyze_results(noisy_results)
        except Exception as e:
            print(f"Error with Belem backend: {e}")
            print("Noisy simulation failed. Please check your Qiskit installation.")


    print("\n" + "=" * 70)
    print("⚛️ Ready to run on IBM Quantum hardware!")
    print("To enable real quantum computer execution:")
    print("1. Get your API token from quantum.ibm.com")
    print("2. Uncomment and update the QiskitRuntimeService.save_account() section above")
    print("3. Uncomment the code below")
    print()

    #  IBM Quantum access configured:
    run_on_hardware = True  # Set to True when ready

    if run_on_hardware:
        try:
            print("🚀 Running BB84 on real quantum hardware...")
            hw_results = bb84_protocol(n_bits=8, seed=None, use_simulator=False)  # Start with fewer qubits
            analyze_results(hw_results)
        except Exception as e:
            print(f"Hardware execution failed: {e}")
            print("Check your IBM Quantum setup and try again.")
    else:
        print("Set run_on_hardware = True above to execute on real quantum computers")
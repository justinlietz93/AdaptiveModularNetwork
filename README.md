# AMN Prototype (`amn_prototype.py`)

**Status: Early Prototype / Proof-of-Concept (Precursor to FUM)**

## Overview

This Python script (`amn_prototype.py`) implements an early prototype of the **Adaptive Modular Network (AMN)** concept. AMN was an initial exploration into brain-inspired AI architectures, aiming to combine:

* Modular Spiking Neural Networks (SNNs) using Leaky Integrate-and-Fire (LIF) neurons.
* Local synaptic plasticity based on a simplified STDP-like rule.
* A global Coordinator Network using traditional neural networks (MLP) and reinforcement learning principles (Bernoulli sampling) to dynamically manage connections *between* modules.
* A basic Self-Improvement Engine (SIE) for simple meta-learning (adjusting the STDP learning rate).

This prototype served as a proof-of-concept to test the feasibility of integrating these disparate components (PyTorch, Norse SNNs, custom rules) and achieving basic learning on simple tasks. The learnings and identified limitations from this AMN prototype directly informed the design of the subsequent, more advanced **Fully Unified Model (FUM)** project.

## Key Components Implemented in this Script

* **`ModularUnit`:** Represents an SNN module using `norse.torch.LIFRecurrent` neurons. Includes internal weights updated by a simplified, correlation-based `apply_stdp` method (Note: not standard time-difference STDP). Designed to run on `device_secondary` (e.g., AMD 7900 XTX).
* **`CoordinatorPolicyNetwork`:** A standard PyTorch MLP (`nn.Sequential`) trained via backpropagation (using `Adam` optimizer). It takes flattened unit activity as input and outputs probabilities for inter-unit connections. Uses `torch.distributions.Bernoulli` to sample a binary connection matrix. Designed to run on `device_primary` (e.g., AMD MI100).
* **`AMN`:** The main class that integrates multiple `ModularUnit` instances and the `CoordinatorPolicyNetwork`. It manages the forward pass (processing units, getting connections, combining outputs) and the `train_step` (applying backprop to coordinator and STDP to units sequentially). *Includes specific heuristics (a `direct_mapping` rule and output boosting logic) to facilitate learning on the simple demonstration task.*
* **`SelfImprovementEngine`:** A basic version that monitors the training loss trend and increases the `stdp_learning_rate` for all units if improvement stalls.

## Functionality Demonstrated

This script focuses on training and evaluating the AMN prototype on a simple task: learning the mapping `f(x) = x + 2`.
* Numbers are encoded into spike trains based on firing rate (`encode_number`).
* The network is trained using MSE loss between the mean firing rate of the output and the target rate.
* The `main` function executes the training loop, tracks the best model state based on output rate accuracy, runs simple generalization tests (`run_tests`), evaluates success based on predefined criteria (`evaluate_results`), and saves the model.
* It successfully demonstrates the complex integration of the different frameworks and the achievement of convergence on this specific task, running on the intended dual AMD GPU setup.

## Known Limitations & Prototype Status

This code represents an **early-stage exploration** and should be viewed as such. Its success on the simple task was aided by built-in heuristics (`direct_mapping`, output boosting). Key limitations that motivated the move to the FUM architecture include:
* The highly simplified STDP rule lacks true temporal sensitivity.
* The SIE is rudimentary and doesn't incorporate the multi-objective complexity planned for FUM.
* The explicit Coordinator Network represents a potential bottleneck and differs from FUM's emergent knowledge graph approach.
* The "duct tape and glue" integration of frameworks (as described in the AMN documentation) limits scalability and robustness.

## Dependencies

* PyTorch (`torch`)
* Norse (`norse.torch`)
* NumPy (`numpy`)

## Running the Code

1.  Ensure PyTorch, Norse, and NumPy are installed.
2.  Verify CUDA/ROCm setup for PyTorch corresponding to your AMD or NVIDIA GPUs.
3.  Configure `device_primary` and `device_secondary` variables at the top of the script to match your available GPU indices.
4.  Adjust other configuration parameters (`num_units`, `neurons_per_unit`, `timesteps`, `max_epochs`, learning rates) if desired.
5.  Execute the script: `python amn_prototype.py`
6.  Logs will be printed to the console (level configured at the top). The best model state (if training improves) or the final state will be saved to a `.pth` file (e.g., `amn_model_best.pth`).
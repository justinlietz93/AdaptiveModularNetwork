import torch
import torch.nn as nn
import norse.torch as norse
import logging
from torch.distributions import Bernoulli
from utils import ensure_tensor # Assuming utils.py is in the same directory

logger = logging.getLogger(__name__)

# --- Core AMN Components ---

# 1. Modular Unit: SNN module with local STDP learning
class ModularUnit(nn.Module):
    """Represents a single spiking neural network module using LIF neurons."""
    def __init__(self, neurons, timesteps, device):
        super(ModularUnit, self).__init__()
        self.device = device
        self.neurons = neurons
        self.timesteps = timesteps # Store timesteps
        logger.debug(f"Initializing ModularUnit with {neurons} neurons on {device} for {timesteps} timesteps")

        if not isinstance(neurons, int) or neurons <= 0:
            raise ValueError(f"neurons must be a positive integer, got {neurons}")
        if not isinstance(timesteps, int) or timesteps <= 0:
            raise ValueError(f"timesteps must be a positive integer, got {timesteps}")

        # Standard LIF neuron parameters for Norse
        self.lif_params = norse.LIFParameters(
            tau_mem_inv=1.0 / 20.0,  # Membrane time constant (20ms)
            tau_syn_inv=1.0 / 10.0,  # Synaptic time constant (10ms)
            v_leak=torch.tensor(-70.0, device=device, dtype=torch.float32), # Resting potential
            v_th=torch.tensor(-55.0, device=device, dtype=torch.float32),   # Firing threshold
            v_reset=torch.tensor(-70.0, device=device, dtype=torch.float32) # Reset potential after spike
        )

        # Initialize Norse's recurrent LIF layer
        try:
            self.lif = norse.LIFRecurrent(
                input_size=neurons,  # Each neuron can receive input from others in the unit
                hidden_size=neurons, # Recurrent connection size matches neuron count
                p=self.lif_params
            ).to(device)
            # Test initialization with dummy input
            with torch.no_grad():
                 dummy_output, _ = self.lif(torch.zeros(1, neurons, device=device))
                 logger.debug(f"LIFRecurrent test output shape: {dummy_output.shape}")
            logger.debug("Successfully initialized LIFRecurrent")
        except Exception as e:
            logger.error(f"Error initializing LIFRecurrent: {str(e)}", exc_info=True)
            raise

        # Internal weights within the module, subject to STDP
        # Initialized randomly; higher initial values (0.3 vs 0.1) might speed up initial learning
        self.weights = torch.rand(neurons, neurons, device=device) * 0.3
        # Learning rate for the simplified STDP rule - will be set by SIE
        self.stdp_learning_rate = 0.03 # Default, can be overwritten (eta_effective)

        # --- FUM STDP Parameters ---
        self.A_plus = 0.1
        self.A_minus = 0.12
        self.tau_plus_inv = 1.0 / 20.0 # Inverse for efficiency (1/ms)
        self.tau_minus_inv = 1.0 / 20.0 # Inverse for efficiency (1/ms)
        self.gamma = 0.95 # Eligibility trace decay factor
        # Eligibility trace tensor, initialized in init or first call
        self.eligibility_trace = torch.zeros_like(self.weights)
        self.lambda_decay = 1e-5 # Weight decay coefficient (small default value)
        # --- End FUM STDP Parameters ---


    def forward(self, spikes_in):
        """Processes input spike train over `timesteps` using LIF dynamics."""
        logger.debug(f"ModularUnit FWD - Input shape: {spikes_in.shape}, Device: {spikes_in.device}")
        spikes_in = ensure_tensor(spikes_in, self.device)
        logger.debug(f"  Ensured Input Device: {spikes_in.device}") # Add check

        # --- Performance Optimization: Process sequence directly ---
        # Input shape should be [timesteps, neurons]
        # Add batch dimension: [timesteps, 1, neurons] for LIFRecurrent
        if spikes_in.dim() != 2 or spikes_in.shape[0] != self.timesteps or spikes_in.shape[1] != self.neurons:
             logger.warning(f"Unexpected input shape for sequence processing: {spikes_in.shape}. Expected [{self.timesteps}, {self.neurons}]")
             # Fallback or raise error? For now, try adding batch dim anyway.
             spikes_in_batch = spikes_in.unsqueeze(1)
        else:
             spikes_in_batch = spikes_in.unsqueeze(1) # Add batch dimension

        logger.debug(f"  Input shape to LIF layer (sequence): {spikes_in_batch.shape}")

        try:
            # Process the whole sequence through the LIF layer
            # Norse handles the recurrent state internally when processing sequences
            output_spikes_batch, _ = self.lif(spikes_in_batch) # state is implicitly managed
            logger.debug(f"  Output shape from LIF layer (sequence): {output_spikes_batch.shape}")

            # Remove batch dimension -> [timesteps, neurons]
            output_spikes = output_spikes_batch.squeeze(1)

        except RuntimeError as e:
            logger.error(f"RuntimeError in LIF sequence forward: {str(e)}", exc_info=True)
            logger.error(f"  Input shape was: {spikes_in_batch.shape}")
            raise
        # --- End Performance Optimization ---

        logger.debug(f"ModularUnit FWD - Output shape: {output_spikes.shape}, Output Device: {output_spikes.device}") # Add check
        return output_spikes

    def apply_stdp(self, pre_spikes, post_spikes):
        """
        Calculates STDP weight changes based on FUM's time-dependent rule
        and updates the eligibility trace.
        Assumes pre_spikes and post_spikes are [timesteps, neurons] tensors.
        """
        logger.debug(f"Applying FUM STDP - Pre shape: {pre_spikes.shape}, Post shape: {post_spikes.shape}")

        # Ensure tensors are on the correct device
        pre_spikes = pre_spikes.to(self.device)
        post_spikes = post_spikes.to(self.device)

        # --- Optimized STDP Calculation ---
        # Find indices (timestep, neuron_index) where spikes occurred
        pre_spike_indices = pre_spikes.nonzero(as_tuple=False)   # Shape [num_pre_spikes, 2]
        post_spike_indices = post_spikes.nonzero(as_tuple=False) # Shape [num_post_spikes, 2]

        if pre_spike_indices.numel() == 0 or post_spike_indices.numel() == 0:
            # No pre or post spikes, no STDP changes
            delta_w_step = torch.zeros_like(self.weights)
        else:
            # Extract times and neuron indices
            t_pre = pre_spike_indices[:, 0].float()
            i_pre = pre_spike_indices[:, 1]
            t_post = post_spike_indices[:, 0].float()
            j_post = post_spike_indices[:, 1]

            # Calculate all pairwise time differences using broadcasting
            # delta_t[k, l] = t_post[l] - t_pre[k]
            delta_t = t_post.unsqueeze(0) - t_pre.unsqueeze(1) # Shape [num_pre_spikes, num_post_spikes]

            # Calculate potential LTP and LTD values for all pairs
            ltp_pot = self.A_plus * torch.exp(-delta_t * self.tau_plus_inv)
            ltd_pot = -self.A_minus * torch.exp(delta_t * self.tau_minus_inv)

            # Apply masks based on time difference
            ltp_mask = delta_t > 0
            ltd_mask = delta_t < 0

            # Combine LTP and LTD contributions, zeroing out where masks are false
            delta_w_pairs = torch.zeros_like(delta_t)
            delta_w_pairs[ltp_mask] = ltp_pot[ltp_mask]
            delta_w_pairs[ltd_mask] = ltd_pot[ltd_mask] # Note: LTD is already negative

            # Map pair contributions back to the weight matrix delta_w_step[i, j]
            # This requires efficient indexing or scatter operations.
            # Using scatter_add_ for accumulation.
            delta_w_step = torch.zeros_like(self.weights)
            # Create indices for scatter_add: pre_indices (i) repeated, post_indices (j) tiled
            pre_indices_expanded = i_pre.unsqueeze(1).expand(-1, j_post.shape[0]) # Shape [num_pre, num_post]
            post_indices_expanded = j_post.unsqueeze(0).expand(i_pre.shape[0], -1) # Shape [num_pre, num_post]

            # Flatten indices and values for scatter_add
            # We need to accumulate delta_w_pairs[k, l] into delta_w_step[i_pre[k], j_post[l]]
            # scatter_add_ needs 1D index tensor. We map (i, j) to flat index: i * num_neurons + j
            flat_indices = pre_indices_expanded * self.neurons + post_indices_expanded
            # Use index_add_ which is often more robust than scatter_add_ for this
            delta_w_step.view(-1).index_add_(0, flat_indices.view(-1), delta_w_pairs.view(-1))

            # Mask out self-connections (i == j)
            # Create a mask for diagonal elements
            diag_mask = torch.eye(self.neurons, device=self.device, dtype=torch.bool)
            delta_w_step[diag_mask] = 0.0

            # TODO: Add logic for inhibitory STDP rules if neuron types are introduced

        # Update eligibility trace: e(t) = gamma * e(t-1) + delta_w(t)
        # Ensure trace is on the correct device
        self.eligibility_trace = self.eligibility_trace.to(self.device)
        self.eligibility_trace = self.gamma * self.eligibility_trace + delta_w_step
        logger.debug(f"Updated eligibility trace. Mean trace value: {self.eligibility_trace.mean().item():.6f}")

        # Note: Weights are NOT updated here, only in update_weights()

    def update_weights(self, reward):
        """Applies the final weight update based on reward and eligibility trace."""
        # Ensure reward is a scalar or compatible tensor
        if isinstance(reward, torch.Tensor):
            reward_val = reward.item() # Use item() if it's a single-element tensor
        else:
            reward_val = reward # Assume scalar

        # --- Implement FUM Reward Modulation (Sec C.7) ---
        # Map reward to modulation factor [-1, 1]
        # Ensure reward_val is a tensor for sigmoid
        reward_tensor = torch.tensor(reward_val, device=self.device, dtype=torch.float32)
        mod_factor = 2 * torch.sigmoid(reward_tensor) - 1
        # Calculate effective learning rate
        eta_effective = self.stdp_learning_rate * (1 + mod_factor)
        # Calculate final weight delta (quadratic scaling) including weight decay
        reward_modulated_trace = eta_effective * reward_tensor * self.eligibility_trace
        weight_decay_term = -self.lambda_decay * self.weights # L2 decay term
        delta_w = reward_modulated_trace + weight_decay_term
        # --- End FUM Reward Modulation ---

        # Apply weight change
        self.weights += delta_w
        # Clamp weights
        self.weights.clamp_(-1, 1) # Clamp between -1 and 1 as per FUM docs
        logger.debug(f"Updated weights using reward {reward_val:.4f}. Mean delta_w: {delta_w.mean().item():.6f}")

        # Reset trace after weight update? FUM docs imply trace persists but is used by reward.
        # Let's not reset for now, allowing accumulation across steps if needed.
        # self.eligibility_trace.zero_()


# 2. Global Coordinator: Determines inter-unit connections using a standard NN
class CoordinatorPolicyNetwork(nn.Module):
    """A standard MLP that outputs connection probabilities between units."""
    def __init__(self, num_units, neurons_per_unit, device):
        super(CoordinatorPolicyNetwork, self).__init__()
        self.device = device
        self.num_units = num_units
        self.neurons_per_unit = neurons_per_unit
        # --- FIX: Update input_size based on full spatio-temporal pattern ---
        # Assuming timesteps is accessible or passed; for now, hardcoding based on config
        # TODO: Pass timesteps to Coordinator init if it becomes variable
        timesteps = 50 # Hardcoded based on current config in train_amn_prototype.py
        input_size = num_units * timesteps * neurons_per_unit # Flattened full spike pattern
        logger.info(f"CoordinatorPolicyNetwork initialized with input size: {input_size}")
        # --- End FIX ---
        hidden_size = 128 # Keep hidden size for now, might need adjustment
        output_size = num_units * num_units # Output one probability per potential connection

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid() # Output probabilities between 0 and 1
        ).to(self.device) # Runs on the specified device

    def forward(self, activity):
        """Calculates connection probabilities based on unit activity."""
        logger.debug(f"Coordinator FWD - Input activity shape: {activity.shape}")
        original_batch_size = activity.size(0) if activity.dim() > 1 else 1

        # Reshape activity to [batch_size, flattened_features] for Linear layers
        if activity.dim() == 1:
             activity = activity.unsqueeze(0) # Add batch dim if needed
        # Flatten features if necessary (e.g., if input is [batch, units, neurons])
        # Input 'activity' is now expected to be already flattened [1, features] from AMN.forward
        activity_flat = activity # Assume input is already [1, features]
        # Check if input needs flattening (e.g., if batch > 1 in future)
        if activity.dim() > 2:
             activity_flat = activity.view(original_batch_size, -1)
        elif activity.dim() == 1: # Add batch dim if somehow lost
             activity_flat = activity.unsqueeze(0)

        logger.debug(f"  Coordinator input shape (should be flat): {activity_flat.shape}")

        # Pass through the MLP
        connection_probs_flat = self.actor(activity_flat) # Output: [batch_size, num_units * num_units]
        logger.debug(f"  Output probs shape (flat): {connection_probs_flat.shape}")

        # Reshape probabilities back into a connection matrix format
        connection_probs = connection_probs_flat.view(original_batch_size, self.num_units, self.num_units)
        logger.debug(f"  Output probs shape (reshaped): {connection_probs.shape}")

        return connection_probs

    def get_action(self, activity):
        """Samples a binary connection matrix based on probabilities."""
        # Ensure activity is on the correct device
        activity = ensure_tensor(activity, device=self.device)
        logger.debug(f"Coordinator ACTION - Input activity shape: {activity.shape}")

        # Get probabilities from the network
        probs = self.forward(activity) # Shape: [batch_size, num_units, num_units]
        logger.debug(f"  Probabilities shape: {probs.shape}")

        # Use Bernoulli distribution to sample binary actions (connect or not connect)
        dist = Bernoulli(probs)
        action_matrix = dist.sample() # Shape: [batch_size, num_units, num_units]
        log_probs = dist.log_prob(action_matrix) # Log probabilities for potential policy gradient updates

        logger.debug(f"  Sampled action matrix shape: {action_matrix.shape}")
        # Return batch dimension even if batch size is 1
        return action_matrix, log_probs


# 4. Self-Improvement Engine (Basic version) - Renumbered from original script
class SelfImprovementEngine:
    """Adjusts STDP learning rate based on performance trend."""
    def __init__(self, amn_network, initial_lr=0.03, update_interval=2, improvement_threshold=0.001):
        self.amn = amn_network # Reference to the main AMN model
        self.performance_history = []
        self.base_eta = initial_lr                 # Base STDP learning rate (eta)
        self.improvement_threshold = improvement_threshold # Min loss decrease to be considered 'improving'
        self.update_interval = update_interval           # How many epochs to look back for trend

        # Set initial base learning rate in all units
        for unit in self.amn.units:
            unit.stdp_learning_rate = self.base_eta # unit.stdp_learning_rate now stores base eta
        logger.info(f"SIE initialized with base STDP learning rate (eta) {self.base_eta:.4f}")

    def update(self, current_loss):
        """Updates STDP learning rate if loss is not decreasing sufficiently."""
        self.performance_history.append(current_loss)

        if len(self.performance_history) > self.update_interval:
            # Calculate trend = loss_now - loss_past
            trend = self.performance_history[-1] - self.performance_history[-1 - self.update_interval]

            # If loss increased or didn't decrease enough (trend >= -threshold)
            if trend >= -self.improvement_threshold:
                # Increase base learning rate slightly (capped at 0.05)
                self.base_eta = min(0.05, self.base_eta + 0.02) # Capped at 0.05
                # Apply updated base rate to all units
                for unit in self.amn.units:
                    unit.stdp_learning_rate = self.base_eta # Update the base eta stored in unit
                logger.info(f"SIE: Increased base STDP learning rate (eta) to {self.base_eta:.4f} (Trend={trend:.4f})")
            else:
                # Log if improvement is good
                logger.info(f"SIE: Base STDP learning rate (eta) stable at {self.base_eta:.4f} (Good Trend={trend:.4f})")

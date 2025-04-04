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
        self.stdp_learning_rate = 0.03 # Default, can be overwritten

    def forward(self, spikes_in):
        """Processes input spike train over `timesteps` using LIF dynamics."""
        logger.debug(f"ModularUnit FWD - Input shape: {spikes_in.shape}, Device: {spikes_in.device}")
        spikes_in = ensure_tensor(spikes_in, self.device)

        # Norse LIFRecurrent expects [batch_size, features] at each timestep
        # We simulate timestep by timestep, assuming batch_size=1 implicitly
        state = None  # Initial hidden state for the recurrent layer
        spikes_out_list = [] # Collect output spikes over time

        for t in range(self.timesteps): # Use self.timesteps
            # Input for this timestep should be [neurons]
            current_spikes_t = spikes_in[t]

            # Reshape to [1, neurons] for Norse LIFRecurrent layer
            current_spikes_batch = current_spikes_t.unsqueeze(0)
            logger.debug(f"  T={t}, Input shape to LIF: {current_spikes_batch.shape}")

            try:
                # Process one timestep through the LIF layer
                spike, state = self.lif(current_spikes_batch, state) # State is managed internally by Norse layer
                logger.debug(f"  T={t}, Output spike shape from LIF: {spike.shape}")

                # Remove batch dimension -> [neurons]
                spike_t = spike.squeeze(0)
                spikes_out_list.append(spike_t)

            except RuntimeError as e:
                logger.error(f"RuntimeError in LIF forward at timestep {t}: {str(e)}", exc_info=True)
                logger.error(f"  Input shape was: {current_spikes_batch.shape}")
                raise

        # Stack the recorded spikes along the time dimension -> [timesteps, neurons]
        output_spikes = torch.stack(spikes_out_list)
        logger.debug(f"ModularUnit FWD - Output shape: {output_spikes.shape}")
        return output_spikes

    def apply_stdp(self, pre_spikes, post_spikes):
        """Applies a simplified STDP rule based on concurrent firing."""
        # Note: This is NOT standard time-difference STDP.
        # It approximates Hebbian learning ("fire together, wire together")
        # and anti-Hebbian learning based on *concurrent* activity within the same timestep.
        logger.debug(f"Applying STDP - Pre shape: {pre_spikes.shape}, Post shape: {post_spikes.shape}")

        # Ensure tensors are on the correct device (secondary GPU for units)
        pre_spikes = pre_spikes.to(self.device)
        post_spikes = post_spikes.to(self.device)

        dw = torch.zeros_like(self.weights) # Initialize weight change tensor

        for t in range(self.timesteps): # Use self.timesteps
            pre_t = ensure_tensor(pre_spikes[t], self.device).squeeze() # Ensure [neurons]
            post_t = ensure_tensor(post_spikes[t], self.device).squeeze() # Ensure [neurons]

            # Check if tensors are valid 1D tensors after ensuring
            if pre_t.dim() != 1 or post_t.dim() != 1:
                 logger.error(f"STDP Error T={t}: Invalid dimensions after ensure/squeeze. Pre: {pre_t.shape}, Post: {post_t.shape}")
                 continue # Skip this timestep if shapes are wrong

            try:
                # Hebbian term: Increase weight if pre and post spike concurrently
                dw += self.stdp_learning_rate * torch.outer(pre_t, post_t)
                # Anti-Hebbian term: Decrease weight if post spikes while pre doesn't (simplified)
                # Note: Original had torch.outer(post_t, pre_t) which is mathematically the transpose of Hebbian term.
                # Using same term `outer(pre_t, post_t)` but subtracting might be intended,
                # or maybe `outer(post_t, pre_t)` implies LTD if post fires before pre in *next* step?
                # Sticking to original code's implementation:
                dw -= self.stdp_learning_rate * 0.12 * torch.outer(post_t, pre_t) # Asymmetry factor 0.12
            except RuntimeError as e:
                logger.error(f"Error in STDP torch.outer at timestep {t}: {e}", exc_info=True)
                logger.error(f"  Pre shape: {pre_t.shape}, Post shape: {post_t.shape}")
                raise

        # Apply accumulated weight changes
        self.weights += dw
        # Keep weights within a reasonable range (e.g., preventing negative or excessively large weights)
        self.weights.clamp_(0.0, 1.0)
        logger.debug("Applied STDP and clamped weights.")


# 2. Global Coordinator: Determines inter-unit connections using a standard NN
class CoordinatorPolicyNetwork(nn.Module):
    """A standard MLP that outputs connection probabilities between units."""
    def __init__(self, num_units, neurons_per_unit, device):
        super(CoordinatorPolicyNetwork, self).__init__()
        self.device = device
        self.num_units = num_units
        self.neurons_per_unit = neurons_per_unit
        input_size = num_units * neurons_per_unit # Flattened activity across all units
        hidden_size = 128
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
        activity_flat = activity.view(original_batch_size, -1)
        logger.debug(f"  Activity reshaped to: {activity_flat.shape}")

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
        self.learning_rate = initial_lr          # Initial STDP learning rate
        self.improvement_threshold = improvement_threshold # Min loss decrease to be considered 'improving'
        self.update_interval = update_interval           # How many epochs to look back for trend

        # Set initial learning rate in all units
        for unit in self.amn.units:
            unit.stdp_learning_rate = self.learning_rate
        logger.info(f"SIE initialized with STDP learning rate {self.learning_rate:.3f}")

    def update(self, current_loss):
        """Updates STDP learning rate if loss is not decreasing sufficiently."""
        self.performance_history.append(current_loss)

        if len(self.performance_history) > self.update_interval:
            # Calculate trend = loss_now - loss_past
            trend = self.performance_history[-1] - self.performance_history[-1 - self.update_interval]

            # If loss increased or didn't decrease enough (trend >= -threshold)
            if trend >= -self.improvement_threshold:
                # Increase learning rate slightly (capped at 0.1)
                self.learning_rate = min(0.1, self.learning_rate + 0.02)
                # Apply updated rate to all units
                for unit in self.amn.units:
                    unit.stdp_learning_rate = self.learning_rate
                logger.info(f"SIE: Increased STDP learning rate to {self.learning_rate:.3f} (Trend={trend:.4f})")
            else:
                # Log if improvement is good
                logger.info(f"SIE: STDP learning rate stable at {self.learning_rate:.3f} (Good Trend={trend:.4f})")

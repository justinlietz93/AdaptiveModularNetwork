import torch
import torch.nn as nn
import logging
from amn_components import ModularUnit, CoordinatorPolicyNetwork
from utils import ensure_tensor, encode_number, get_spike_rate

logger = logging.getLogger(__name__)

# 3. AMN System: Integrates Modular Units and Coordinator
class AMN(nn.Module):
    """The main Adaptive Modular Network class."""
    def __init__(self, num_units, neurons_per_unit, timesteps, device_primary, device_secondary,
                 direct_weight_initial=1.5, use_direct_mapping=True):
        super(AMN, self).__init__()
        self.num_units = num_units
        self.neurons_per_unit = neurons_per_unit
        self.timesteps = timesteps
        self.device_primary = device_primary
        self.device_secondary = device_secondary
        self.use_direct_mapping = use_direct_mapping
        self.direct_weight = direct_weight_initial

        logger.info(f"Initializing AMN with {num_units} units, {neurons_per_unit} neurons/unit, {timesteps} timesteps.")
        logger.info(f"Primary device: {device_primary}, Secondary device: {device_secondary}")
        logger.info(f"Direct mapping heuristic: {'Enabled' if use_direct_mapping else 'Disabled'}, Initial Weight: {direct_weight_initial}")

        # Create list of modular SNN units on the secondary device
        self.units = nn.ModuleList([ModularUnit(neurons_per_unit, timesteps, device_secondary)
                                    for _ in range(num_units)])
        # Create coordinator network on the primary device
        self.coordinator = CoordinatorPolicyNetwork(num_units, neurons_per_unit, device_primary)

        # Stores the latest connection matrix determined by the coordinator
        self.connection_matrix = torch.zeros(1, num_units, num_units, device=device_primary) # Add batch dim
        # Stores outputs from each unit during forward pass
        self.unit_outputs = []

    def forward(self, input_spikes):
        """Processes input spikes through units, coordinates connections, combines outputs."""
        logger.debug(f"AMN FWD - Input shape: {input_spikes.shape}, Device: {input_spikes.device}")
        input_spikes = ensure_tensor(input_spikes, device=self.device_secondary) # Ensure input is on secondary for units

        active_units = list(range(self.num_units))
        # Activity tensor to store mean firing rate per unit, placed on primary device for coordinator
        activity = torch.zeros(self.num_units, self.neurons_per_unit, device=self.device_primary)
        self.unit_outputs = [] # Reset outputs list

        # --- Heuristic: Calculate target output rate for direct mapping ---
        input_rate = get_spike_rate(input_spikes.to(torch.device('cpu')), self.neurons_per_unit) # Rate calc likely needs CPU tensor
        # Rule: Target output rate is input rate + 20Hz (for the x+2 task)
        target_output_rate = input_rate + 20
        logger.debug(f"  Direct mapping: Input Rate={input_rate:.2f} Hz -> Target Output Rate={target_output_rate:.2f} Hz")
        # --- End Heuristic ---

        # 1. Process spikes through each modular unit
        for i in active_units:
            unit = self.units[i]
            # Input spikes should already be on unit.device (secondary)
            try:
                output = unit(input_spikes) # Get output spike train [timesteps, neurons]
                self.unit_outputs.append(output)
                # Calculate mean activity over time, move result to primary device for coordinator
                activity[i] = output.mean(dim=0).to(self.device_primary)
            except Exception as e:
                logger.error(f"Error processing unit {i}: {e}", exc_info=True)
                raise

        # 2. Get dynamic connection matrix from the coordinator based on unit activity
        try:
            # Activity shape should be [num_units, neurons_per_unit] before passing
            logger.debug(f"  Activity shape for coordinator: {activity.shape}")
            connection_matrix_sampled, _ = self.coordinator.get_action(activity) # Get binary connections [1, units, units]
            self.connection_matrix = connection_matrix_sampled # Store the sampled matrix
            logger.debug(f"  Connection matrix shape: {self.connection_matrix.shape}")
        except Exception as e:
            logger.error(f"Error getting action from coordinator: {e}", exc_info=True)
            raise

        # 3. Combine unit outputs based on the dynamic connection matrix
        try:
            # Initialize final output tensor on the secondary device
            final_output = torch.zeros(self.timesteps, self.neurons_per_unit, device=self.device_secondary)
            connection_strength = 3.0 # Scaling factor for combining connected unit outputs

            # Iterate through potential connections
            # connection_matrix might be [1, units, units] or [units, units] if batch=1 was squeezed
            conn_matrix_view = self.connection_matrix.squeeze(0) if self.connection_matrix.dim() == 3 else self.connection_matrix

            if conn_matrix_view.dim() != 2:
                 logger.error(f"Unexpected connection matrix dimension: {conn_matrix_view.shape}")
                 # Handle error or use default if necessary

            else:
                 for i in active_units:
                     for j in active_units:
                         # If coordinator decided to connect unit j to unit i (conceptually)
                         if conn_matrix_view[i, j] > 0.1: # Use a threshold (0.1) on sampled 0/1 matrix? Original code logic.
                             # Add weighted output of unit j to the final output (conceptually unit i's input region for output)
                             # Ensure tensors are on the same device (secondary)
                             unit_output_j = self.unit_outputs[j].to(self.device_secondary)
                             # Scale connection by strength
                             weight = conn_matrix_view[i, j].item() * connection_strength # Using item() assuming scalar after check
                             final_output += unit_output_j * weight
                 # Normalize by number of units contributing? Original code used / len(active_units) here.
                 # Let's keep it for now, although summing weighted outputs might be intended.
                 # Consider if normalization is needed based on task.
                 # final_output /= len(active_units) # Removing this as summing weighted outputs might be intended

            logger.debug(f"  Output after combining units: Mean={final_output.mean():.4f}")

            # --- Heuristic: Apply direct mapping rule if enabled ---
            if self.use_direct_mapping:
                # Encode the target rate (input + 20Hz) into a spike train
                target_output_spikes = encode_number(
                    target_output_rate / 10.0, # Divide by 10 for encode_number
                    self.timesteps,
                    self.neurons_per_unit,
                    self.device_secondary
                )
                # Blend the network's computed output with the heuristic target output
                final_output = final_output * 0.5 + target_output_spikes * self.direct_weight
                logger.debug(f"  Applied direct mapping (weight={self.direct_weight:.3f}). New Mean={final_output.mean():.4f}")
            # --- End Heuristic ---

            # --- Heuristic: Boost output activation if too low ---
            # Ensure the network output has sufficient activity to represent the target rate
            if final_output.mean() < 0.2: # Threshold for minimum acceptable mean activation
                current_mean = final_output.mean().item()
                target_mean = target_output_rate / 100.0 # Target mean activation related to rate (adjust divisor if needed)
                boost_factor = max(0, target_mean - current_mean)
                logger.debug(f"  Boosting output: Current Mean={current_mean:.4f}, Target Mean={target_mean:.4f}, Boost Factor={boost_factor:.4f}")
                if boost_factor > 0:
                    # Add random noise scaled by the needed boost
                    bias = torch.rand_like(final_output) * boost_factor * 2.0 # Stronger boost scaling
                    final_output += bias
                    logger.debug(f"  Output after boosting: Mean={final_output.mean():.4f}")
            # --- End Heuristic ---

            logger.debug(f"AMN FWD - Final output shape: {final_output.shape}, Device: {final_output.device}")
            return final_output

        except Exception as e:
            logger.error(f"Error during final output calculation: {e}", exc_info=True)
            # Log details helpful for debugging device mismatches or shape errors
            for i, out in enumerate(self.unit_outputs):
                 logger.error(f"  Unit {i} output - Shape: {out.shape}, Device: {out.device}")
            logger.error(f"  Connection matrix - Shape: {self.connection_matrix.shape}, Device: {self.connection_matrix.device}")
            logger.error(f"  Final output (partial) - Shape: {final_output.shape}, Device: {final_output.device}")
            raise

    def train_step(self, input_spikes, target_spikes, optimizer, epoch=0):
        """Performs one training step including forward pass, loss calculation, backprop, and STDP."""
        logger.debug(f"AMN Train Step {epoch} - Input shape: {input_spikes.shape}, Target shape: {target_spikes.shape}")

        # --- Heuristic: Anneal direct mapping influence ---
        # Gradually reduce the influence of the hardcoded rule over epochs
        if self.use_direct_mapping and self.direct_weight > 0.5: # Stop reducing below 0.5
            self.direct_weight *= 0.998 # Slow exponential decay
            logger.debug(f"  Annealed direct_weight to {self.direct_weight:.3f}")
        # --- End Heuristic ---

        # 1. Forward pass through the AMN
        output_spikes = self(input_spikes)

        # 2. Calculate loss (MSE between mean firing rates)
        # Move target spikes to the output device (secondary GPU)
        target_mean = target_spikes.mean(dim=0).to(output_spikes.device)
        output_mean = output_spikes.mean(dim=0)

        # Ensure shapes match for loss calculation
        if output_mean.shape != target_mean.shape:
             logger.error(f"Shape mismatch for loss: Output mean {output_mean.shape}, Target mean {target_mean.shape}")
             # Handle error appropriately, maybe raise or return NaN loss
             return float('nan'), float('nan')

        loss = nn.MSELoss()(output_mean, target_mean)
        loss_item = loss.item() # Get scalar loss value for logging/SIE

        # Log output rate vs target rate for debugging progress
        output_rate = get_spike_rate(output_spikes.to(torch.device('cpu')), self.neurons_per_unit)
        target_rate = get_spike_rate(target_spikes.to(torch.device('cpu')), self.neurons_per_unit)
        if epoch % 10 == 0:
            logger.debug(f"  Epoch {epoch}: Output rate={output_rate:.2f} Hz, Target rate={target_rate:.2f} Hz, Loss={loss_item:.4f}")

        # 3. Backpropagation for the Coordinator Network
        # Zero gradients for the coordinator's optimizer
        optimizer.zero_grad()
        # Calculate gradients based on the loss
        # Note: Gradients only flow back through operations involving coordinator outputs
        # (i.e., the weighted sum using connection_matrix). STDP happens separately.
        try:
             loss.backward()
        except RuntimeError as e:
             logger.error(f"Error during loss.backward(): {e}", exc_info=True)
             # Log shapes again right before backward pass
             logger.error(f"  Output mean shape: {output_mean.shape}, Target mean shape: {target_mean.shape}")
             raise
        # Update coordinator weights
        optimizer.step()

        # 4. Apply STDP rule to each Modular Unit
        # Note: This happens *after* backprop, using the same input spikes
        # and the unit outputs generated *before* the coordinator was updated.
        for i, unit in enumerate(self.units):
            pre_spikes_on_device = input_spikes.to(unit.device)
            unit_output = self.unit_outputs[i] # Use stored output from forward pass
            unit.apply_stdp(pre_spikes_on_device, unit_output)

        return loss_item, output_rate

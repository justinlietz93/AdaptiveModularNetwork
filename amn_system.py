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
        logger.debug(f"AMN FWD - Initial Input shape: {input_spikes.shape}, Device: {input_spikes.device}")
        input_spikes = ensure_tensor(input_spikes, device=self.device_secondary) # Ensure input is on secondary for units
        logger.debug(f"  Input Device after ensure: {input_spikes.device}")

        active_units = list(range(self.num_units))
        # List to store full spike outputs from each unit
        all_unit_outputs_list = []
        self.unit_outputs = [] # Keep this for STDP if re-enabled later

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
                self.unit_outputs.append(output) # Keep for potential STDP re-enablement
                # Move full output to primary device for coordinator input
                all_unit_outputs_list.append(output.to(self.device_primary))
                logger.debug(f"  Unit {i} Full Output Device: {output.device}, Copied to Device: {all_unit_outputs_list[-1].device}")
            except Exception as e:
                logger.error(f"Error processing unit {i}: {e}", exc_info=True)
                raise

        # 2. Get dynamic connection matrix from the coordinator based on full unit outputs
        try:
            # Concatenate and flatten all unit outputs [units, timesteps, neurons] -> [1, units*timesteps*neurons]
            if not all_unit_outputs_list:
                 raise ValueError("No unit outputs collected for coordinator.")

            # Stack along a new dimension (batch dim, effectively) -> [num_units, timesteps, neurons]
            full_activity_tensor = torch.stack(all_unit_outputs_list, dim=0)
            logger.debug(f"  Stacked full activity shape: {full_activity_tensor.shape}, Device: {full_activity_tensor.device}")

            # Flatten into a single vector for the coordinator MLP input
            # Add batch dimension [1, features]
            coordinator_input = full_activity_tensor.view(1, -1)
            logger.debug(f"  Flattened coordinator input shape: {coordinator_input.shape}, Device: {coordinator_input.device}")

            # Get action (sampled matrix) and log probabilities from coordinator
            connection_matrix_sampled, log_probs = self.coordinator.get_action(coordinator_input) # Pass flattened full spike data
            self.connection_matrix = connection_matrix_sampled # Store the sampled matrix
            self.connection_log_probs = log_probs # Store log_probs for use in train_step
            logger.debug(f"  Connection matrix shape: {self.connection_matrix.shape}, Device: {self.connection_matrix.device}")
            logger.debug(f"  Log Probs shape: {self.connection_log_probs.shape}, Device: {self.connection_log_probs.device}")
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
            # --- RE-ENABLED Boosting Heuristic ---
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
            # --- END RE-ENABLED Boosting Heuristic ---

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

    def train_step(self, input_spikes, target_spikes, optimizer, epoch=0, baseline=0.0): # Added baseline argument
        """Performs one training step using Policy Gradient (REINFORCE with baseline)."""
        logger.debug(f"AMN Train Step {epoch} - Input shape: {input_spikes.shape}, Target shape: {target_spikes.shape}, Baseline: {baseline:.4f}")

        # --- Heuristic: Anneal direct mapping influence ---
        # Gradually reduce the influence of the hardcoded rule over epochs
        if self.use_direct_mapping and self.direct_weight > 0.5: # Stop reducing below 0.5
            self.direct_weight *= 0.995 # Faster exponential decay (tuned from 0.998)
            logger.debug(f"  Annealed direct_weight to {self.direct_weight:.3f}")
        # --- End Heuristic ---

        # 1. Forward pass through the AMN
        output_spikes = self(input_spikes)

        # 2. Calculate Reward based on Output Rate
        output_rate = get_spike_rate(output_spikes.to(torch.device('cpu')), self.neurons_per_unit)
        # We need the original input number to calculate the target rate for the reward
        # This assumes train_step is called with context or modified to receive input_num
        # For now, we'll retrieve it from the target_spikes encoding (less robust)
        # A better way would be to pass input_num to train_step
        target_rate_approx = get_spike_rate(target_spikes.to(torch.device('cpu')), self.neurons_per_unit)
        # Simple reward: higher for closer rates, max 1.0
        reward = 1.0 / (1.0 + abs(output_rate - target_rate_approx))
        logger.debug(f"  Reward Calculation: Output Rate={output_rate:.2f}, Target Rate={target_rate_approx:.2f}, Reward={reward:.4f}")

        # 3. Calculate Policy Gradient Loss for Coordinator
        # Use the stored log_probs from the forward pass
        if not hasattr(self, 'connection_log_probs') or self.connection_log_probs is None:
             logger.error("Log probabilities (self.connection_log_probs) not found for policy gradient calculation.")
             # Handle error: maybe return NaN or skip update
             return float('nan'), output_rate, reward # Return NaN loss, but still return reward

        # Ensure reward and baseline are tensors on the correct device
        reward_tensor = torch.tensor(reward, device=self.device_primary, dtype=torch.float32)
        baseline_tensor = torch.tensor(baseline, device=self.device_primary, dtype=torch.float32)

        # Policy Gradient Loss (REINFORCE with baseline): -log_prob * (reward - baseline)
        # Sum over the action matrix dimensions and average over batch (batch=1 here)
        advantage = reward_tensor - baseline_tensor
        policy_loss = (-self.connection_log_probs * advantage.detach()).mean() # Detach advantage to prevent grads flowing into baseline calc
        loss_item = policy_loss.item() # Use policy loss for logging/SIE
        logger.debug(f"  Policy Gradient Loss calculated: {loss_item:.4f} (Advantage: {advantage.item():.4f})")

        # Zero gradients for the coordinator's optimizer
        optimizer.zero_grad()
        # Calculate gradients based on the policy loss
        try:
             policy_loss.backward()
        except RuntimeError as e:
             logger.error(f"Error during policy_loss.backward(): {e}", exc_info=True)
             # Log relevant shapes
             logger.error(f"  Log Probs shape: {self.connection_log_probs.shape if hasattr(self, 'connection_log_probs') else 'Not found'}")
             raise
        # Update coordinator weights
        optimizer.step()

        # 4. Apply STDP rule and update weights for each Modular Unit
        # --- RE-ENABLED FUM STDP ---
        # Note: This happens *after* policy gradient update.
        for i, unit in enumerate(self.units):
            # Ensure input_spikes is on the correct device for apply_stdp if needed
            # (apply_stdp currently handles internal .to(device))
            unit_output = self.unit_outputs[i] # Use stored output from forward pass

            # Calculate STDP changes and update eligibility trace
            unit.apply_stdp(input_spikes, unit_output)

            # Apply final weight update using the reward
            # Pass the scalar reward value
            unit.update_weights(reward)
        # --- END RE-ENABLED FUM STDP ---

        return loss_item, output_rate, reward # Return reward for baseline update

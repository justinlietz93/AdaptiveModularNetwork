import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def ensure_tensor(x, device=None):
    """
    Ensures the input is a PyTorch tensor with at least 1 dimension.
    If input is a scalar, converts it and adds a batch dimension.

    Args:
        x: Input data (scalar, list, numpy array, or tensor).
        device (torch.device, optional): Target device. Defaults to None.

    Returns:
        torch.Tensor: Tensor representation of x on the specified device.
    """
    if not isinstance(x, torch.Tensor):
        # Convert numpy arrays or lists/scalars to tensor
        x = torch.tensor(x, device=device, dtype=torch.float32) # Added dtype

    # Ensure at least 1 dimension (e.g., for single neuron values)
    if x.dim() == 0:
        x = x.unsqueeze(0)

    # Ensure correct device placement if device is specified
    if device and x.device != device:
         x = x.to(device)

    return x

def encode_number(n, timesteps, neurons_per_unit, device):
    """
    Encodes a number n into a simple rate-coded spike train.
    Target rate is n * 10 Hz, capped at 50 Hz.
    Spikes are generated probabilistically based on this rate.

    Args:
        n (float): Number to encode.
        timesteps (int): Duration of the simulation step.
        neurons_per_unit (int): Number of neurons in the unit.
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Spike tensor [timesteps, neurons_per_unit] on the specified device.
    """
    spikes = torch.zeros(timesteps, neurons_per_unit, device=device)
    # Map number n to a firing rate (Hz), e.g., n=3 -> 30Hz
    target_rate = min(n * 10.0, 50.0) # Cap rate at 50Hz

    # Generate Poisson-like spikes for each timestep
    # Probability of spiking = target_rate / 1000 (assuming 1ms timestep implicitly)
    # The original code uses rate / 100, maybe dt is assumed differently or just scaling factor?
    # Sticking to original code's scaling for now: prob = rate / 100
    spike_prob = target_rate / 100.0

    for t in range(timesteps):
        spikes[t] = (torch.rand(neurons_per_unit, device=device) < spike_prob).float()

    logger.debug(f"Encoded number {n} -> Rate {target_rate:.1f} Hz -> Spike Prob {spike_prob:.3f}")
    logger.debug(f"  Output spikes shape: {spikes.shape}, Device: {spikes.device}, Mean spikes: {spikes.mean():.3f}")
    return spikes

def get_spike_rate(spikes, neurons_per_unit):
    """
    Calculates the average firing rate in Hz from a spike tensor.

    Args:
        spikes (torch.Tensor): The spike tensor [timesteps, neurons].
        neurons_per_unit (int): Number of neurons used for normalization.

    Returns:
        float: The calculated average firing rate.
    """
    if not isinstance(spikes, torch.Tensor) or spikes.numel() == 0:
        return 0.0
    # Mean spikes per neuron per timestep
    mean_spike_prob = spikes.mean().item()
    # Convert to Hz (assuming 1ms timesteps implicitly -> 1000 timesteps/sec)
    # Original code: mean(dim=0).sum() / neurons_per_unit * 100. Let's analyze:
    # spikes.mean(dim=0) -> shape [neurons], avg spike prob per neuron over time
    # .sum() -> sum of avg spike probs across all neurons
    # / neurons_per_unit -> overall avg spike prob per neuron
    # * 100 -> This scaling seems arbitrary, maybe intended to map back to the input 'n' value?
    # If target rate was n*10, then output rate should be measured consistently.
    # Let's use mean spike probability * 1000 (for Hz assuming 1ms steps)
    # Reverting to original code's calculation method for consistency for now:
    calculated_rate = spikes.mean(dim=0).sum().item() / neurons_per_unit * 100.0
    # Alternative (potentially more standard Hz calc assuming 1ms steps):
    # calculated_rate = mean_spike_prob * 1000.0 # Rate in Hz
    return calculated_rate

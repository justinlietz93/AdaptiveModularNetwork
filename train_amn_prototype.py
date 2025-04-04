import torch
import torch.optim as optim
import logging
import time
import numpy as np # For calculating average epoch time
import matplotlib.pyplot as plt # For plotting
import os # For creating directories

# Import components from the new modules
from amn_system import AMN
from amn_components import SelfImprovementEngine
from utils import encode_number, get_spike_rate
from evaluation import run_tests, evaluate_results

# --- Logging Setup ---
# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for best practice

# Set desired logging level (DEBUG for detailed, INFO for normal)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG) # Enable detailed debugging output

# --- Configuration ---
# Device assignment for hybrid GPU setup
# Check if CUDA is available before assigning devices
if torch.cuda.is_available():
    if torch.cuda.device_count() >= 2:
        device_primary = torch.device("cuda:0")   # MI100 (intended for Coordinator, PyTorch ops)
        device_secondary = torch.device("cuda:1") # 7900 XTX (intended for SNN units, Norse ops)
        logger.info("Using cuda:0 (Primary) and cuda:1 (Secondary)")
    elif torch.cuda.device_count() == 1:
        device_primary = torch.device("cuda:0")
        device_secondary = torch.device("cuda:0")
        logger.info("Only one GPU found. Using cuda:0 for both Primary and Secondary.")
    else: # Should not happen if torch.cuda.is_available() is true, but for safety
        device_primary = torch.device("cpu")
        device_secondary = torch.device("cpu")
        logger.warning("CUDA available but no devices found? Using CPU.")
else:
    device_primary = torch.device("cpu")
    device_secondary = torch.device("cpu")
    logger.info("CUDA not available. Using CPU for both Primary and Secondary.")


# Network parameters
num_units = 10
neurons_per_unit = 100
timesteps = 50 # Duration of each simulation step in ms (typical)

# Training parameters
max_epochs = 200 # Increased from 100 for potentially more complex learning
learning_rate_adam = 0.0005 # Adam LR for Coordinator (Trying lower LR again with new loss)
initial_stdp_lr = 0.03 # Initial STDP LR for SIE
use_direct_mapping_heuristic = False # Disabled heuristic
initial_direct_weight = 1.3 # Tuned from 1.4 (Value doesn't matter when disabled)

# Evaluation parameters
test_cases = [(1, 3), (4, 6)] # Input number -> Expected output number (x+2 task)
target_output_rate_for_train_task = 50.0 # Target rate for the original single training example (input 3)

# --- Training Data ---
# Expand training data beyond a single example to improve generalization
# Input number -> Target output number (x+2 task)
training_pairs = [(i, i + 2) for i in range(10)] # e.g., (0, 2), (1, 3), ..., (9, 11)
logger.info(f"Using training pairs: {training_pairs}")

# --- Output Directories ---
models_dir = "models"
benchmarks_dir = "benchmarks"

# --- Plotting Function ---
def plot_metrics(losses, output_rates, stdp_lrs, save_dir=".", filename="amn_training_metrics.png"):
    """Generates and saves plots for training metrics."""
    epochs = range(len(losses))
    if not epochs:
        logger.warning("No data to plot.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('AMN Training Metrics')

    # Plot Loss
    axs[0].plot(epochs, losses, label='Loss', color='tab:red')
    axs[0].set_ylabel('MSE Loss')
    axs[0].grid(True)
    axs[0].legend()

    # Plot Output Rate
    axs[1].plot(epochs, output_rates, label='Output Rate (Hz)', color='tab:blue')
    axs[1].axhline(y=target_output_rate_for_train_task, color='grey', linestyle='--', label=f'Target Rate ({target_output_rate_for_train_task} Hz)')
    axs[1].set_ylabel('Output Rate (Hz)')
    axs[1].grid(True)
    axs[1].legend()

    # Plot STDP Learning Rate
    axs[2].plot(epochs, stdp_lrs, label='STDP Learning Rate', color='tab:green')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('STDP LR')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    save_path = os.path.join(save_dir, filename)
    try:
        os.makedirs(save_dir, exist_ok=True) # Ensure directory exists
        plt.savefig(save_path)
        logger.info(f"Training metrics plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {save_path}: {e}", exc_info=True)
    plt.close(fig) # Close the figure to free memory

# --- Main Training and Testing Logic ---
def main():
    """Main function to initialize, train, test, and evaluate the AMN."""
    start_time = time.time()
    final_runtime = 0
    final_loss = float('inf')
    final_output_rate = 0
    test_results = []
    success_status = False # Default to False

    # --- Benchmarking Variables ---
    epoch_times = []
    losses = []
    output_rates = []
    stdp_lrs = []
    final_inference_time = 0
    test_run_time = 0
    # --- End Benchmarking Variables ---

    try:
        # --- Create Output Directories ---
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(benchmarks_dir, exist_ok=True)
        # --- End Create Output Directories ---

        init_start_time = time.time()
        logger.info("Initializing AMN, Optimizer, and SIE...")
        # Instantiate AMN with configuration
        amn_model = AMN(
            num_units=num_units,
            neurons_per_unit=neurons_per_unit,
            timesteps=timesteps,
            device_primary=device_primary,
            device_secondary=device_secondary,
            use_direct_mapping=use_direct_mapping_heuristic,
            direct_weight_initial=initial_direct_weight
        )
        # Optimizer only targets the Coordinator Network parameters
        optimizer = optim.Adam(amn_model.coordinator.parameters(), lr=learning_rate_adam)
        # Instantiate SIE with the model and initial LR
        sie = SelfImprovementEngine(amn_model, initial_lr=initial_stdp_lr)
        init_end_time = time.time()
        logger.info(f"Initialization complete in {init_end_time - init_start_time:.2f} seconds.")

        # Pre-encode all training examples
        logger.info("Pre-encoding training examples...")
        encoded_training_data = []
        for input_num, target_num in training_pairs:
            input_spikes = encode_number(input_num, timesteps, neurons_per_unit, device_secondary)
            target_spikes = encode_number(target_num, timesteps, neurons_per_unit, device_secondary)
            encoded_training_data.append({'input': input_spikes, 'target': target_spikes, 'in_val': input_num, 'tgt_val': target_num})
        logger.info(f"Encoded {len(encoded_training_data)} training examples.")

        # Variables to track the best performing model state during training
        best_loss_tracker = float('inf')
        best_output_rate_tracker = 0
        best_model_state = None
        baseline_reward_ema = 0.0 # Exponential moving average for baseline
        baseline_alpha = 0.1 # Smoothing factor for EMA

        # Training loop
        logger.info(f"Starting training for {max_epochs} epochs...")
        train_start_time = time.time()
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            epoch_losses = []
            epoch_output_rates = []
            amn_model.train() # Ensure model is in training mode

            # Iterate through encoded training data
            # Consider shuffling data each epoch: np.random.shuffle(encoded_training_data)
            for train_data in encoded_training_data:
                input_spikes_train = train_data['input']
                target_spikes_train = train_data['target']

                try:
                    # Perform one training step using the AMN method, passing baseline
                    loss, output_rate, reward = amn_model.train_step( # Expect train_step to return reward now
                        input_spikes_train, target_spikes_train, optimizer, epoch, baseline=baseline_reward_ema
                    )

                    # Update baseline EMA
                    baseline_reward_ema = (1 - baseline_alpha) * baseline_reward_ema + baseline_alpha * reward

                    # Handle potential NaN loss from train_step
                    if torch.isnan(torch.tensor(loss)):
                        logger.error(f"Epoch {epoch}, Input {train_data['in_val']}: Encountered NaN loss. Skipping step.")
                        continue # Skip this training example if loss is NaN

                    epoch_losses.append(loss)
                    epoch_output_rates.append(output_rate)

                except RuntimeError as e:
                    logger.error(f"Error during training epoch {epoch}, Input {train_data['in_val']}: {str(e)}", exc_info=True)
                    # Log tensor shapes to help debug
                    logger.error(f"  Input spikes shape: {input_spikes_train.shape}")
                    logger.error(f"  Target spikes shape: {target_spikes_train.shape}")
                    raise # Re-raise after logging details

            # --- Calculate Average Metrics for Epoch ---
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            avg_epoch_output_rate = np.mean(epoch_output_rates) if epoch_output_rates else float('nan')

            if torch.isnan(torch.tensor(avg_epoch_loss)):
                 logger.error(f"Epoch {epoch}: Average loss is NaN. Stopping training.")
                 break

            final_loss = avg_epoch_loss # Track last valid average loss
            final_output_rate = avg_epoch_output_rate # Track last valid average output rate

            # --- Store Average Metrics ---
            losses.append(avg_epoch_loss)
            output_rates.append(avg_epoch_output_rate) # Store average rate for plotting
            stdp_lrs.append(sie.base_eta) # Store current base_eta *after* potential update
            # --- End Store Average Metrics ---

            # Update the Self-Improvement Engine based on average epoch loss
            # Note: SIE might need adjustment if policy loss scale differs significantly from MSE
            sie.update(avg_epoch_loss) # Update happens after storing metrics for the epoch

            # Log progress periodically
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)
            if epoch % 10 == 0 or epoch == max_epochs - 1:
                logger.info(f"Epoch {epoch}/{max_epochs}, Avg Loss: {avg_epoch_loss:.4f}, Avg Output Rate: {avg_epoch_output_rate:.2f} Hz, STDP LR: {sie.base_eta:.4f}, Time: {epoch_time:.2f}s") # Use sie.base_eta

            # Save the best model based on average epoch loss (simpler criterion for multi-example training)
            if avg_epoch_loss < best_loss_tracker:
                 best_loss_tracker = avg_epoch_loss
                 best_output_rate_tracker = avg_epoch_output_rate # Store corresponding rate
                 # --- FIX: Corrected dictionary definition ---
                 best_model_state = {
                     'amn_state_dict': amn_model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'sie_base_eta': sie.base_eta, # Save SIE base_eta state
                     'amn_direct_weight': amn_model.direct_weight, # Save AMN heuristic state
                     'epoch': epoch,
                     'loss': avg_epoch_loss, # Use avg_epoch_loss here
                     'output_rate': avg_epoch_output_rate # Use avg_epoch_output_rate here
                 }
                 # --- End FIX ---
                 logger.info(f"  --> New best model state saved at epoch {epoch} (Avg Loss: {avg_epoch_loss:.4f})")

            # Convergence check based on average loss (adjust threshold if needed)
            if avg_epoch_loss < 0.005: # Stricter loss threshold for convergence with multiple examples?
                 logger.info(f"Average loss below threshold at epoch {epoch}. Stopping training.")
                 break

        # After training, restore the best model state found
        if best_model_state is not None:
            amn_model.load_state_dict(best_model_state['amn_state_dict'])
            # Restore heuristic state if saved
            if 'amn_direct_weight' in best_model_state:
                amn_model.direct_weight = best_model_state['amn_direct_weight']
            # Restore SIE state if saved
            if 'sie_base_eta' in best_model_state:
                sie.base_eta = best_model_state['sie_base_eta']
                for unit in amn_model.units: # Re-apply to units
                    unit.stdp_learning_rate = sie.base_eta # Ensure units have the restored base eta

            # Optionally restore optimizer state if needed for continued training
            # optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
            logger.info(f"Restored best model state from epoch {best_model_state['epoch']} (Output Rate: {best_model_state['output_rate']:.2f} Hz, Loss: {best_model_state['loss']:.4f})")
            # Use the best state's performance for final reporting
            final_loss = best_model_state['loss']
            final_output_rate = best_model_state['output_rate']
        else:
             logger.warning("No best model state was saved during training. Using final state for evaluation.")
             # final_loss and final_output_rate already hold the last values

        # Final inference check with the training input using the loaded model (best or final)
        try:
            amn_model.eval()
            with torch.no_grad():
                 # Re-encode just in case, ensure it's on the correct device
                 # --- FIX: Use a specific input value for final check ---
                 final_check_input_number = 3 # Use the original single-example input for consistency check
                 final_check_target_number = final_check_input_number + 2
                 final_check_target_rate = final_check_target_number * 10.0
                 # --- End FIX ---
                 final_inf_start_time = time.time()
                 input_spikes_final_check = encode_number(final_check_input_number, timesteps, neurons_per_unit, device_secondary)
                 output_final = amn_model(input_spikes_final_check)
            # Use the utility function for rate calculation
            output_rate_final = get_spike_rate(output_final.to(torch.device('cpu')), neurons_per_unit)
            final_inf_end_time = time.time()
            final_inference_time = final_inf_end_time - final_inf_start_time
            # --- FIX: Update log message ---
            logger.info(f"--- Final Inference Check (Loaded Model): Input={final_check_input_number}, Predicted Output Rate={output_rate_final:.2f} Hz (Target={final_check_target_rate} Hz) ---")
            # --- End FIX ---
            logger.info(f"    (Final inference check took {final_inference_time:.3f} seconds)")
            # Update final_output_rate with this potentially more accurate measure from eval mode
            final_output_rate = output_rate_final

        except RuntimeError as e:
            logger.error(f"Error during final inference run: {str(e)}", exc_info=True)
            raise

        # Run generalization tests using the evaluation function
        test_run_start_time = time.time()
        test_results = run_tests(
            amn_model,
            test_cases,
            timesteps,
            neurons_per_unit,
            # Removed duplicate timesteps, neurons_per_unit
            device_secondary
        )
        test_run_end_time = time.time()
        test_run_time = test_run_end_time - test_run_start_time
        logger.info(f"Generalization tests completed in {test_run_time:.2f} seconds.")

    except Exception as e:
        # Catch any unexpected errors during setup or training/testing
        logger.error(f"Unhandled exception in main execution: {str(e)}", exc_info=True)
        # Attempt to preserve some final state info if possible
        final_loss = final_loss if 'final_loss' in locals() and not torch.isnan(torch.tensor(final_loss)) else float('inf')
        final_output_rate = final_output_rate if 'final_output_rate' in locals() else 0
        test_results = test_results if 'test_results' in locals() else []
        success_status = False # Mark as failure if exception occurred
    finally:
        # Calculate runtime regardless of success/failure
        final_runtime = time.time() - start_time
        logger.info(f"Total Runtime: {final_runtime:.0f} seconds")

        # Perform final evaluation only if amn_model exists and no critical error before this point
        if 'amn_model' in locals():
             # Pass collected metrics to evaluation
             success_status = evaluate_results(
                 amn_model=amn_model,
                 final_loss=final_loss,
                 final_output_rate=final_output_rate,
                 target_output_rate_for_train=target_output_rate_for_train_task,
                 test_results_list=test_results,
                 total_runtime=final_runtime,
                 # --- Pass Benchmark Data ---
                 epoch_times=epoch_times,
                 losses=losses,
                 output_rates=output_rates,
                 stdp_lrs=stdp_lrs,
                 final_inference_time=final_inference_time,
                 test_run_time=test_run_time
                 # --- End Pass Benchmark Data ---
             )

        # Save the model state (best if available, otherwise final)
        model_to_save = None
        save_filename = None
        if 'best_model_state' in locals() and best_model_state is not None:
            model_to_save = best_model_state
            save_filename = "amn_model_best.pth"
            logger.info(f"Saving best model state from epoch {best_model_state['epoch']} to {os.path.join(models_dir, save_filename)}")
        elif 'amn_model' in locals(): # Save final state if no best state was logged or error occurred
             model_to_save = {
                 'amn_state_dict': amn_model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
                 'sie_base_eta': sie.base_eta if 'sie' in locals() else None, # Save base_eta
                 'amn_direct_weight': amn_model.direct_weight if 'amn_model' in locals() else None,
                 'loss': final_loss,
                 'output_rate': final_output_rate,
                 'epoch': epoch if 'epoch' in locals() else -1 # Save last epoch number
             }
             save_filename = "amn_model_final.pth"
             logger.info(f"Saving final model state to {os.path.join(models_dir, save_filename)}")

        if model_to_save and save_filename:
            save_path = os.path.join(models_dir, save_filename)
            try:
                torch.save(model_to_save, save_path)
                logger.info(f"Model state successfully saved to {save_path}")
            except Exception as save_e:
                logger.error(f"Failed to save model state to {save_path}: {save_e}", exc_info=True)

        # --- Generate Plots ---
        # Check if metrics were collected before plotting
        if losses and output_rates and stdp_lrs:
             plot_metrics(losses, output_rates, stdp_lrs, save_dir=benchmarks_dir) # Pass benchmarks_dir
        else:
             logger.warning("Skipping plot generation due to missing metric data (likely caused by early error).")
        # --- End Generate Plots ---


if __name__ == "__main__":
    torch.manual_seed(42) # Ensure reproducibility for random initializations
    np.random.seed(42)    # Ensure reproducibility for numpy operations if any
    try:
        main()
    except Exception as e:
        # Log crash if main() itself fails catastrophically
        logger.critical(f"CRITICAL FAILURE: main() crashed: {str(e)}", exc_info=True)

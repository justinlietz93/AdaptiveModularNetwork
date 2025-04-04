import torch
import torch.optim as optim
import logging
import time

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
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG) # Uncomment for detailed debugging output

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
learning_rate_adam = 0.001 # Adam LR for Coordinator
initial_stdp_lr = 0.03 # Initial STDP LR for SIE
use_direct_mapping_heuristic = True
initial_direct_weight = 1.5

# Evaluation parameters
test_cases = [(1, 3), (4, 6)] # Input number -> Expected output number (x+2 task)
target_output_rate_for_train_task = 50.0 # Target rate for input 3 (3+2=5 -> 5*10=50Hz)

# --- Main Training and Testing Logic ---
def main():
    """Main function to initialize, train, test, and evaluate the AMN."""
    start_time = time.time()
    final_runtime = 0
    final_loss = float('inf')
    final_output_rate = 0
    test_results = []
    success_status = False # Default to False

    try:
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

        # Define the simple training task: Input 3 -> Target Output 5 (representing x+2=5)
        input_number = 3
        target_number = 5
        logger.info(f"Setting up training task: Input={input_number}, Target={target_number}")
        # Encode numbers into spike trains using the utility function
        input_spikes_train = encode_number(input_number, timesteps, neurons_per_unit, device_secondary)
        target_spikes_train = encode_number(target_number, timesteps, neurons_per_unit, device_secondary)

        logger.info(f"Starting training for {max_epochs} epochs...")

        # Variables to track the best performing model state during training
        best_loss_tracker = float('inf')
        best_output_rate_tracker = 0
        best_model_state = None

        # Training loop
        for epoch in range(max_epochs):
            try:
                amn_model.train() # Ensure model is in training mode
                # Perform one training step using the AMN method
                loss, output_rate = amn_model.train_step(
                    input_spikes_train, target_spikes_train, optimizer, epoch
                )

                # Handle potential NaN loss from train_step
                if torch.isnan(torch.tensor(loss)):
                    logger.error(f"Epoch {epoch}: Encountered NaN loss. Stopping training.")
                    break

                final_loss = loss # Keep track of the last valid loss
                final_output_rate = output_rate # Keep track of the last valid output rate

                # Update the Self-Improvement Engine
                sie.update(loss)

                # Log progress periodically
                if epoch % 10 == 0 or epoch == max_epochs - 1:
                    logger.info(f"Epoch {epoch}/{max_epochs}, Loss: {loss:.4f}, Output Rate: {output_rate:.2f} Hz")

                # Save the best model based on proximity to target rate
                current_target_rate = target_output_rate_for_train_task
                if abs(output_rate - current_target_rate) < abs(best_output_rate_tracker - current_target_rate):
                    best_output_rate_tracker = output_rate
                    best_loss_tracker = loss
                    best_model_state = {
                        'amn_state_dict': amn_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'sie_learning_rate': sie.learning_rate, # Save SIE state if needed
                        'amn_direct_weight': amn_model.direct_weight, # Save AMN heuristic state
                        'epoch': epoch,
                        'loss': loss,
                        'output_rate': output_rate
                    }
                    logger.info(f"  --> New best model state saved at epoch {epoch}")

                # Basic convergence check (using configured target rate)
                if loss < 0.01 and abs(output_rate - current_target_rate) <= 2.0:
                    logger.info(f"Converged to target range at epoch {epoch}. Stopping training.")
                    break

            except RuntimeError as e:
                logger.error(f"Error during training epoch {epoch}: {str(e)}", exc_info=True)
                # Log tensor shapes to help debug
                logger.error(f"  Input spikes shape: {input_spikes_train.shape}")
                logger.error(f"  Target spikes shape: {target_spikes_train.shape}")
                raise # Re-raise after logging details

        # After training, restore the best model state found
        if best_model_state is not None:
            amn_model.load_state_dict(best_model_state['amn_state_dict'])
            # Restore heuristic state if saved
            if 'amn_direct_weight' in best_model_state:
                amn_model.direct_weight = best_model_state['amn_direct_weight']
            # Restore SIE state if saved
            if 'sie_learning_rate' in best_model_state:
                sie.learning_rate = best_model_state['sie_learning_rate']
                for unit in amn_model.units: # Re-apply to units
                    unit.stdp_learning_rate = sie.learning_rate

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
                 input_spikes_final_check = encode_number(input_number, timesteps, neurons_per_unit, device_secondary)
                 output_final = amn_model(input_spikes_final_check)
            # Use the utility function for rate calculation
            output_rate_final = get_spike_rate(output_final.to(torch.device('cpu')), neurons_per_unit)
            logger.info(f"--- Training Task Result (Loaded Model): Input={input_number}, Predicted Output Rate={output_rate_final:.2f} Hz (Target={target_output_rate_for_train_task} Hz) ---")
            # Update final_output_rate with this potentially more accurate measure from eval mode
            final_output_rate = output_rate_final

        except RuntimeError as e:
            logger.error(f"Error during final inference run: {str(e)}", exc_info=True)
            raise

        # Run generalization tests using the evaluation function
        test_results = run_tests(
            amn_model,
            test_cases,
            timesteps,
            neurons_per_unit,
            device_secondary
        )

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
             success_status = evaluate_results(
                 amn_model,
                 final_loss,
                 final_output_rate,
                 target_output_rate_for_train_task, # Pass the target rate
                 test_results,
                 final_runtime
             )

        # Save the model state (best if available, otherwise final)
        model_to_save = None
        save_path = None
        if 'best_model_state' in locals() and best_model_state is not None:
            model_to_save = best_model_state
            save_path = "amn_model_best.pth"
            logger.info(f"Saving best model state from epoch {best_model_state['epoch']} to {save_path}")
        elif 'amn_model' in locals(): # Save final state if no best state was logged or error occurred
             model_to_save = {
                 'amn_state_dict': amn_model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
                 'sie_learning_rate': sie.learning_rate if 'sie' in locals() else None,
                 'amn_direct_weight': amn_model.direct_weight if 'amn_model' in locals() else None,
                 'loss': final_loss,
                 'output_rate': final_output_rate,
                 'epoch': epoch if 'epoch' in locals() else -1 # Save last epoch number
             }
             save_path = "amn_model_final.pth"
             logger.info(f"Saving final model state to {save_path}")

        if model_to_save and save_path:
            try:
                torch.save(model_to_save, save_path)
                logger.info(f"Model state successfully saved to {save_path}")
            except Exception as save_e:
                logger.error(f"Failed to save model state to {save_path}: {save_e}", exc_info=True)


if __name__ == "__main__":
    torch.manual_seed(42) # Ensure reproducibility for random initializations
    np.random.seed(42)    # Ensure reproducibility for numpy operations if any
    try:
        main()
    except Exception as e:
        # Log crash if main() itself fails catastrophically
        logger.critical(f"CRITICAL FAILURE: main() crashed: {str(e)}", exc_info=True)

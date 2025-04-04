import torch
import torch
import logging
from utils import encode_number, get_spike_rate, encode_sequence # Assuming utils.py is in the same directory
import numpy as np # For calculations
import os # For path joining and directory creation
import time # For timestamp in log file

logger = logging.getLogger(__name__)

# 6. Tests and Evaluation (Renumbered from original script)
def run_tests(amn_model, test_cases, timesteps, neurons_per_unit, device_secondary):
    """Runs generalization tests for the sequence prediction task."""
    logger.info("--- Running Generalization Tests (Sequence Task) ---")
    # Test cases: List of tuples [((input_seq), target_number)]
    results = []
    amn_model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for testing
        for input_seq, target_num in test_cases: # Unpack sequence and target number
            # Use encode_sequence for the input
            input_spikes = encode_sequence(input_seq, timesteps, neurons_per_unit, device_secondary)
            # Pass input through the trained AMN
            output_spikes = amn_model(input_spikes)
            # Decode the output spike rate
            output_rate_hz = get_spike_rate(output_spikes.to(torch.device('cpu')), neurons_per_unit)
            # Calculate target rate based on the target number
            target_rate_hz = target_num * 10.0 # Target rate based on encoding
            # Calculate relative error
            error = abs(output_rate_hz - target_rate_hz) / target_rate_hz if target_rate_hz != 0 else float('inf')
            # Store sequence as input in results
            results.append((input_seq, target_num, output_rate_hz, error))
            # Update log message for sequence
            logger.info(f"  Test: Input={input_seq}, Target={target_rate_hz:.1f} Hz, Output={output_rate_hz:.2f} Hz, Error={error:.2%}")
    amn_model.train() # Set model back to training mode
    return results

def evaluate_results(amn_model, final_loss, final_output_rate, target_output_rate_for_train, test_results_list, total_runtime,
                     epoch_times=None, losses=None, output_rates=None, stdp_lrs=None,
                     final_inference_time=None, test_run_time=None): # Added benchmark args
    """Evaluates the training outcome based on defined criteria and reports benchmarks."""
    logger.info("--- Final Evaluation ---")
    success_overall = True
    # target_output_rate_for_train is passed in now

    # 1. Learning Ability Check
    # Did the model learn the primary task (e.g., input 3 -> output 50Hz)?
    learning_ok = final_loss < 0.01 and abs(final_output_rate - target_output_rate_for_train) <= 2.0 # Within +/- 2Hz
    logger.info(f"1. Learning Ability: Loss={final_loss:.4f} (<0.01?), Output={final_output_rate:.2f} Hz ({target_output_rate_for_train-2:.1f}-{target_output_rate_for_train+2:.1f}?) -> {'PASS' if learning_ok else 'FAIL'}")
    if not learning_ok:
        success_overall = False
        logger.warning("  Fix Suggestion: Check encoding/decoding functions. Increase training epochs or adjust STDP learning rate.")

    # 2. Self-Improvement Check
    # Did the SIE adjust the learning rate based on progress?
    initial_sie_lr = 0.03 # Assuming default initial LR for this check, could be passed if needed
    # Check if *any* unit's learning rate ended higher than the initial rate
    sie_triggered_check = any(unit.stdp_learning_rate > initial_sie_lr for unit in amn_model.units)
    # Consider successful if SIE triggered AND final loss is reasonably low
    sie_ok = sie_triggered_check and final_loss < 0.05
    logger.info(f"2. Self-Improvement: SIE Triggered={sie_triggered_check}, Final Loss Reduced (<0.05?) -> {'PASS' if sie_ok else 'FAIL'}")
    if not sie_ok:
        # Not necessarily a failure, could be optional or task-dependent
        logger.warning("  Note: SIE did not significantly increase LR or loss remained high. Check SIE logic/thresholds if improvement was expected.")
        # success_overall = False # Decide if this should be a failure condition

    # 3. Generalization Check
    # Did the model perform well on unseen test cases?
    try:
         # Check if all test case errors are below 20%
         gen_ok = all(result[3] < 0.2 for result in test_results_list)
         test_errors_str = [f'{r[3]:.2%}' for r in test_results_list]
    except (TypeError, IndexError):
         gen_ok = False
         test_errors_str = ["Error processing results"]
         logger.error("  Error processing test results list for generalization check.")

    logger.info(f"3. Generalization: Test Errors={test_errors_str} (<20%?) -> {'PASS' if gen_ok else 'FAIL'}")
    if not gen_ok:
        success_overall = False
        logger.warning("  Fix Suggestion: Requires more diverse training examples or potentially architectural changes (e.g., enhancing Coordinator recurrence, improving STDP).")

    # 4. Stability Check
    # Did the training complete without errors within a reasonable time?
    max_runtime_seconds = 7200 # 2 hours
    # Note: Crash detection relies on main loop catching exceptions.
    stab_ok = total_runtime < max_runtime_seconds
    logger.info(f"4. Stability: Runtime={total_runtime:.0f}s (<{max_runtime_seconds}s?), No Crashes (Implied by completion) -> {'PASS' if stab_ok else 'FAIL'}")
    if not stab_ok:
        success_overall = False
        logger.warning("  Fix Suggestion: If runtime excessive, profile code or reduce network size/epochs. If crashes occurred, check error logs (GPU sync, memory issues).")

    # --- Benchmarking Report ---
    logger.info("--- Performance Benchmarks ---")
    num_epochs_run = len(losses) if losses else 0
    logger.info(f"Epochs Run: {num_epochs_run}")
    if epoch_times:
        avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
        logger.info(f"Average Epoch Time: {avg_epoch_time:.3f} seconds")
    if losses:
        min_loss = min(losses) if losses else float('inf')
        logger.info(f"Minimum Loss Achieved: {min_loss:.4f}")
        # Log final loss again for clarity in this section
        logger.info(f"Final Loss (from best model): {final_loss:.4f}")
    if output_rates:
         # Log final output rate again for clarity
         logger.info(f"Final Output Rate (from best model): {final_output_rate:.2f} Hz")
    if stdp_lrs:
        final_stdp_lr = stdp_lrs[-1] if stdp_lrs else 'N/A'
        # Check if final_stdp_lr is a number before formatting
        if isinstance(final_stdp_lr, (int, float)):
             logger.info(f"Final STDP Learning Rate: {final_stdp_lr:.4f}")
        else:
             logger.info(f"Final STDP Learning Rate: {final_stdp_lr}")
    if final_inference_time is not None:
        logger.info(f"Final Inference Check Time: {final_inference_time:.3f} seconds")
    if test_run_time is not None:
        logger.info(f"Generalization Test Time: {test_run_time:.2f} seconds")
    logger.info(f"Total Runtime (incl. evaluation): {total_runtime:.0f} seconds")

    # --- Log Benchmarks to File ---
    log_dir = "benchmarks"
    log_file = os.path.join(log_dir, "analysis_log.txt")
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(log_file, 'a') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- Analysis Log: {timestamp} ---\n")
            f.write(f"Overall Result: {'SUCCESS' if success_overall else 'FAILURE'}\n")
            f.write(f"Epochs Run: {num_epochs_run}\n")
            f.write(f"Minimum Loss Achieved: {min_loss:.4f}\n")
            f.write(f"Final Loss (Best Model): {final_loss:.4f}\n")
            f.write(f"Final Output Rate (Best Model): {final_output_rate:.2f} Hz\n")
            if isinstance(final_stdp_lr, (int, float)):
                 f.write(f"Final STDP Learning Rate: {final_stdp_lr:.4f}\n")
            else:
                 f.write(f"Final STDP Learning Rate: {final_stdp_lr}\n")
            f.write(f"Average Epoch Time: {avg_epoch_time:.3f} seconds\n")
            f.write(f"Final Inference Check Time: {final_inference_time:.3f} seconds\n")
            f.write(f"Generalization Test Time: {test_run_time:.2f} seconds\n")
            f.write(f"Total Runtime: {total_runtime:.0f} seconds\n")
            f.write("Generalization Test Results:\n")
            for res in test_results_list:
                 f.write(f"  Input={res[0]}, Target={res[1]*10.0:.1f} Hz, Output={res[2]:.2f} Hz, Error={res[3]:.2%}\n")
            f.write("-" * (len(f"--- Analysis Log: {timestamp} ---")) + "\n\n")
        logger.info(f"Benchmark analysis appended to {log_file}")
    except Exception as e:
        logger.error(f"Failed to write benchmark analysis log: {e}", exc_info=True)
    # --- End Log Benchmarks to File ---

    # --- End Benchmarking Report ---


    # Final Verdict
    if success_overall:
        logger.info("--- SUCCESS: AMN prototype demonstrated core capabilities and potential. Ready for next phase. ---")
    else:
        logger.error("--- FAILURE: AMN prototype requires fixes based on evaluation warnings before proceeding. ---")
    return success_overall

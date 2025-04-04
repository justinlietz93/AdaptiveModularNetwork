import torch
import logging
from utils import encode_number, get_spike_rate # Assuming utils.py is in the same directory

logger = logging.getLogger(__name__)

# 6. Tests and Evaluation (Renumbered from original script)
def run_tests(amn_model, test_cases, timesteps, neurons_per_unit, device_secondary):
    """Runs simple generalization tests (e.g., input 1 -> output 3)."""
    logger.info("--- Running Generalization Tests ---")
    # Test cases: List of tuples [(input_number, expected_output_number)]
    results = []
    amn_model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for testing
        for x_input, target_output in test_cases:
            input_spikes = encode_number(x_input, timesteps, neurons_per_unit, device_secondary)
            # Pass input through the trained AMN
            output_spikes = amn_model(input_spikes)
            # Decode the output spike rate
            output_rate_hz = get_spike_rate(output_spikes.to(torch.device('cpu')), neurons_per_unit)
            target_rate_hz = target_output * 10.0 # Target rate based on encoding
            # Calculate relative error
            error = abs(output_rate_hz - target_rate_hz) / target_rate_hz if target_rate_hz != 0 else float('inf')
            results.append((x_input, target_output, output_rate_hz, error))
            logger.info(f"  Test: Input={x_input}, Target={target_rate_hz:.1f} Hz, Output={output_rate_hz:.2f} Hz, Error={error:.2%}")
    amn_model.train() # Set model back to training mode
    return results

def evaluate_results(amn_model, final_loss, final_output_rate, target_output_rate_for_train, test_results_list, total_runtime):
    """Evaluates the training outcome based on defined criteria."""
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

    # Final Verdict
    if success_overall:
        logger.info("--- SUCCESS: AMN prototype demonstrated core capabilities and potential. Ready for next phase. ---")
    else:
        logger.error("--- FAILURE: AMN prototype requires fixes based on evaluation warnings before proceeding. ---")
    return success_overall

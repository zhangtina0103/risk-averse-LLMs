#!/usr/bin/env python3
"""
Hyperparameter Search for DPO Training (v3 - with mentor feedback)
Optimizes for: CARA rate (70% PRIMARY), Cooperate rate (20%), Parse rate (10%)

TARGET: 70%+ CARA rate (>90% for ID val and training set)

Changes from v2:
- Added training timeout (2 hours)
- Added flush=True to all prints
- Log failed trials
- Clear GPU memory between runs
- Extended beta range: [0.05, 0.1, 0.2, 0.3, 0.5]
- Lower LR range: [5e-6, 1e-5, 2e-5]
- Allows <think> tags (matching training data format)

Usage:
    python search.py --n_trials 10 \
        --eval_repo_path ../risk-averse-ai-eval \
        --data_path ../risk-averse-ai-eval/data_cleaned.json \
        --val_csv ../risk-averse-ai-eval/data/val_set_medium_stakes.csv
"""

import subprocess
import json
import os
import sys
import time
import gc
from datetime import datetime
import pandas as pd

# Enable line buffering for real-time output
sys.stdout.reconfigure(line_buffering=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

SEARCH_SPACE = {
    'learning_rate': [5e-6, 1e-5, 2e-5],       # Lower LR range (mentor feedback)
    'beta': [0.1, 0.2, 0.3, 0.5],              # Higher beta values (mentor feedback)
    'num_epochs': [2, 3, 4],
    'max_grad_norm': [0.3, 0.5, 1.0],
}

FIXED_PARAMS = {
    'batch_size': 2,
    'gradient_accumulation_steps': 8,
    'model_name': 'Qwen/Qwen3-8B',
}

EVAL_PARAMS = {
    'base_model': 'Qwen/Qwen3-8B',
    'temperature': 0.0,
    'max_new_tokens': 4096,
    'num_situations': 50,
}

# Timeouts
TRAINING_TIMEOUT = 7200   # 2 hours for training
EVAL_TIMEOUT = 5400       # 90 minutes for evaluation


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_gpu_memory():
    """Clear GPU memory between runs"""
    print("🧹 Clearing GPU memory...", flush=True)
    try:
        subprocess.run(["python3", "-c",
            "import torch; import gc; torch.cuda.empty_cache(); gc.collect()"],
            timeout=30)
        time.sleep(5)
        print("✅ GPU memory cleared", flush=True)
    except Exception as e:
        print(f"⚠️ Could not clear GPU memory: {e}", flush=True)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def train_model(config, run_id, data_path):
    """Train model with given hyperparameters"""
    output_dir = f"./hyperparam_search/run_{run_id:03d}"

    cmd = [
        'python', 'train.py',
        '--data_path', data_path,
        '--model_name', config['model_name'],
        '--output_dir', output_dir,
        '--learning_rate', str(config['learning_rate']),
        '--beta', str(config['beta']),
        '--num_epochs', str(config['num_epochs']),
        '--batch_size', str(config['batch_size']),
        '--gradient_accumulation_steps', str(config['gradient_accumulation_steps']),
        '--max_grad_norm', str(config['max_grad_norm']),
    ]

    print(f"\n{'='*80}", flush=True)
    print(f"🚀 Training Run {run_id}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"LR={config['learning_rate']}, Beta={config['beta']}, Epochs={config['num_epochs']}, GradNorm={config['max_grad_norm']}", flush=True)
    print(f"Data: {data_path}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Timeout: {TRAINING_TIMEOUT}s ({TRAINING_TIMEOUT//3600}h {(TRAINING_TIMEOUT%3600)//60}m)", flush=True)
    print(f"{'='*80}\n", flush=True)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=TRAINING_TIMEOUT)
        # Print last 30 lines of output
        for line in result.stdout.split('\n')[-30:]:
            if line.strip():
                print(line, flush=True)

        if os.path.exists(f"{output_dir}/adapter_config.json"):
            print(f"✅ Training completed", flush=True)
            return output_dir, True, None
        print(f"❌ Model not saved", flush=True)
        return output_dir, False, "Model files not saved"
    except subprocess.TimeoutExpired:
        print(f"❌ Training timed out after {TRAINING_TIMEOUT//3600}h!", flush=True)
        return output_dir, False, f"Training timeout ({TRAINING_TIMEOUT}s)"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr[-500:] if e.stderr else 'Unknown error'
        print(f"❌ Training failed!", flush=True)
        print(f"Error: {error_msg}", flush=True)
        return output_dir, False, error_msg


def evaluate_model(model_path, run_id, eval_repo_path, val_csv):
    """Evaluate trained model using risk-averse-ai-eval repo

    IMPORTANT: This allows <think> tags to match training data format!
    """
    output_file = f"./hyperparam_search/results_{run_id:03d}.json"

    # Get absolute paths
    abs_model_path = os.path.abspath(model_path)
    abs_output_file = os.path.abspath(output_file)
    abs_val_csv = os.path.abspath(val_csv)

    # Path to evaluate.py in the external repo
    evaluate_script = os.path.join(eval_repo_path, 'evaluate.py')

    if not os.path.exists(evaluate_script):
        print(f"❌ evaluate.py not found at: {evaluate_script}", flush=True)
        print(f"   Please check eval_repo_path: {eval_repo_path}", flush=True)
        return None

    cmd = [
        'python', evaluate_script,
        '--model_path', abs_model_path,
        '--base_model', EVAL_PARAMS['base_model'],
        '--val_csv', abs_val_csv,
        '--num_situations', str(EVAL_PARAMS['num_situations']),
        '--temperature', str(EVAL_PARAMS['temperature']),
        '--max_new_tokens', str(EVAL_PARAMS['max_new_tokens']),
        '--output', abs_output_file,
        '--save_responses',
        # NOTE: --disable_thinking REMOVED to allow <think> tags like training data
    ]

    print(f"\n📊 Evaluating Run {run_id}...", flush=True)
    print(f"   Using evaluate.py from: {evaluate_script}", flush=True)
    print(f"   Validation CSV: {val_csv}", flush=True)
    print(f"   Timeout: {EVAL_TIMEOUT}s ({EVAL_TIMEOUT//60}m)", flush=True)
    print(f"   ⚠️  Allowing <think> tags to match training format", flush=True)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=EVAL_TIMEOUT)
        print(result.stdout, flush=True)

        if not os.path.exists(output_file):
            print(f"❌ Results file not found", flush=True)
            return None

        with open(output_file, 'r') as f:
            results = json.load(f)

        return {
            'parse_rate': results['metrics']['parse_rate'],
            'cooperate_rate': results['metrics']['cooperate_rate'],
            'cara_rate': results['metrics']['best_cara_rate'],
            'num_valid': results['num_valid'],
            'num_total': results['num_total'],
        }
    except subprocess.TimeoutExpired:
        print(f"❌ Evaluation timed out ({EVAL_TIMEOUT//60} min)", flush=True)
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed!", flush=True)
        print(f"   stdout: {e.stdout[-500:] if e.stdout else 'None'}", flush=True)
        print(f"   stderr: {e.stderr[-500:] if e.stderr else 'None'}", flush=True)
        return None
    except Exception as e:
        print(f"❌ Evaluation error: {e}", flush=True)
        return None


def compute_score(metrics):
    """
    Composite score: 70% CARA + 20% Cooperate + 10% Parse

    CARA is PRIMARY - weighted at 70% since it's the main objective
    TARGET: 70%+ CARA rate (mentor wants as high as possible)

    Parse penalty: If < 90%, score is multiplied by 0.5 (harsh penalty)
    """
    parse_penalty = 1.0 if metrics['parse_rate'] >= 0.9 else 0.5

    score = (
        0.7 * metrics['cara_rate'] +
        0.2 * metrics['cooperate_rate'] +
        0.1 * metrics['parse_rate']
    ) * parse_penalty

    return score


# ============================================================================
# SEARCH
# ============================================================================

def random_search(n_trials=10, seed=42, eval_repo_path='../risk-averse-ai-eval',
                  data_path='data_cleaned.json', val_csv='data/val_set_medium_stakes.csv'):
    """Random search over hyperparameter space"""
    import random
    random.seed(seed)

    print(f"\n{'='*80}", flush=True)
    print(f"🎲 RANDOM SEARCH ({n_trials} trials)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"🎯 Optimization objective:", flush=True)
    print(f"   70% CARA Rate (PRIMARY - TARGET: 70%+)", flush=True)
    print(f"   20% Cooperate Rate (correlates with CARA)", flush=True)
    print(f"   10% Parse Rate (quality check, must be 90%+)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"📁 Paths:", flush=True)
    print(f"   Training data:  {data_path}", flush=True)
    print(f"   Validation CSV: {val_csv}", flush=True)
    print(f"   Eval repo:      {eval_repo_path}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"🔧 Search Space (updated per mentor feedback):", flush=True)
    print(f"   Learning rates: {SEARCH_SPACE['learning_rate']} (lower range)", flush=True)
    print(f"   Beta values:    {SEARCH_SPACE['beta']} (higher range)", flush=True)
    print(f"   Epochs:         {SEARCH_SPACE['num_epochs']}", flush=True)
    print(f"   Grad norms:     {SEARCH_SPACE['max_grad_norm']}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"⏱️  Timeouts:", flush=True)
    print(f"   Training: {TRAINING_TIMEOUT}s ({TRAINING_TIMEOUT//3600}h)", flush=True)
    print(f"   Evaluation: {EVAL_TIMEOUT}s ({EVAL_TIMEOUT//60}m)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"⚠️  IMPORTANT: Allowing <think> tags in evaluation", flush=True)
    print(f"   This matches the training data format with thinking tags", flush=True)
    print(f"{'='*80}\n", flush=True)

    results = []
    best_score = 0
    best_run = None

    for i in range(1, n_trials + 1):
        # Clear GPU memory before each run
        clear_gpu_memory()

        # Sample random config
        config = FIXED_PARAMS.copy()
        for key, values in SEARCH_SPACE.items():
            config[key] = random.choice(values)

        # Train
        model_path, success, error_msg = train_model(config, i, data_path)

        if not success:
            # Log failed training (mentor feedback: save even failures!)
            print(f"⚠️  Logging failed training run {i}", flush=True)
            failed_result = {
                'run_id': i,
                'config': config,
                'status': 'training_failed',
                'error': error_msg,
                'metrics': None,
                'score': 0,
                'timestamp': datetime.now().isoformat(),
            }
            results.append(failed_result)
            save_results(results)
            print(f"⚠️  Skipping evaluation\n", flush=True)
            continue

        # Evaluate
        metrics = evaluate_model(model_path, i, eval_repo_path, val_csv)
        if metrics is None:
            # Log failed evaluation
            print(f"⚠️  Logging failed evaluation run {i}", flush=True)
            failed_result = {
                'run_id': i,
                'config': config,
                'status': 'evaluation_failed',
                'error': 'Evaluation returned None',
                'metrics': None,
                'score': 0,
                'timestamp': datetime.now().isoformat(),
            }
            results.append(failed_result)
            save_results(results)
            print(f"⚠️  Skipping this run\n", flush=True)
            continue

        # Score
        score = compute_score(metrics)

        result = {
            'run_id': i,
            'config': config,
            'status': 'success',
            'error': None,
            'metrics': metrics,
            'score': score,
            'timestamp': datetime.now().isoformat(),
        }

        results.append(result)

        # Track best
        if score > best_score:
            best_score = score
            best_run = i

        # Print summary
        print(f"\n{'='*80}", flush=True)
        print(f"✅ Run {i}/{n_trials} Complete", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"CARA Rate:      {metrics['cara_rate']*100:6.1f}% ⭐ PRIMARY (target: 70%+)", flush=True)
        print(f"Cooperate Rate: {metrics['cooperate_rate']*100:6.1f}% (20% weight)", flush=True)
        print(f"Parse Rate:     {metrics['parse_rate']*100:6.1f}% (10% weight)", flush=True)
        print(f"Valid/Total:    {metrics['num_valid']}/{metrics['num_total']}", flush=True)
        print(f"Composite Score: {score:.4f}", flush=True)

        # Check if statistically significant
        if metrics['cara_rate'] >= 0.51:
            print(f"✅ Statistically significant (p < 0.05)!", flush=True)
        if metrics['cara_rate'] >= 0.70:
            print(f"🎯 TARGET REACHED! CARA ≥ 70%!", flush=True)

        if i == best_run:
            print(f"🏆 NEW BEST!", flush=True)
        print(f"{'='*80}\n", flush=True)

        # Save intermediate
        save_results(results)

    return results


def save_results(results):
    """Save to JSON and CSV"""
    os.makedirs('./hyperparam_search', exist_ok=True)

    # Full JSON
    with open('./hyperparam_search/all_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # CSV summary (only successful runs for sorting)
    rows = []
    for r in results:
        row = {
            'run_id': r['run_id'],
            'status': r.get('status', 'unknown'),
            'score': r['score'],
            'learning_rate': r['config']['learning_rate'],
            'beta': r['config']['beta'],
            'num_epochs': r['config']['num_epochs'],
            'max_grad_norm': r['config']['max_grad_norm'],
        }
        if r.get('metrics'):
            row['cara_rate'] = r['metrics']['cara_rate']
            row['cooperate_rate'] = r['metrics']['cooperate_rate']
            row['parse_rate'] = r['metrics']['parse_rate']
            row['num_valid'] = r['metrics']['num_valid']
        else:
            row['cara_rate'] = None
            row['cooperate_rate'] = None
            row['parse_rate'] = None
            row['num_valid'] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('score', ascending=False)
    df.to_csv('./hyperparam_search/results_summary.csv', index=False)

    print(f"💾 Saved to ./hyperparam_search/", flush=True)


def print_summary(results):
    """Print final summary"""
    # Filter successful runs
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') != 'success']

    print(f"\n{'='*80}", flush=True)
    print("📈 FINAL RESULTS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total runs: {len(results)}", flush=True)
    print(f"Successful: {len(successful)}", flush=True)
    print(f"Failed: {len(failed)}", flush=True)

    if failed:
        print(f"\n⚠️  Failed Runs:", flush=True)
        for r in failed:
            print(f"   Run {r['run_id']}: {r.get('status')} - {r.get('error', 'Unknown')[:50]}", flush=True)

    if not successful:
        print("\n❌ No successful runs!", flush=True)
        return

    # Sort by score
    successful = sorted(successful, key=lambda x: x['score'], reverse=True)

    # Top 5
    print(f"\n🏆 Top 5 Configurations:", flush=True)
    print(f"{'='*80}\n", flush=True)
    for i, r in enumerate(successful[:min(5, len(successful))], 1):
        cara = r['metrics']['cara_rate']*100
        sig_marker = "✅ SIG" if cara >= 51 else ""
        target_marker = "🎯 TARGET" if cara >= 70 else ""
        print(f"Rank {i} (Score: {r['score']:.4f}) {sig_marker} {target_marker}", flush=True)
        print(f"  CARA={cara:.1f}%, Coop={r['metrics']['cooperate_rate']*100:.1f}%, Parse={r['metrics']['parse_rate']*100:.1f}%", flush=True)
        print(f"  LR={r['config']['learning_rate']}, Beta={r['config']['beta']}, Epochs={r['config']['num_epochs']}, GradNorm={r['config']['max_grad_norm']}", flush=True)
        print(flush=True)

    # Best config
    best = successful[0]
    print(f"{'='*80}", flush=True)
    print("🏆 BEST CONFIGURATION", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Composite Score: {best['score']:.4f}", flush=True)
    print(f"\n📊 Metrics:", flush=True)
    print(f"  CARA Rate:      {best['metrics']['cara_rate']*100:6.1f}% (target: 70%+)", flush=True)
    print(f"  Cooperate Rate: {best['metrics']['cooperate_rate']*100:6.1f}%", flush=True)
    print(f"  Parse Rate:     {best['metrics']['parse_rate']*100:6.1f}% (target: 90%+)", flush=True)
    print(f"  Valid/Total:    {best['metrics']['num_valid']}/{best['metrics']['num_total']}", flush=True)

    if best['metrics']['cara_rate'] >= 0.70:
        print(f"\n🎯 TARGET ACHIEVED! CARA ≥ 70%", flush=True)
    elif best['metrics']['cara_rate'] >= 0.51:
        print(f"\n✅ Statistically significant but below 70% target", flush=True)
    else:
        print(f"\n⚠️  Below statistical significance threshold (51%)", flush=True)

    print(f"\n⚙️  Command to retrain with best hyperparameters:", flush=True)
    print(f"python train.py \\", flush=True)
    print(f"  --learning_rate {best['config']['learning_rate']} \\", flush=True)
    print(f"  --beta {best['config']['beta']} \\", flush=True)
    print(f"  --num_epochs {best['config']['num_epochs']} \\", flush=True)
    print(f"  --max_grad_norm {best['config']['max_grad_norm']} \\", flush=True)
    print(f"  --output_dir ./final_best_model", flush=True)
    print(f"{'='*80}\n", flush=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter search for risk-averse DPO (v3 with mentor feedback)')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of trials (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--eval_repo_path', type=str, default='../risk-averse-ai-eval',
                        help='Path to risk-averse-ai-eval repo (default: ../risk-averse-ai-eval)')
    parser.add_argument('--data_path', type=str, default='../risk-averse-ai-eval/data_cleaned.json',
                        help='Path to training data JSON (default: ../risk-averse-ai-eval/data_cleaned.json)')
    parser.add_argument('--val_csv', type=str, default='../risk-averse-ai-eval/data/val_set_medium_stakes.csv',
                        help='Path to validation CSV (default: ../risk-averse-ai-eval/data/val_set_medium_stakes.csv)')

    args = parser.parse_args()

    # Verify all required files exist
    errors = []

    eval_script = os.path.join(args.eval_repo_path, 'evaluate.py')
    if not os.path.exists(eval_script):
        errors.append(f"❌ evaluate.py not found at {eval_script}")

    if not os.path.exists(args.data_path):
        errors.append(f"❌ Training data not found at {args.data_path}")

    if not os.path.exists(args.val_csv):
        errors.append(f"❌ Validation CSV not found at {args.val_csv}")

    if errors:
        print("\n" + "="*80, flush=True)
        print("ERROR: Missing required files", flush=True)
        print("="*80, flush=True)
        for error in errors:
            print(error, flush=True)
        print("\nPlease provide correct paths:", flush=True)
        print("  python search.py \\", flush=True)
        print("    --eval_repo_path /path/to/risk-averse-ai-eval \\", flush=True)
        print("    --data_path /path/to/data_cleaned.json \\", flush=True)
        print("    --val_csv /path/to/val_set_medium_stakes.csv", flush=True)
        print(flush=True)
        return

    print(f"\n🔬 HYPERPARAMETER OPTIMIZATION FOR RISK-AVERSE DPO (v3)", flush=True)
    print(f"Trials: {args.n_trials}, Seed: {args.seed}", flush=True)
    print(f"Target: 70%+ CARA rate\n", flush=True)

    # Run search
    results = random_search(
        n_trials=args.n_trials,
        seed=args.seed,
        eval_repo_path=args.eval_repo_path,
        data_path=args.data_path,
        val_csv=args.val_csv
    )

    # Print summary
    print_summary(results)

    print("\n✅ Hyperparameter search complete!", flush=True)
    print(f"📁 Results: ./hyperparam_search/", flush=True)
    print(f"   - results_summary.csv (sortable)", flush=True)
    print(f"   - all_results.json (full details)", flush=True)
    print(f"   - results_XXX.json (individual eval results with full CoTs)\n", flush=True)


if __name__ == '__main__':
    main()

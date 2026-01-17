#!/usr/bin/env python3
"""
Hyperparameter Search for DPO Training
Optimizes for: CARA rate (70% PRIMARY), Cooperate rate (20%), Parse rate (10%)

Usage:
    python hyperparam_search.py --n_trials 10
"""

import subprocess
import json
import os
import itertools
from datetime import datetime
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

SEARCH_SPACE = {
    'learning_rate': [1e-5, 2e-5, 5e-5],
    'beta': [0.05, 0.1, 0.2],
    'num_epochs': [2, 3, 4],
    'max_grad_norm': [0.3, 0.5, 1.0],
}

FIXED_PARAMS = {
    'batch_size': 2,
    'gradient_accumulation_steps': 8,
    'data_path': 'data_cleaned.json',
    'model_name': 'Qwen/Qwen3-8B',
}

EVAL_PARAMS = {
    'val_csv': 'data/val_set_medium_stakes.csv',
    'num_situations': 50,
    'base_model': 'Qwen/Qwen3-8B',
    'temperature': 0.0,
    'max_new_tokens': 4096,
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def train_model(config, run_id):
    """Train model with given hyperparameters"""
    output_dir = f"./hyperparam_search/run_{run_id:03d}"

    cmd = [
        'python', 'train.py',
        '--data_path', config['data_path'],
        '--model_name', config['model_name'],
        '--output_dir', output_dir,
        '--learning_rate', str(config['learning_rate']),
        '--beta', str(config['beta']),
        '--num_epochs', str(config['num_epochs']),
        '--batch_size', str(config['batch_size']),
        '--gradient_accumulation_steps', str(config['gradient_accumulation_steps']),
        '--max_grad_norm', str(config['max_grad_norm']),
    ]

    print(f"\n{'='*80}")
    print(f"🚀 Training Run {run_id}")
    print(f"{'='*80}")
    print(f"LR={config['learning_rate']}, Beta={config['beta']}, Epochs={config['num_epochs']}, GradNorm={config['max_grad_norm']}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Print last 30 lines of output
        for line in result.stdout.split('\n')[-30:]:
            if line.strip():
                print(line)

        if os.path.exists(f"{output_dir}/adapter_config.json"):
            print(f"✅ Training completed")
            return output_dir, True
        print(f"❌ Model not saved")
        return output_dir, False
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed!")
        if e.stderr:
            print(f"Error: {e.stderr[-500:]}")
        return output_dir, False


def evaluate_model(model_path, run_id):
    """Evaluate trained model"""
    output_file = f"./hyperparam_search/results_{run_id:03d}.json"

    cmd = [
        'python', 'evaluate.py',
        '--model_path', model_path,
        '--base_model', EVAL_PARAMS['base_model'],
        '--val_csv', EVAL_PARAMS['val_csv'],
        '--num_situations', str(EVAL_PARAMS['num_situations']),
        '--temperature', str(EVAL_PARAMS['temperature']),
        '--max_new_tokens', str(EVAL_PARAMS['max_new_tokens']),
        '--output', output_file,
        '--save_responses',
        '--disable_thinking',
    ]

    print(f"\n📊 Evaluating Run {run_id}...")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        print(result.stdout)

        if not os.path.exists(output_file):
            print(f"❌ Results file not found")
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
        print(f"❌ Evaluation timed out (30 min)")
        return None
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return None


def compute_score(metrics):
    """
    Composite score: 70% CARA + 20% Cooperate + 10% Parse

    CARA is PRIMARY - weighted at 70% since it's the main objective
    Cooperate is secondary (20%) - usually correlates with CARA anyway
    Parse is quality check (10%) - must be high but not the goal

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

def random_search(n_trials=10, seed=42):
    """Random search over hyperparameter space"""
    import random
    random.seed(seed)

    print(f"\n{'='*80}")
    print(f"🎲 RANDOM SEARCH ({n_trials} trials)")
    print(f"{'='*80}")
    print(f"🎯 Optimization objective:")
    print(f"   70% CARA Rate (PRIMARY - choose CARA-optimal, target 80%+)")
    print(f"   20% Cooperate Rate (correlates with CARA)")
    print(f"   10% Parse Rate (quality check, must be 90%+)")
    print(f"{'='*80}\n")

    results = []
    best_score = 0
    best_run = None

    for i in range(1, n_trials + 1):
        # Sample random config
        config = FIXED_PARAMS.copy()
        for key, values in SEARCH_SPACE.items():
            config[key] = random.choice(values)

        # Train
        model_path, success = train_model(config, i)
        if not success:
            print(f"⚠️  Skipping evaluation\n")
            continue

        # Evaluate
        metrics = evaluate_model(model_path, i)
        if metrics is None:
            print(f"⚠️  Skipping this run\n")
            continue

        # Score
        score = compute_score(metrics)

        result = {
            'run_id': i,
            'config': config,
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
        print(f"\n{'='*80}")
        print(f"✅ Run {i}/{n_trials} Complete")
        print(f"{'='*80}")
        print(f"CARA Rate:      {metrics['cara_rate']*100:6.1f}% ⭐ PRIMARY (70% weight)")
        print(f"Cooperate Rate: {metrics['cooperate_rate']*100:6.1f}% (20% weight)")
        print(f"Parse Rate:     {metrics['parse_rate']*100:6.1f}% (10% weight)")
        print(f"Valid/Total:    {metrics['num_valid']}/{metrics['num_total']}")
        print(f"Composite Score: {score:.4f}")
        if i == best_run:
            print(f"🏆 NEW BEST!")
        print(f"{'='*80}\n")

        # Save intermediate
        save_results(results)

    return results


def save_results(results):
    """Save to JSON and CSV"""
    os.makedirs('./hyperparam_search', exist_ok=True)

    # Full JSON
    with open('./hyperparam_search/all_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # CSV summary
    rows = []
    for r in results:
        rows.append({
            'run_id': r['run_id'],
            'score': r['score'],
            'cara_rate': r['metrics']['cara_rate'],
            'cooperate_rate': r['metrics']['cooperate_rate'],
            'parse_rate': r['metrics']['parse_rate'],
            'num_valid': r['metrics']['num_valid'],
            'learning_rate': r['config']['learning_rate'],
            'beta': r['config']['beta'],
            'num_epochs': r['config']['num_epochs'],
            'max_grad_norm': r['config']['max_grad_norm'],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('score', ascending=False)
    df.to_csv('./hyperparam_search/results_summary.csv', index=False)

    print(f"💾 Saved to ./hyperparam_search/")


def print_summary(results):
    """Print final summary"""
    if not results:
        print("\n❌ No successful runs!")
        return

    # Sort by score
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*80}")
    print("📈 FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Total successful runs: {len(results)}\n")

    # Top 5
    print(f"🏆 Top 5 Configurations:")
    print(f"{'='*80}\n")
    for i, r in enumerate(results[:min(5, len(results))], 1):
        print(f"Rank {i} (Score: {r['score']:.4f})")
        print(f"  CARA={r['metrics']['cara_rate']*100:.1f}%, Coop={r['metrics']['cooperate_rate']*100:.1f}%, Parse={r['metrics']['parse_rate']*100:.1f}%")
        print(f"  LR={r['config']['learning_rate']}, Beta={r['config']['beta']}, Epochs={r['config']['num_epochs']}, GradNorm={r['config']['max_grad_norm']}")
        print()

    # Best config
    best = results[0]
    print(f"{'='*80}")
    print("🏆 BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"Composite Score: {best['score']:.4f}")
    print(f"\n📊 Metrics:")
    print(f"  CARA Rate:      {best['metrics']['cara_rate']*100:6.1f}% (target: 80%+)")
    print(f"  Cooperate Rate: {best['metrics']['cooperate_rate']*100:6.1f}%")
    print(f"  Parse Rate:     {best['metrics']['parse_rate']*100:6.1f}% (target: 90%+)")
    print(f"  Valid/Total:    {best['metrics']['num_valid']}/{best['metrics']['num_total']}")
    print(f"\n⚙️  Command to retrain with best hyperparameters:")
    print(f"python train.py \\")
    print(f"  --learning_rate {best['config']['learning_rate']} \\")
    print(f"  --beta {best['config']['beta']} \\")
    print(f"  --num_epochs {best['config']['num_epochs']} \\")
    print(f"  --max_grad_norm {best['config']['max_grad_norm']} \\")
    print(f"  --output_dir ./final_best_model")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter search for risk-averse DPO')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of trials (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    print(f"\n🔬 HYPERPARAMETER OPTIMIZATION FOR RISK-AVERSE DPO")
    print(f"Trials: {args.n_trials}, Seed: {args.seed}\n")

    # Run search
    results = random_search(n_trials=args.n_trials, seed=args.seed)

    # Print summary
    print_summary(results)

    print("\n✅ Hyperparameter search complete!")
    print(f"📁 Results: ./hyperparam_search/")
    print(f"   - results_summary.csv (sortable)")
    print(f"   - all_results.json (full details)\n")


if __name__ == '__main__':
    main()

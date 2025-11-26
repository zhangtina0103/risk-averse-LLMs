#!/bin/bash

# All-in-one script for risk-averse DPO training from Excel data

set -e  # Exit on error

echo "==================================="
echo "Risk-Averse DPO Training Pipeline"
echo "==================================="

# Configuration
EXCEL_FILE="../strict_disagreements_10k_with_prompts_and_bad_formats.xlsx"
DATASET_FILE="risk_averse_dpo.json"
MODEL_NAME="microsoft/phi-2"
OUTPUT_DIR="./results-risk-averse"
WANDB_PROJECT="risk-averse-dpo"
SAMPLE_SIZE=500

# Step 1: Convert Excel to DPO format
echo ""
echo "Step 1: Converting Excel to DPO format..."
python convert_excel_to_dpo.py \
    --input "$EXCEL_FILE" \
    --output "$DATASET_FILE" \
    --sample_size $SAMPLE_SIZE

if [ ! -f "$DATASET_FILE" ]; then
    echo "Error: Dataset conversion failed!"
    exit 1
fi

echo "✓ Dataset created: $DATASET_FILE"

# Step 2: Train the model
echo ""
echo "Step 2: Training risk-averse model with DPO..."
python train_risk_preferences.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET_FILE" \
    --epochs 3 \
    --beta 0.5 \
    --batch_size 4 \
    --lr 1e-5 \
    --wandb_project "$WANDB_PROJECT" \
    --output_dir "$OUTPUT_DIR"

echo "✓ Training complete!"
echo ""
echo "Model saved to: $OUTPUT_DIR/final-model"
echo ""
echo "Next steps:"
echo "1. Evaluate model on test scenarios"
echo "2. Compare risk-averse vs risk-neutral choices"
echo "3. Check W&B dashboard for training metrics"
echo ""

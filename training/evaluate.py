"""
Model evaluation script.
Evaluates trained models and generates performance metrics.
"""

import os
import argparse
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def evaluate_model(model, dataloader, device, model_type='video'):
    """
    Evaluate model on test set.

    Args:
        model: Trained PyTorch model
        dataloader: Test data loader
        device: Device to run evaluation on
        model_type: 'video' or 'audio'

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1 (fake)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    auc = roc_auc_score(all_labels, all_probabilities)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    }

    return metrics, cm, (fpr, tpr)


def plot_confusion_matrix(cm, output_path, model_type='video'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(f'{model_type.capitalize()} Model - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(fpr, tpr, auc, output_path, model_type='video'):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_type.capitalize()} Model - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to {output_path}")


def generate_performance_report(metrics, output_path, model_type='video'):
    """Generate markdown performance report"""
    report = f"""# {model_type.capitalize()} Model Performance Report

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) |
| Precision | {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%) |
| Recall | {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%) |
| F1 Score | {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%) |
| AUC | {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%) |

## Confusion Matrix

|           | Predicted Real | Predicted Fake |
|-----------|----------------|----------------|
| **True Real** | {metrics['confusion_matrix'][0][0]} (TN) | {metrics['confusion_matrix'][0][1]} (FP) |
| **True Fake** | {metrics['confusion_matrix'][1][0]} (FN) | {metrics['confusion_matrix'][1][1]} (TP) |

### Performance Analysis

- **True Positives (TP)**: {metrics['confusion_matrix'][1][1]} - Correctly identified fakes
- **True Negatives (TN)**: {metrics['confusion_matrix'][0][0]} - Correctly identified reals
- **False Positives (FP)**: {metrics['confusion_matrix'][0][1]} - Real classified as fake
- **False Negatives (FN)**: {metrics['confusion_matrix'][1][0]} - Fake classified as real

**False Positive Rate**: {metrics['confusion_matrix'][0][1] / (metrics['confusion_matrix'][0][0] + metrics['confusion_matrix'][0][1]):.4f}

**False Negative Rate**: {metrics['confusion_matrix'][1][0] / (metrics['confusion_matrix'][1][0] + metrics['confusion_matrix'][1][1]):.4f}

## Interpretation

- The model achieved **{metrics['accuracy']*100:.2f}% accuracy** on the test set.
- **Precision** of {metrics['precision']*100:.2f}% means that when the model predicts "fake", it is correct {metrics['precision']*100:.2f}% of the time.
- **Recall** of {metrics['recall']*100:.2f}% means the model correctly identifies {metrics['recall']*100:.2f}% of all fake samples.
- **AUC** of {metrics['auc']:.4f} indicates {"excellent" if metrics['auc'] > 0.9 else "good" if metrics['auc'] > 0.8 else "fair"} discriminative ability.

---
*Generated by DeepSafe Evaluation Script*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Performance report saved to {output_path}")


def main(args):
    """Main evaluation function"""

    print(f"Evaluating {args.model_type} model...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    # (This is a placeholder - actual implementation would load the specific model)
    # For now, print instructions
    print("\n" + "="*50)
    print("MODEL EVALUATION INSTRUCTIONS")
    print("="*50)
    print("\nTo evaluate a trained model:")
    print("1. Load your trained PyTorch model")
    print("2. Prepare test dataloader")
    print("3. Call evaluate_model() function")
    print("4. Generate visualizations and report")
    print("\nExample:")
    print("```python")
    print("from evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve")
    print("metrics, cm, (fpr, tpr) = evaluate_model(model, test_loader, device)")
    print("plot_confusion_matrix(cm, 'confusion_matrix.png')")
    print("plot_roc_curve(fpr, tpr, metrics['auc'], 'roc_curve.png')")
    print("```")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--model_type', type=str, choices=['video', 'audio'],
                        default='video', help='Type of model to evaluate')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='../docs',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run evaluation
    main(args)

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from ultralytics import YOLO
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(
    weights: str,
    data_yaml: str,
    output_dir: str = "docs",
    conf_threshold: float = 0.25
) -> Dict:

    # Load model
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    
    logger.info(f"Loading model from {weights}")
    model = YOLO(str(weights_path))
    
    # Load data config
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")
    
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', [])
    logger.info(f"Classes: {class_names}")
    
    # Run validation on test set
    logger.info("Running validation on test set...")
    results = model.val(
        data=str(data_path),
        split='test',
        conf=conf_threshold,
        save_json=False,
        plots=True
    )
    
    # Extract detection metrics
    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
    }
    
    logger.info(f"Detection Metrics:")
    logger.info(f"  mAP@0.5: {metrics['mAP50']:.4f}")
    logger.info(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    
    # Get test images for image-level classification metrics
    test_dir = Path(data_config.get('test', 'data/roboflow_export/test'))
    if not test_dir.exists():
        logger.warning(f"Test directory not found: {test_dir}")
        test_dir = Path(data_config.get('path', '.')) / 'test'
    
    # Collect image-level predictions
    image_files = list(test_dir.glob('images/*.jpg')) + \
                 list(test_dir.glob('images/*.jpeg')) + \
                 list(test_dir.glob('images/*.png'))
    
    if not image_files:
        logger.warning("No test images found for classification metrics")
        image_files = []
    
    y_true = []
    y_pred = []
    
    logger.info(f"Computing image-level classification metrics on {len(image_files)} images...")
    
    for img_file in image_files:
        # Get ground truth from label file
        label_file = test_dir / 'labels' / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    gt_class = int(lines[0].split()[0])
                    y_true.append(gt_class)
                    
                    # Run prediction
                    pred_results = model.predict(
                        source=str(img_file),
                        conf=conf_threshold,
                        verbose=False
                    )
                    
                    # Get highest confidence prediction
                    if len(pred_results[0].boxes) > 0:
                        pred_class = int(pred_results[0].boxes[0].cls[0])
                    else:
                        pred_class = -1  # No detection
                    
                    y_pred.append(pred_class)
    
    # Compute classification metrics
    if y_true and y_pred:
        # Filter out no-detection cases for main metrics
        valid_indices = [i for i, p in enumerate(y_pred) if p != -1]
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        if y_true_valid and y_pred_valid:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_valid, y_pred_valid, average='weighted', zero_division=0
            )
            accuracy = np.mean(np.array(y_true_valid) == np.array(y_pred_valid))
            
            metrics['classification_accuracy'] = float(accuracy)
            metrics['classification_precision'] = float(precision)
            metrics['classification_recall'] = float(recall)
            metrics['classification_f1'] = float(f1)
            
            logger.info(f"\nClassification Metrics (image-level):")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true_valid, y_pred_valid)
            
            # Plot confusion matrix
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.title('Confusion Matrix - Image-Level Classification')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            cm_path = output_path / 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Save results to markdown
    output_path = Path(output_dir)
    results_md_path = output_path / 'results.md'
    
    with open(results_md_path, 'w') as f:
        f.write("# Counterfeit Shoe Detector - Evaluation Results\n\n")
        f.write("## Detection Metrics\n\n")
        f.write(f"- **mAP@0.5**: {metrics['mAP50']:.4f}\n")
        f.write(f"- **mAP@0.5:0.95**: {metrics['mAP50-95']:.4f}\n")
        f.write(f"- **Precision**: {metrics['precision']:.4f}\n")
        f.write(f"- **Recall**: {metrics['recall']:.4f}\n\n")
        
        if 'classification_accuracy' in metrics:
            f.write("## Classification Metrics (Image-Level)\n\n")
            f.write(f"- **Accuracy**: {metrics['classification_accuracy']:.4f}\n")
            f.write(f"- **Precision**: {metrics['classification_precision']:.4f}\n")
            f.write(f"- **Recall**: {metrics['classification_recall']:.4f}\n")
            f.write(f"- **F1-Score**: {metrics['classification_f1']:.4f}\n\n")
        
        f.write("## Model Configuration\n\n")
        f.write(f"- **Model**: {weights}\n")
        f.write(f"- **Confidence Threshold**: {conf_threshold}\n")
        f.write(f"- **Classes**: {', '.join(class_names)}\n\n")
        
        f.write("## Confusion Matrix\n\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n")
    
    logger.info(f"Results saved to {results_md_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model for counterfeit shoe detection"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs",
        help="Output directory (default: docs)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    
    args = parser.parse_args()
    
    try:
        metrics = evaluate_model(
            weights=args.weights,
            data_yaml=args.data,
            output_dir=args.output,
            conf_threshold=args.conf
        )
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
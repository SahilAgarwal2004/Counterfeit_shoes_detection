import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_yaml(yaml_path: str) -> Dict:
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Data YAML not found: {yaml_path}")
    
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded data config from {yaml_path}")
    logger.info(f"  Classes: {config.get('names', [])}")
    logger.info(f"  Number of classes: {config.get('nc', 'unknown')}")
    
    return config


def validate_roboflow_export(export_dir: str) -> bool:
    export_path = Path(export_dir)
    
    required_dirs = ['train', 'valid', 'test']
    required_files = ['data.yaml']
    
    logger.info(f"Validating Roboflow export at {export_dir}")
    
    for dir_name in required_dirs:
        dir_path = export_path / dir_name
        if not dir_path.exists():
            logger.warning(f"Missing directory: {dir_name}")
            return False
        
        images_path = dir_path / 'images'
        labels_path = dir_path / 'labels'
        
        if not images_path.exists() or not labels_path.exists():
            logger.warning(f"Missing images or labels in {dir_name}")
            return False
    
    for file_name in required_files:
        file_path = export_path / file_name
        if not file_path.exists():
            logger.warning(f"Missing file: {file_name}")
            return False
    
    logger.info("Roboflow export structure is valid")
    return True


def draw_bounding_boxes(
    image: np.ndarray,
    predictions: List[Dict],
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> np.ndarray:
    img_copy = image.copy()
    
    if class_colors is None:
        class_colors = {
            'real': (0, 255, 0),      # Green
            'fake': (0, 0, 255),      # Red
            'authentic': (0, 255, 0),  # Green
            'counterfeit': (0, 0, 255) # Red
        }
    
    for pred in predictions:
        class_name = pred['class']
        confidence = pred['confidence']
        bbox = pred['bbox']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color for this class
        color = class_colors.get(class_name.lower(), (255, 255, 0))  # Default yellow
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_w, label_h = label_size
        
        # Draw label background
        cv2.rectangle(
            img_copy,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 10, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_copy,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return img_copy


def safe_load_model(weights_path: str):
    from ultralytics import YOLO
    
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            f"Please train a model first using: python yolo/train.py"
        )
    
    try:
        logger.info(f"Loading model from {weights_path}")
        model = YOLO(str(weights))
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def get_default_augmentation_config() -> Dict:

    return {
        'hsv_h': 0.015,      # HSV-Hue augmentation
        'hsv_s': 0.7,        # HSV-Saturation augmentation
        'hsv_v': 0.4,        # HSV-Value augmentation
        'degrees': 0.0,      # Rotation (+/- degrees)
        'translate': 0.1,    # Translation (+/- fraction)
        'scale': 0.5,        # Scale (+/- gain)
        'shear': 0.0,        # Shear (+/- degrees)
        'perspective': 0.0,  # Perspective (+/- fraction)
        'flipud': 0.0,       # Flip up-down probability
        'fliplr': 0.5,       # Flip left-right probability
        'mosaic': 1.0,       # Mosaic probability
        'mixup': 0.0         # MixUp probability
    }


def normalize_data_yaml(input_yaml: str, output_yaml: str) -> None:

    config = load_data_yaml(input_yaml)
    
    # Ensure paths are relative to project root
    output_path = Path(output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Normalized data.yaml saved to {output_yaml}")


def sanity_check_inference(weights_path: str, test_image: str) -> bool:
 
    try:
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        
        # Add parent directory to path
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from yolo.infer import infer_single_image
        
        logger.info("Running sanity check...")
        result = infer_single_image(weights_path, test_image)
        
        # Check required keys
        required_keys = ['image', 'predictions', 'final_label', 'final_confidence']
        for key in required_keys:
            if key not in result:
                logger.error(f"Missing key in result: {key}")
                return False
        
        # Check predictions format
        if result['predictions']:
            pred = result['predictions'][0]
            pred_keys = ['class', 'confidence', 'bbox']
            for key in pred_keys:
                if key not in pred:
                    logger.error(f"Missing key in prediction: {key}")
                    return False
        
        logger.info("Sanity check passed!")
        logger.info(f"  Image: {result['image']}")
        logger.info(f"  Final label: {result['final_label']}")
        logger.info(f"  Confidence: {result['final_confidence']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Sanity check failed: {e}")
        return False


def format_prediction_summary(prediction: Dict) -> str:

    lines = [
        f"Image: {Path(prediction['image']).name}",
        f"Final Label: {prediction['final_label']}",
        f"Confidence: {prediction['final_confidence']:.3f}",
        f"\nDetections: {len(prediction['predictions'])}"
    ]
    
    for i, pred in enumerate(prediction['predictions'], 1):
        lines.append(
            f"  {i}. {pred['class']}: {pred['confidence']:.3f} "
            f"[{', '.join(f'{x:.1f}' for x in pred['bbox'])}]"
        )
    
    return '\n'.join(lines)


def calculate_iou(box1: List[float], box2: List[float]) -> float:

    # Get coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def non_max_suppression(
    predictions: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:

    if not predictions:
        return predictions
    
    # Sort by confidence (descending)
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    while sorted_preds:
        # Take the prediction with highest confidence
        best = sorted_preds.pop(0)
        keep.append(best)
        
        # Remove predictions with high IoU
        sorted_preds = [
            pred for pred in sorted_preds
            if calculate_iou(best['bbox'], pred['bbox']) < iou_threshold
        ]
    
    return keep


def resize_image_keep_aspect(
    image: np.ndarray,
    target_size: int = 640
) -> Tuple[np.ndarray, float]:

    h, w = image.shape[:2]
    
    # Calculate scale factor
    scale = target_size / max(h, w)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return resized, scale


def create_letterbox_image(
    image: np.ndarray,
    target_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:

    h, w = image.shape[:2]
    
    # Calculate scale
    scale = min(target_size / h, target_size / w)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    # Add padding
    top, bottom = pad_h, target_size - new_h - pad_h
    left, right = pad_w, target_size - new_w - pad_w
    
    letterboxed = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    
    return letterboxed, (scale, scale), (pad_w, pad_h)


def count_parameters(model) -> int:

    try:
        return sum(p.numel() for p in model.model.parameters())
    except:
        return 0


def get_model_info(weights_path: str) -> Dict:
    model = safe_load_model(weights_path)
    
    info = {
        'path': weights_path,
        'exists': Path(weights_path).exists(),
        'size_mb': Path(weights_path).stat().st_size / (1024 * 1024) if Path(weights_path).exists() else 0,
        'parameters': count_parameters(model),
        'model_type': 'YOLOv8'
    }
    
    return info


def save_predictions_to_json(predictions: List[Dict], output_path: str) -> None:
    import json
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"Predictions saved to {output_path}")


def load_predictions_from_json(json_path: str) -> List[Dict]:
    import json
    
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    
    logger.info(f"Loaded {len(predictions)} predictions from {json_path}")
    
    return predictions
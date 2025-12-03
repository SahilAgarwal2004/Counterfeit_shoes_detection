import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional
import cv2
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_inference(
    weights: str,
    source: Union[str, Path],
    conf_threshold: float = 0.25,
    save_crops: bool = False,
    output_dir: str = "runs/predict",
    save_json: bool = True
) -> List[Dict]:

    # Load model
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    
    logger.info(f"Loading model from {weights}")
    model = YOLO(str(weights_path))
    
    # Prepare source
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    
    # Get list of images
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = list(source_path.glob("*.jpg")) + \
                     list(source_path.glob("*.jpeg")) + \
                     list(source_path.glob("*.png"))
    
    if not image_files:
        raise ValueError(f"No images found in {source}")
    
    logger.info(f"Found {len(image_files)} images")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_crops:
        crops_path = output_path / "crops"
        crops_path.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    all_predictions = []
    
    for img_file in image_files:
        logger.info(f"Processing {img_file.name}...")
        
        # Run inference
        results = model.predict(
            source=str(img_file),
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        # Parse results
        result = results[0]
        predictions = []
        
        for box in result.boxes:
            pred = {
                "class": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            }
            predictions.append(pred)
        
        # Determine final label (highest confidence)
        if predictions:
            best_pred = max(predictions, key=lambda x: x["confidence"])
            final_label = best_pred["class"]
            final_confidence = best_pred["confidence"]
        else:
            final_label = "no_detection"
            final_confidence = 0.0
        
        # Create prediction summary
        pred_summary = {
            "image": str(img_file),
            "predictions": predictions,
            "final_label": final_label,
            "final_confidence": final_confidence
        }
        all_predictions.append(pred_summary)
        
        # Save annotated image
        annotated = result.plot()
        output_img_path = output_path / img_file.name
        cv2.imwrite(str(output_img_path), annotated)
        
        # Save crops if requested
        if save_crops and predictions:
            img = cv2.imread(str(img_file))
            for i, pred in enumerate(predictions):
                x1, y1, x2, y2 = map(int, pred["bbox"])
                crop = img[y1:y2, x1:x2]
                crop_name = f"{img_file.stem}_{i}_{pred['class']}.jpg"
                crop_path = crops_path / crop_name
                cv2.imwrite(str(crop_path), crop)
        
        logger.info(f"  Final: {final_label} (conf: {final_confidence:.3f})")
    
    # Save JSON summary
    if save_json:
        json_path = output_path / "predictions.json"
        with open(json_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        logger.info(f"Predictions saved to {json_path}")
    
    logger.info(f"Output saved to {output_path}")
    return all_predictions


def infer_single_image(
    weights: str,
    image_path: str,
    conf_threshold: float = 0.25
) -> Dict:
    results = run_inference(
        weights=weights,
        source=image_path,
        conf_threshold=conf_threshold,
        save_crops=False,
        output_dir="runs/predict_temp",
        save_json=False
    )
    return results[0] if results else None


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference for counterfeit shoe detection"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image file or directory"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped detections"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/predict",
        help="Output directory (default: runs/predict)"
    )
    
    args = parser.parse_args()
    
    try:
        predictions = run_inference(
            weights=args.weights,
            source=args.source,
            conf_threshold=args.conf,
            save_crops=args.save_crop,
            output_dir=args.output
        )
        logger.info(f"Processed {len(predictions)} images successfully")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
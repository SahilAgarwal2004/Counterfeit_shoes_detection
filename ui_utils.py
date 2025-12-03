import streamlit as st
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np


def display_prediction_card(
    label: str,
    confidence: float,
    is_authentic: bool = True
) -> None:
    """
    Display a styled prediction card in Streamlit.
    
    Args:
        label: Predicted class label
        confidence: Prediction confidence
        is_authentic: Whether the item is authentic (True) or counterfeit (False)
    """
    if is_authentic:
        icon = "✅"
        title = "Authentic Sneaker"
        color = "#d4edda"
        border_color = "#28a745"
    else:
        icon = "⚠️"
        title = "Counterfeit Detected"
        color = "#f8d7da"
        border_color = "#dc3545"
    
    html = f"""
    <div style="
        background-color: {color};
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid {border_color};
        margin: 1rem 0;
    ">
        <h2 style="margin: 0; color: {border_color};">{icon} {title}</h2>
        <p style="margin: 0.5rem 0; font-size: 1.2rem;">
            <strong>Class:</strong> {label.upper()}
        </p>
        <p style="margin: 0.5rem 0; font-size: 1.2rem;">
            <strong>Confidence:</strong> {confidence:.1%}
        </p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def display_metrics_summary(metrics: Dict) -> None:
    """
    Display model evaluation metrics in a formatted grid.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    st.subheader("📊 Model Performance Metrics")
    
    # Detection metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "mAP@0.5",
            f"{metrics.get('mAP50', 0):.3f}",
            help="Mean Average Precision at IoU 0.5"
        )
    
    with col2:
        st.metric(
            "Precision",
            f"{metrics.get('precision', 0):.3f}",
            help="Detection precision"
        )
    
    with col3:
        st.metric(
            "Recall",
            f"{metrics.get('recall', 0):.3f}",
            help="Detection recall"
        )
    
    with col4:
        if 'classification_f1' in metrics:
            st.metric(
                "F1-Score",
                f"{metrics.get('classification_f1', 0):.3f}",
                help="Classification F1 score"
            )
    
    # Classification metrics if available
    if 'classification_accuracy' in metrics:
        st.subheader("Classification Metrics (Image-Level)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{metrics['classification_accuracy']:.3f}")
        
        with col2:
            st.metric("Precision", f"{metrics['classification_precision']:.3f}")
        
        with col3:
            st.metric("Recall", f"{metrics['classification_recall']:.3f}")


def create_download_button(
    data: str,
    filename: str,
    label: str = "Download",
    mime: str = "application/json"
) -> None:
    """
    Create a styled download button.
    
    Args:
        data: Data to download
        filename: Name of the downloaded file
        label: Button label
        mime: MIME type of the data
    """
    st.download_button(
        label=f"📥 {label}",
        data=data,
        file_name=filename,
        mime=mime,
        use_container_width=True
    )


def format_bbox_label(
    class_name: str,
    confidence: float,
    show_confidence: bool = True
) -> str:
    """
    Format bounding box label text.
    
    Args:
        class_name: Predicted class name
        confidence: Prediction confidence
        show_confidence: Whether to include confidence in label
        
    Returns:
        Formatted label string
    """
    if show_confidence:
        return f"{class_name}: {confidence:.2f}"
    return class_name


def draw_styled_boxes(
    image: np.ndarray,
    predictions: List[Dict],
    show_labels: bool = True,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw styled bounding boxes on image with custom colors.
    
    Args:
        image: Input image (RGB format)
        predictions: List of prediction dictionaries
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
        
    Returns:
        Image with drawn bounding boxes
    """
    img_copy = image.copy()
    
    # Define colors for different classes
    class_colors = {
        'real': (0, 255, 0),        # Green
        'fake': (255, 0, 0),        # Red
        'authentic': (0, 255, 0),   # Green
        'counterfeit': (255, 0, 0), # Red
        'genuine': (0, 255, 0),     # Green
    }
    
    for pred in predictions:
        class_name = pred['class']
        confidence = pred['confidence']
        bbox = pred['bbox']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color (RGB format for display)
        color = class_colors.get(class_name.lower(), (255, 255, 0))  # Default yellow
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
        
        # Draw label if requested
        if show_labels:
            label = format_bbox_label(class_name, confidence, show_confidence)
            
            # Calculate label size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(
                img_copy,
                (x1, y1 - label_h - 15),
                (x1 + label_w + 10, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_copy,
                label,
                (x1 + 5, y1 - 8),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
    
    return img_copy


def load_example_images(
    example_dir: str = "data/examples",
    max_images: int = 6
) -> List[Tuple[Path, str]]:
    """
    Load example images for gallery display.
    
    Args:
        example_dir: Directory containing example images
        max_images: Maximum number of images to load
        
    Returns:
        List of (image_path, label) tuples
    """
    example_path = Path(example_dir)
    
    if not example_path.exists():
        return []
    
    examples = []
    
    # Look for images with specific naming conventions
    for img_file in example_path.glob("*.jpg"):
        # Try to infer label from filename
        name = img_file.stem.lower()
        if 'real' in name or 'authentic' in name:
            label = "Authentic"
        elif 'fake' in name or 'counterfeit' in name:
            label = "Counterfeit"
        else:
            label = "Unknown"
        
        examples.append((img_file, label))
        
        if len(examples) >= max_images:
            break
    
    return examples


def create_info_box(
    title: str,
    content: str,
    box_type: str = "info"
) -> None:
    """
    Create a styled information box.
    
    Args:
        title: Box title
        content: Box content
        box_type: Type of box ('info', 'warning', 'success', 'error')
    """
    colors = {
        'info': ('#d1ecf1', '#0c5460', '#bee5eb'),
        'warning': ('#fff3cd', '#856404', '#ffeaa7'),
        'success': ('#d4edda', '#155724', '#c3e6cb'),
        'error': ('#f8d7da', '#721c24', '#f5c6cb')
    }
    
    bg_color, text_color, border_color = colors.get(box_type, colors['info'])
    
    html = f"""
    <div style="
        background-color: {bg_color};
        color: {text_color};
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid {border_color};
        margin: 1rem 0;
    ">
        <h4 style="margin: 0 0 0.5rem 0; color: {text_color};">{title}</h4>
        <p style="margin: 0;">{content}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def display_json_pretty(data: Dict, title: str = "JSON Output") -> None:
    """
    Display JSON data in a collapsible, formatted section.
    
    Args:
        data: Dictionary to display
        title: Section title
    """
    with st.expander(f"📄 {title}"):
        st.json(data)


def create_confidence_bar(confidence: float, threshold: float = 0.5) -> None:
    """
    Create a visual confidence bar.
    
    Args:
        confidence: Confidence value (0-1)
        threshold: Threshold for color coding
    """
    color = "#28a745" if confidence >= threshold else "#dc3545"
    
    html = f"""
    <div style="margin: 1rem 0;">
        <div style="
            background-color: #e9ecef;
            border-radius: 0.5rem;
            height: 30px;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                background-color: {color};
                width: {confidence * 100}%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                transition: width 0.3s ease;
            ">
                {confidence:.1%}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
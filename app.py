import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import json
import time
from PIL import Image
import sys

# Add yolo directory to path
sys.path.append(str(Path(__file__).parent))

from yolo.infer import infer_single_image
from yolo.utils import safe_load_model, draw_bounding_boxes
from ui_utils import (
    display_prediction_card,
    display_metrics_summary,
    create_download_button,
    load_example_images
)

# Page configuration
st.set_page_config(
    page_title="Counterfeit Shoe Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-real {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .prediction-fake {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

def main():
    # Header
    st.markdown('<div class="main-header"> Counterfeit Shoe Detector</div>', unsafe_allow_html=True)
    st.markdown("### AI-powered detection of authentic vs counterfeit sneakers")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_path = st.text_input(
            "Model Weights Path",
            value="models/best.pt",
            help="Path to trained YOLO model weights"
        )
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detection"
        )
        
        # Display options
        st.subheader("Display Options")
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_bbox = st.checkbox("Show bounding boxes", value=True)
        show_json = st.checkbox("Show raw JSON output", value=False)
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This app uses YOLOv8 to detect and classify sneakers as "
            "authentic or counterfeit. Upload an image to get started!"
        )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Example Gallery", "📈 Model Info"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a sneaker image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of a sneaker"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Run inference button
                if st.button("🚀 Detect Counterfeit", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Save uploaded file temporarily
                            temp_path = Path("temp_upload.jpg")
                            image.save(temp_path)
                            
                            # Run inference
                            start_time = time.time()
                            prediction = infer_single_image(
                                weights=model_path,
                                image_path=str(temp_path),
                                conf_threshold=conf_threshold
                            )
                            inference_time = time.time() - start_time
                            
                            # Store prediction
                            st.session_state.predictions.append(prediction)
                            
                            # Display results in col2
                            with col2:
                                st.subheader("Detection Results")
                                
                                # Inference time
                                st.metric("Inference Time", f"{inference_time:.3f}s")
                                
                                # Final prediction
                                final_label = prediction['final_label']
                                final_conf = prediction['final_confidence']
                                
                                if final_label != 'no_detection':
                                    # Determine if real or fake
                                    is_real = final_label.lower() in ['real', 'authentic', 'genuine']
                                    
                                    # Display prediction card
                                    if is_real:
                                        st.markdown(
                                            f'<div class="prediction-real">'
                                            f'<h3>✅ Authentic Sneaker</h3>'
                                            f'<p>Confidence: {final_conf:.1%}</p>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(
                                            f'<div class="prediction-fake">'
                                            f'<h3>⚠️ Counterfeit Detected</h3>'
                                            f'<p>Confidence: {final_conf:.1%}</p>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                                    
                                    # Show annotated image
                                    if show_bbox and prediction['predictions']:
                                        img_array = np.array(image)
                                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                                        annotated = draw_bounding_boxes(img_bgr, prediction['predictions'])
                                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                                        st.image(annotated_rgb, caption="Annotated Image", use_container_width=True)
                                    
                                    # Show all detections
                                    if show_confidence and prediction['predictions']:
                                        st.subheader("All Detections")
                                        for i, pred in enumerate(prediction['predictions'], 1):
                                            st.write(
                                                f"**{i}.** {pred['class'].upper()} - "
                                                f"Confidence: {pred['confidence']:.1%}"
                                            )
                                    
                                    # Show JSON
                                    if show_json:
                                        st.subheader("Raw Output")
                                        st.json(prediction)
                                    
                                    # Download button
                                    json_str = json.dumps(prediction, indent=2)
                                    st.download_button(
                                        label=" Download Prediction JSON",
                                        data=json_str,
                                        file_name="prediction.json",
                                        mime="application/json"
                                    )
                                else:
                                    st.warning("No sneaker detected in image. Try another image.")
                            
                            # Clean up
                            temp_path.unlink(missing_ok=True)
                            
                        except FileNotFoundError:
                            st.error(
                                f"Model weights not found at `{model_path}`. "
                                "Please train a model first or check the path."
                            )
                        except Exception as e:
                            st.error(f"Error during inference: {str(e)}")
            else:
                with col2:
                    st.info("👈 Upload an image to get started")
    
    with tab2:
        st.subheader("Example Predictions")
        
        example_images = load_example_images()
        
        if example_images:
            cols = st.columns(3)
            for i, (img_path, label) in enumerate(example_images):
                with cols[i % 3]:
                    img = Image.open(img_path)
                    st.image(img, caption=f"Example: {label}", use_container_width=True)
        else:
            st.info(
                "No example images available. Run inference on the Detection tab "
                "to populate this gallery."
            )
        
        # Show recent predictions
        if st.session_state.predictions:
            st.subheader("Recent Predictions")
            for i, pred in enumerate(reversed(st.session_state.predictions[-6:])):
                with st.expander(f"Prediction {len(st.session_state.predictions) - i}"):
                    st.json(pred)
    
    with tab3:
        st.subheader("Model Information")
        
        try:
            model = safe_load_model(model_path)
            st.success(f"✅ Model loaded successfully from `{model_path}`")
            
            # Display model info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Type", "YOLOv8")
                st.metric("Task", "Object Detection + Classification")
            
            with col2:
                st.metric("Input Size", "640x640")
                st.metric("Confidence Threshold", f"{conf_threshold:.2f}")
            
        except FileNotFoundError:
            st.error(
                f"❌ Model weights not found at `{model_path}`\n\n"
                "Please train a model first using:\n"
                "```bash\n"
                "python yolo/train.py --data data/roboflow_export/data.yaml --epochs 40\n"
                "```"
            )
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
        
        # Training instructions
        with st.expander("📚 Training Instructions"):
            st.markdown("""
            ### How to Train Your Own Model
            
            1. **Prepare Data**: Place your Roboflow export in `data/`
            
            2. **Train Model**:
            ```bash
            python yolo/train.py --data data/data.yaml --epochs 40 --batch 16
            ```
            
            3. **Evaluate Model**:
            ```bash
            python yolo/evaluate.py --weights models/best.pt --data data/data.yaml
            ```
            
            4. **Run Inference**:
            ```bash
            python yolo/infer.py --weights models/best.pt --source path/to/images/
            ```
            """)

if __name__ == "__main__":
    main()
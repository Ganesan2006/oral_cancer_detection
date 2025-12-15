import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import requests
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Oral Cancer Detection",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [role="tab"] {
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 600;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 2px solid #ddd;
    }
    .high-risk {
        background-color: #fee;
        border-color: #f88;
    }
    .low-risk {
        background-color: #efe;
        border-color: #8f8;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    h2 {
        color: #2c5aa0;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load the trained DenseNet121 model"""
    try:
        # Try loading from local path first
        model = tf.keras.models.load_model('oral_cancer_model.keras')
        return model
    except:
        st.warning("⚠️ Model file 'best_model.keras' not found locally.")
        st.info("📌 Please ensure you've uploaded your trained model file.")
        return None

# ==================== IMAGE PREPROCESSING ====================
def preprocess_image(img):
    """Preprocess image to match model input requirements"""
    # Convert to RGB if image is grayscale or has alpha channel
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to 224x224
    img = img.resize((224, 224))
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Normalize to 0-1 range
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
# ==================== PREDICTION FUNCTION ====================
def make_prediction(img_array, model):
    """Make prediction using the model"""
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    # Binary classification: 0 = Non-Cancerous, 1 = Cancerous
    is_cancerous = confidence > 0.5
    return {
        'confidence': confidence,
        'is_cancerous': is_cancerous,
        'class_name': 'Cancerous 🚨' if is_cancerous else 'Non-Cancerous ✅',
        'risk_level': 'High Risk' if is_cancerous else 'Low Risk'
    }

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown("# 🦷 Oral Cancer Detection System")
    st.markdown("AI-powered detection using Deep Learning (DenseNet121)")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model failed to load. Please check the model file path.")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Predict", "ℹ️ About Model", "📊 Metrics", "❓ FAQ"])
    
    # ==================== TAB 1: PREDICT ====================
    with tab1:
        st.subheader("Upload Oral Image for Analysis")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### 📁 Input Methods")
            
            upload_method = st.radio(
                "Choose input method:",
                ["Upload Image", "Use Sample Image"]
            )
            
            uploaded_file = None
            sample_img = None
            
            if upload_method == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Choose an oral image (JPG, PNG)",
                    type=['jpg', 'jpeg', 'png'],
                    help="Supported formats: JPG, PNG"
                )
            else:
                st.markdown("""
                **Sample image URLs** (oral cavity images):
                - Replace with your actual sample image URL if available
                """)
                sample_url = st.text_input(
                    "Enter image URL:",
                    value="",
                    help="Paste a public image URL"
                )
                if sample_url:
                    try:
                        response = requests.get(sample_url, timeout=5)
                        sample_img = Image.open(io.BytesIO(response.content))
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
            
            # Process uploaded file
            img_to_process = None
            if uploaded_file is not None:
                img_to_process = Image.open(uploaded_file)
            elif sample_img is not None:
                img_to_process = sample_img
            
            # Display uploaded image
            if img_to_process is not None:
                st.markdown("### 📷 Preview")
                st.image(img_to_process, use_column_width=True, caption="Uploaded Image")
        
        with col2:
            st.markdown("### 🔍 Prediction Results")
            
            if img_to_process is not None:
                # Show processing spinner
                with st.spinner("🔄 Analyzing image..."):
                    # Preprocess image
                    img_array = preprocess_image(img_to_process)
                    
                    # Make prediction
                    result = make_prediction(img_array, model)
                
                # Display prediction results
                confidence_pct = result['confidence'] * 100
                
                # Color-coded prediction box
                box_class = "high-risk" if result['is_cancerous'] else "low-risk"
                
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h3>🎯 Diagnosis</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">
                        {result['class_name']}
                    </p>
                    <p style="color: #666; margin: 5px 0;">
                        Risk Level: <strong>{result['risk_level']}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence visualization
                st.markdown("### 📊 Confidence Score")
                
                col_metric1, col_metric2 = st.columns(2)
                
                with col_metric1:
                    st.metric(
                        "Cancerous Probability",
                        f"{confidence_pct:.2f}%",
                        delta=None
                    )
                
                with col_metric2:
                    st.metric(
                        "Non-Cancerous Probability",
                        f"{(100-confidence_pct):.2f}%",
                        delta=None
                    )
                
                # Confidence bar
                st.markdown("#### Confidence Distribution")
                
                col_bar1, col_bar2 = st.columns([confidence_pct/100, (100-confidence_pct)/100])
                
                with col_bar1:
                    st.markdown(f'<div style="background-color:#ff6b6b; height:40px; border-radius:5px; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">{confidence_pct:.1f}%</div>', unsafe_allow_html=True)
                
                with col_bar2:
                    st.markdown(f'<div style="background-color:#51cf66; height:40px; border-radius:5px; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">{100-confidence_pct:.1f}%</div>', unsafe_allow_html=True)
                
                # Timestamp
                st.markdown(f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Important note
                st.warning("""
                ⚠️ **IMPORTANT DISCLAIMER**
                
                This tool is for informational purposes only and **NOT a substitute for professional medical diagnosis**. 
                Please consult with a qualified healthcare professional for proper medical evaluation and treatment.
                """)
            else:
                st.info("👆 Please upload an image or provide a URL to get started")
    
    # ==================== TAB 2: MODEL INFO ====================
    with tab2:
        st.subheader("Model Architecture & Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏗️ Architecture")
            st.markdown("""
            **Base Model:** DenseNet121
            - Pre-trained on ImageNet
            - Transfer Learning applied
            - Frozen base layers (fine-tuning)
            
            **Custom Top Layers:**
            - Global Average Pooling 2D
            - Dense Layer (256 units)
            - LeakyReLU Activation
            - Dropout (0.5)
            - Output Dense Layer (1 unit, Sigmoid)
            """)
        
        with col2:
            st.markdown("### 📈 Training Configuration")
            st.markdown("""
            **Input Size:** 224 × 224 × 3 (RGB)
            
            **Training Parameters:**
            - Optimizer: Adam (lr=1e-3)
            - Loss Function: Binary Crossentropy
            - Batch Size: 32
            - Epochs: 10
            
            **Model Size:**
            - Total Parameters: 7.3M
            - Trainable Parameters: 262K
            """)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔄 Data Augmentation")
            st.markdown("""
            - Rotation: 20°
            - Width Shift: 10%
            - Height Shift: 10%
            - Shear Range: 10%
            - Zoom Range: 10%
            - Horizontal Flip: Yes
            """)
        
        with col2:
            st.markdown("### 📊 Dataset Split")
            st.markdown("""
            **Training:** 4,008 images
            
            **Validation:** 395 images
            
            **Testing:** 206 images
            
            **Total:** 4,609 images
            
            **Classes:** Binary (2)
            """)
        
        with col3:
            st.markdown("### 🎯 Callbacks Used")
            st.markdown("""
            1. **Early Stopping**
               - Monitor: val_loss
               - Patience: 7 epochs
            
            2. **LR Reduction**
               - Monitor: val_accuracy
               - Factor: 0.5
            
            3. **Model Checkpoint**
               - Save best model
            """)
    
    # ==================== TAB 3: METRICS ====================
    with tab3:
        st.subheader("Model Performance Metrics")
        
        st.markdown("""
        ### Performance Summary
        
        This DenseNet121 model trained on oral cancer dataset demonstrates:
        
        ✅ **Binary Classification Task**
        - Class 0: Non-Cancerous tissue
        - Class 1: Cancerous tissue
        
        ✅ **Transfer Learning Approach**
        - Leverages pre-trained ImageNet weights
        - Minimal parameters to train
        - Fast convergence
        
        ✅ **Robust Training**
        - Data augmentation for generalization
        - Early stopping to prevent overfitting
        - Learning rate reduction for fine-tuning
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">📊 Total Images<br><strong>4,609</strong></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">🎯 Train/Val Split<br><strong>91%/9%</strong></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">🧠 Model Size<br><strong>~27.85 MB</strong></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">⚡ Inference<br><strong>Fast</strong></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🔍 Evaluation Notes")
        st.markdown("""
        To view detailed performance metrics (accuracy, precision, recall, F1-score):
        1. Run your Jupyter notebook with test data
        2. Check confusion matrix and classification report
        3. Evaluate on holdout test set (206 images)
        
        **Next Steps for Production:**
        - Validate model on external dataset
        - Monitor prediction confidence distribution
        - Implement prediction logging
        - Set up confidence threshold alerts
        """)
    
    # ==================== TAB 4: FAQ ====================
    with tab4:
        st.subheader("Frequently Asked Questions")
        
        faqs = [
            {
                "q": "How do I deploy this app?",
                "a": """
                **Streamlit Cloud Deployment:**
                1. Push code to GitHub
                2. Go to share.streamlit.io
                3. Connect your GitHub repo
                4. Select main branch and app.py
                
                **Alternative Hosting:**
                - Heroku
                - AWS
                - Google Cloud
                - Azure
                """
            },
            {
                "q": "What image format should I use?",
                "a": "JPG or PNG format. The model expects 224×224 RGB images. Larger images will be resized automatically."
            },
            {
                "q": "How accurate is the model?",
                "a": "The model is trained on 4,609 oral cavity images. For accuracy metrics, check your test set evaluation. Always consult a medical professional for diagnosis."
            },
            {
                "q": "Can I improve the model?",
                "a": """
                **Improvements:**
                - Use more training data
                - Try deeper architectures (ResNet, EfficientNet)
                - Ensemble multiple models
                - Fine-tune all layers
                - Experiment with hyperparameters
                """
            },
            {
                "q": "How do I save predictions?",
                "a": "Add logging to save predictions to CSV/database. Modify app.py to store results with timestamp and confidence scores."
            },
            {
                "q": "What's the inference time?",
                "a": "Typically 0.2-0.5 seconds per image on CPU. Much faster on GPU acceleration."
            }
        ]
        
        for idx, faq in enumerate(faqs, 1):
            with st.expander(f"❓ {faq['q']}"):
                st.markdown(faq['a'])
        
        st.markdown("---")
        
        st.markdown("""
        ### 📚 Additional Resources
        
        **Model Architecture:**
        - [DenseNet Paper](https://arxiv.org/abs/1608.06993)
        - [Keras DenseNet121](https://keras.io/api/applications/densenet/)
        
        **Deployment:**
        - [Streamlit Docs](https://docs.streamlit.io)
        - [Streamlit Cloud](https://streamlit.io/cloud)
        
        **Medical Imaging:**
        - Best practices for medical AI
        - Regulatory compliance (HIPAA, etc.)
        - Clinical validation
        """)

if __name__ == "__main__":
    main()

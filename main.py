import os
import json
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("plant_disease_model_final.h5")
    with open("class_indices_v2.json", "r") as f:
        class_indices = json.load(f)
    return model, class_indices

model, class_indices = load_model_and_classes()

# ====================== REMEDIES ======================
remedies = {
    "Pepper__bell___Bacterial_spot": [
        "Apply copper-based bactericide every 7 days",
        "Avoid overhead watering, use drip irrigation",
        "Rotate crops - wait 3 years before planting peppers again",
        "Remove and destroy heavily infected plants"
    ],
    "Pepper__bell___healthy": ["🌱 Your pepper plant looks healthy! Keep up the good care."],
    
    "Potato___Early_blight": [
        "Remove infected lower leaves immediately",
        "Apply mancozeb or chlorothalonil fungicide",
        "Mulch around plants to prevent soil splash",
        "Rotate crops every 2–3 years"
    ],
    "Potato___Late_blight": [
        "Destroy infected plants right away (do not compost)",
        "Apply copper-based fungicide quickly",
        "Never wet the leaves - use drip irrigation only",
        "Choose late-blight resistant varieties next season"
    ],
    "Potato___healthy": ["🌱 Your potato plant is healthy! Continue regular care."],
    
    "Tomato_Bacterial_spot": [
        "Apply copper bactericide regularly",
        "Rotate crops (avoid tomato family for 3 years)",
        "Remove infected leaves and plants promptly",
        "Use disease-resistant tomato varieties"
    ],
    "Tomato_Early_blight": [
        "Remove bottom infected leaves early",
        "Apply mancozeb fungicide",
        "Mulch base to stop soil splash",
        "Practice crop rotation"
    ],
    "Tomato_Late_blight": [
        "Remove and destroy infected plants immediately",
        "Apply copper or Ridomil fungicide",
        "Avoid overhead watering completely",
        "Plant resistant varieties in future"
    ],
    "Tomato_Leaf_Mold": [
        "Improve ventilation and reduce humidity",
        "Apply fungicide if infection spreads",
        "Remove lower infected leaves",
        "Space plants properly for air flow"
    ],
    "Tomato_Septoria_leaf_spot": [
        "Remove infected leaves as soon as seen",
        "Apply fungicide preventively",
        "Mulch to prevent soil splash",
        "Rotate crops regularly"
    ]
}

# ====================== IMAGE PREPROCESSING ======================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((192, 192))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ====================== CUSTOM CSS ======================
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        color: #2e8b57;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #e6ffe6, #f0fff0);
        border-left: 6px solid #2e8b57;
        margin: 15px 0;
    }
    .remedy-box {
        background-color: #f8fff8;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2e8b57;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("data:image/jpeg;base64,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", width=80)
    st.title("Plant Doctor")
    st.markdown("### AI Disease Detection")
    st.write("Upload a clear photo of a leaf")
    
    st.divider()
    

# ====================== MAIN UI ======================
st.markdown('<div class="main-title">🌿 Plant Disease Detector</div>', unsafe_allow_html=True)
st.markdown("### Upload a leaf image to get instant diagnosis & treatment advice")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("📤 Choose a leaf image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

with col2:
    st.write("")
    camera_file = st.camera_input("📸 Or take a photo")

# Use camera if available, else uploaded file
if camera_file is not None:
    image_file = camera_file
    display_image = Image.open(camera_file)
elif uploaded_file is not None:
    image_file = uploaded_file
    display_image = Image.open(uploaded_file)
else:
    image_file = None

if image_file is not None:
    st.image(display_image.resize((400, 400)), caption="Uploaded Leaf", use_container_width=False)
    
    if st.button("🔍 Analyze Disease", type="primary", use_container_width=True):
        with st.spinner("Analyzing with AI model..."):
            img_array = preprocess_image(image_file.getvalue())
            predictions = model.predict(img_array, verbose=0)[0]
            
            pred_index = np.argmax(predictions)
            predicted_class = class_indices[str(pred_index)]
            confidence = float(predictions[pred_index]) * 100
            
            # Main Result
            st.markdown(f"""
                <div class="prediction-box">
                    <h3>Prediction: {predicted_class.replace('_', ' ')}</h3>
                    <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Remedies Section
            st.subheader("🛠️ Recommended Treatment")
            st.markdown('<div class="remedy-box">', unsafe_allow_html=True)
            
            if predicted_class in remedies:
                for tip in remedies[predicted_class]:
                    st.write(f"• {tip}")
            else:
                st.write("• Consult your local agriculture officer for specific advice.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Extra Tip
            st.info("💡 **Pro Tip**: Always isolate infected plants and sterilize your tools to prevent spread.")

# Footer
st.divider()
st.caption("🌱 AI-powered plant disease detection")
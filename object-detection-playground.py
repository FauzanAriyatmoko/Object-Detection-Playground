from inference_sdk import InferenceHTTPClient
import streamlit as st
import tempfile
import os
import base64

st.title("🔍 Object Detection Playground")

# SIDEBAR: Always visible detection parameters
st.sidebar.header("Detection Parameters")

# Class filter input
class_filter = st.sidebar.text_input(
    "Class Filter (For example: car, person, or a comma-separated list of classes. Leave blank to select all classes.)"
)

# Model selection dropdown
model_options = ["rfdetr-base", "yolov11n-640"]
selected_model = st.sidebar.selectbox("Select Model", options=model_options, index=0)

# Label and bounding box color palettes
color_palette_options = [
    "ROBOFLOW",
    "Matplotlib Viridis",
    "Matplotlib Plasma",
    "Matplotlib Inferno",
    "Matplotlib Magma",
    "Matplotlib Cividis",
    "Matplotlib Pastel1",
    "Matplotlib Pastel2",
    "Matplotlib Set1",
    "Matplotlib Set2",
    "Matplotlib Set3",
    "Matplotlib Accent",
    "Matplotlib Dark2",
    "Matplotlib Paired",
    "Matplotlib Tab10",
    "Matplotlib Tab20",
    "Matplotlib Tab20b",
    "Matplotlib Tab20c"
]

label_color_palette = st.sidebar.selectbox(
    "Label Color Palette",
    options=color_palette_options,
    index=6  # default to "Matplotlib Pastel1"
)

bounding_box_color_palette = st.sidebar.selectbox(
    "Bounding Box Color Palette",
    options=color_palette_options,
    index=0  # default to "ROBOFLOW"
)

# Numeric parameters
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
bbox_thickness = st.sidebar.slider("Bounding Box Thickness", 1, 50, 15)
text_scale = st.sidebar.slider("Text Scale", 1, 20, 8)
text_thickness = st.sidebar.slider("Text Thickness", 1, 30, 4)

# Text Position Dropdown
text_position_options = [
    "TOP_LEFT",
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "TOP_CENTER",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_CENTER",
    "BOTTOM_RIGHT",
]

text_position = st.sidebar.selectbox(
    "Text Position",
    options=text_position_options,
    index=0  # default to "TOP_LEFT"
)

# Text Color Dropdown
text_color_options = [
    "Black",
    "White",
    "Blue",
]

text_color = st.sidebar.selectbox(
    "Text Color",
    options=text_color_options,
    index=0  # default to "Black"
)

# MAIN CONTENT: File uploader at the top
uploaded_file = st.file_uploader(
    "Drag & drop an image or click to upload",
    type=["jpg", "jpeg", "png"],
    key="file_uploader"
)

# Initialize session state for uploaded image and detected image
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "detected_image" not in st.session_state:
    st.session_state.detected_image = None

# Update session state only if a new file is uploaded
if uploaded_file is not None and uploaded_file != st.session_state.uploaded_image:
    st.session_state.uploaded_image = uploaded_file
    st.session_state.detected_image = None

# Load API key from streamlit secrets
api_key = st.secrets["roboflow_api_key"]

if not api_key:
    raise ValueError("❌ ROBOFLOW_KEY not set in streamlit secrets")

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# MAIN CONTENT: Input and Output Columns
col_input, col_output = st.columns(2)

# INPUT COLUMN
with col_input:
    st.header("Input")

    if st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", width='stretch')

        # Initialize running state
        if "is_running" not in st.session_state:
            st.session_state.is_running = False

        # Run Object Detection button
        button_placeholder = st.empty()
        
        if not st.session_state.is_running:
            if button_placeholder.button("Run Object Detection", key="run_detection_btn"):
                st.session_state.is_running = True
                st.rerun()
        
        if st.session_state.is_running:
            button_placeholder.text("Running object detection, please wait...")
            
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(st.session_state.uploaded_image.read())
                tmp_path = tmp.name

            # Run Roboflow workflow
            result = client.run_workflow(
                workspace_name="dikshants-blog-workspace",
                workflow_id="object-detection-playground-workflow",
                images={"image": tmp_path},
                parameters={
                    "model": selected_model,
                    "label_color_palette": label_color_palette,
                    "bounding_box_color_palette": bounding_box_color_palette,
                    "text_thickness": text_thickness,
                    "text_scale": text_scale,
                    "text_color": text_color,
                    "confidence": confidence,
                    "bounding_box_thickness": bbox_thickness,
                    "class_filter": [cls.strip() for cls in class_filter.split(',')] if class_filter else [],
                    "text_position": text_position,
                },
                use_cache=True
            )

            # Decode detected image and store in session state
            base64_data = result[0]['label_visualization']
            st.session_state.detected_image = base64.b64decode(base64_data)

            # Cleanup temp file
            os.remove(tmp_path)
            
            # Reset running state and show button again
            st.session_state.is_running = False
            st.rerun()

# OUTPUT COLUMN
with col_output:
    st.header("Output")
    if st.session_state.detected_image:
        st.image(st.session_state.detected_image, caption="Detected Objects", width='stretch')

        # Optional download button
        st.download_button(
            label="Download Detected Image",
            data=st.session_state.detected_image,
            file_name="object_detected.jpg",
            mime="image/jpeg"
        )
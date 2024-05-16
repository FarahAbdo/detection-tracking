from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")
st.sidebar.markdown("### Developed by Farah Abdou")

# Sidebar

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")

# Upload an image for detection
source_img = st.sidebar.file_uploader(
    "Upload an image for detection", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

try:
    if source_img is None:
        default_image_path = str(settings.DEFAULT_IMAGE)
        default_image = PIL.Image.open(default_image_path)
        st.image(default_image_path, caption="Default Image",
                 use_column_width=True)
    else:
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img, caption="Uploaded Image",
                 use_column_width=True)
except Exception as ex:
    st.error("Error occurred while opening the image.")
    st.error(ex)

if source_img is not None and st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    st.image(res_plotted, caption='Detected Image',
             use_column_width=True)
    try:
        with st.expander("Detection Results"):
            for box in boxes:
                st.write(box.data)
    except Exception as ex:
        st.write("Error displaying detection results.")
        st.write(ex)

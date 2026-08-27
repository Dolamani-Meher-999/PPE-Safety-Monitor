import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PPE Safety Monitor",
    page_icon="🦺",
    layout="wide"
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = (
    r"runs\detect\runs\detect\helmet_10epochs-2"
    r"\weights\best.pt"
)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🦺 PPE Safety Monitor")

st.write(
    "Real-time helmet and safety violation detection "
    "using YOLO11 and computer vision."
)

st.divider()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.header("📷 Analyze an Image")

uploaded_file = st.file_uploader(
    "Upload a traffic image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )

    # Run YOLO
    results = model(
        image,
        conf=0.40,
        verbose=False
    )

    # Annotated image
    annotated_image = results[0].plot()

    st.subheader("Detection Result")

    st.image(
        annotated_image,
        channels="BGR",
        use_container_width=True
    )

    # --------------------------------------------------
    # DETECTION SUMMARY
    # --------------------------------------------------

    helmet_count = 0
    no_helmet_count = 0
    rider_count = 0

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name == "Helmet":
            helmet_count += 1

        elif class_name == "NoHelmet":
            no_helmet_count += 1

        elif class_name == "BikeWithRider":
            rider_count += 1

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🏍️ Riders",
            rider_count
        )

    with col2:
        st.metric(
            "⛑️ Helmets",
            helmet_count
        )

    with col3:
        st.metric(
            "⚠️ No Helmet",
            no_helmet_count
        )

    # --------------------------------------------------
    # SAFETY STATUS
    # --------------------------------------------------

    if no_helmet_count > 0:

        st.error(
            "🚨 SAFETY VIOLATION — NO HELMET DETECTED"
        )

    else:

        st.success(
            "✅ SAFETY OK"
        )

else:

    st.info(
        "Upload an image to start helmet detection."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PPE Safety Monitor | YOLO11 + OpenCV + Streamlit"
)
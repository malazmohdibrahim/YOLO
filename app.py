import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="YOLO Object Counter",
    page_icon="🔎",
    layout="wide"
)

@st.cache_resource
def load_model():
    # YOLOv8 nano is lightweight and suitable for a simple Streamlit application.
    return YOLO("yolov8n.pt")

st.title("🔎 YOLO Object Detection & Object Counter")
st.write(
    "Upload an image and the application will detect people and objects, "
    "count each object type, and calculate useful statistics."
)

with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Higher values show only more confident detections."
    )

    st.info(
        "The first run downloads the YOLOv8n model automatically. "
        "Internet access is required the first time."
    )

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is None:
    st.markdown("### How it works")
    st.markdown(
        "1. Upload an image.\n"
        "2. YOLO detects people and objects.\n"
        "3. The application counts every detected object.\n"
        "4. It calculates totals, object-type counts, percentages, and people statistics."
    )
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

with st.spinner("Loading YOLO model..."):
    model = load_model()

with st.spinner("Detecting objects..."):
    results = model.predict(
        source=np.array(image),
        conf=confidence,
        verbose=False
    )

result = results[0]

# Plot detections on the image.
annotated = result.plot()
annotated_image = Image.fromarray(annotated[..., ::-1])

# Extract detections.
detections = []
names = result.names

if result.boxes is not None:
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = names[class_id]

        xyxy = box.xyxy[0].tolist()
        detections.append({
            "Object Type": class_name,
            "Confidence": conf,
            "X1": round(xyxy[0], 1),
            "Y1": round(xyxy[1], 1),
            "X2": round(xyxy[2], 1),
            "Y2": round(xyxy[3], 1)
        })

df = pd.DataFrame(detections)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

with col2:
    st.subheader("Detected Objects")
    st.image(annotated_image, use_container_width=True)

st.divider()

total_objects = len(df)
people_count = int((df["Object Type"] == "person").sum()) if not df.empty else 0
object_types = int(df["Object Type"].nunique()) if not df.empty else 0

m1, m2, m3 = st.columns(3)
m1.metric("Total Objects", total_objects)
m2.metric("People Detected", people_count)
m3.metric("Different Object Types", object_types)

if df.empty:
    st.warning(
        "No objects were detected. Try lowering the confidence threshold "
        "or uploading another image."
    )
    st.stop()

# Count each object category.
counts = (
    df["Object Type"]
    .value_counts()
    .rename_axis("Object Type")
    .reset_index(name="Count")
)

counts["Percentage of Total"] = (
    counts["Count"] / total_objects * 100
).round(2)

st.subheader("📊 Object Type Analysis")
st.dataframe(
    counts,
    use_container_width=True,
    hide_index=True
)

# Additional calculations.
most_common = counts.iloc[0]["Object Type"]
most_common_count = int(counts.iloc[0]["Count"])

person_percentage = round((people_count / total_objects) * 100, 2)

c1, c2, c3 = st.columns(3)
c1.metric("Most Common Type", most_common)
c2.metric("Most Common Count", most_common_count)
c3.metric("People Percentage", f"{person_percentage}%")

st.subheader("🔢 Detailed Detection Data")

display_df = df[["Object Type", "Confidence"]].copy()
display_df["Confidence"] = (display_df["Confidence"] * 100).round(2).astype(str) + "%"

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# CSV download.
csv_data = counts.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Object Count Report",
    data=csv_data,
    file_name="object_detection_report.csv",
    mime="text/csv"
)

st.caption(
    "Detection model: YOLOv8n (Ultralytics). "
    "Counts represent detected bounding boxes, not necessarily unique physical objects."
)

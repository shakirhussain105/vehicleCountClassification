import streamlit as st
from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import deque

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Vehicle Counting & Classification", layout="wide")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "best.pt"  # your trained model

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
class_names = model.names

# =====================================
# PROCESS VIDEO WITH TRACKING
# =====================================
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    line_y = int(h * 0.55)
    counts = {name: 0 for name in class_names.values()}

    # tracker storage: {track_id: (cx, cy, counted)}
    trackers = {}
    max_disappeared = 5  # frames before forgetting a tracker
    disappeared = {}

    track_id_counter = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.25, iou=0.5, verbose=False)

        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss  = results[0].boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, clss):
                cls_id = int(cls)
                if cls_id not in class_names:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, cls_id, (x1, y1, x2, y2)))

        # simple distance-based tracker
        new_trackers = {}
        for det in detections:
            cx, cy, cls_id, box = det
            matched = False
            for tid, (tcx, tcy, tcls, counted) in trackers.items():
                if tcls == cls_id and np.hypot(cx - tcx, cy - tcy) < 50:
                    new_trackers[tid] = (cx, cy, cls_id, counted)
                    matched = True
                    if not counted and abs(cy - line_y) < 4:
                        counts[class_names[cls_id]] += 1
                        new_trackers[tid] = (cx, cy, cls_id, True)
                    break
            if not matched:
                track_id_counter += 1
                counted = abs(cy - line_y) < 4
                if counted:
                    counts[class_names[cls_id]] += 1
                new_trackers[track_id_counter] = (cx, cy, cls_id, counted)

        trackers = new_trackers

        # draw
        cv2.line(frame, (50, line_y), (w-50, line_y), (0,0,255), 3)
        for cx, cy, cls_id, counted in trackers.values():
            cv2.circle(frame, (cx, cy), 4, (255,0,0), -1)
        for det in detections:
            x1, y1, x2, y2 = det[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        y_text = 30
        for k, v in counts.items():
            cv2.putText(frame, f"{k}: {v}", (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            y_text += 25

        out.write(frame)
        frame_no += 1
        progress.progress(frame_no / total_frames)

    cap.release()
    out.release()
    progress.empty()
    return counts

# =====================================
# UI
# =====================================
st.title("🚗 Vehicle Counting & Classification System")

video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

if video:
    in_path = os.path.join(UPLOAD_FOLDER, video.name)
    out_path = os.path.join(OUTPUT_FOLDER, "processed_" + video.name)

    with open(in_path, "wb") as f:
        f.write(video.read())

    if st.button("▶ Process Video"):
        with st.spinner("Processing video..."):
            counts = process_video(in_path, out_path)

        st.success("Processing complete ✅")
       # st.video(out_path)

        st.subheader("📊 Vehicle Counts")
        for k, v in counts.items():
            st.write(f"**{k}**: {v}")

        with open(out_path, "rb") as f:
            st.download_button(
                "⬇ Download Result Video",
                f,
                file_name="vehicle_counted.mp4",
                mime="video/mp4"
            )

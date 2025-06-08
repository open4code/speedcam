import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# --- Konfiguration ---
MODEL_PATH = 'yolov8n.pt' # Das kleinste YOLOv8 Modell

# --- Hilfsfunktionen ---

@st.cache_resource
def load_yolo_model():
    """Lädt das YOLOv8-Modell und speichert es im Cache."""
    model = YOLO(MODEL_PATH)
    return model

def calculate_distance(p1, p2):
    """Berechnet den euklidischen Abstand zwischen zwei Punkten."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_centroid(bbox):
    """Berechnet den Mittelpunkt (x_center, y_center) einer Bounding Box."""
    x1, y1, x2, y2 = map(int, bbox)
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# --- Streamlit App ---

st.set_page_config(page_title="Geschwindigkeitsschätzung mit YOLOv8", layout="wide")

st.title("Geschwindigkeitsschätzung mit YOLOv8 (Streamlit)")
st.markdown("""
Diese App schätzt die Geschwindigkeit von Objekten in einem Video mit Ultralytics YOLOv8.
**Hinweis:** Die Geschwindigkeitsberechnung ist eine Näherung und hängt stark von der "Pixel pro Meter"-Kalibrierung ab.
Für genaue Ergebnisse sind eine Kamerakalibrierung und eine fortgeschrittenere Tracking-Methode erforderlich.
""")

st.sidebar.header("Einstellungen")
uploaded_file = st.sidebar.file_uploader("Video hochladen (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
confidence_threshold = st.sidebar.slider("Konfidenzschwelle für YOLO-Erkennung", 0.1, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IOU-Schwelle für NMS", 0.1, 1.0, 0.7, 0.05)
# Kalibrierung für Geschwindigkeit
pixels_per_meter = st.sidebar.slider("Pixel pro Meter (Kalibrierung)", 10, 500, 100, 10)


if uploaded_file is not None:
    st.video(uploaded_file)

    st.info("Video wird verarbeitet... Dies kann je nach Videolänge und Modellgröße einige Zeit dauern.")

    # Temporäre Datei speichern
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(uploaded_file.read())
        video_path = temp_video_file.name

    model = load_yolo_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Fehler beim Öffnen des Videos.")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Dictionary zum Speichern der Tracking-Informationen
    # {'object_id': {'last_centroid': (x,y), 'last_frame_idx': int}}
    tracked_objects = {}
    next_object_id = 0

    # Liste zum Speichern der verarbeiteten Frames
    processed_frames = []

    frame_idx = 0
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # KORRIGIERT: CAP_PROP_FRAME_COUNT statt CAP_PROP_COUNT

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence_threshold, iou=iou_threshold, verbose=False) # verbose=False für weniger Konsolen-Output

        current_centroids = []
        current_bboxes = []
        current_classes = []

        # Extrahieren der Detektionen
        for r in results:
            boxes = r.boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf > confidence_threshold:
                    current_centroids.append(get_centroid(bbox))
                    current_bboxes.append(bbox)
                    current_classes.append(cls)

        # Einfache Tracking-Logik (nahegelegenster Mittelpunkt)
        new_tracked_objects = {}
        matched_current_indices = set()

        for obj_id, obj_info in tracked_objects.items():
            last_centroid = obj_info['last_centroid']
            min_dist = float('inf')
            best_match_idx = -1

            for i, current_centroid in enumerate(current_centroids):
                if i in matched_current_indices: # Bereits zugeordnet
                    continue
                dist = calculate_distance(last_centroid, current_centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            
            # Schwellenwert für Matching: Wenn Distanz zu groß, als neues Objekt ansehen
            if best_match_idx != -1 and min_dist < 100: # 100 Pixel als Beispiel-Schwellenwert
                matched_current_indices.add(best_match_idx)
                current_centroid = current_centroids[best_match_idx]
                current_bbox = current_bboxes[best_match_idx]
                current_cls = current_classes[best_match_idx]

                # Geschwindigkeit berechnen (Pixel/Frame)
                distance_pixels = calculate_distance(last_centroid, current_centroid)
                time_diff_frames = frame_idx - obj_info['last_frame_idx']
                
                speed_pixels_per_frame = 0
                if time_diff_frames > 0:
                    speed_pixels_per_frame = distance_pixels / time_diff_frames
                
                # Geschwindigkeit in Meter/Sekunde umrechnen
                speed_meters_per_sec = (speed_pixels_per_frame / pixels_per_meter) * fps # Hier 'fps' zur Umrechnung von pro Frame zu pro Sekunde

                new_tracked_objects[obj_id] = {
                    'last_centroid': current_centroid,
                    'last_frame_idx': frame_idx,
                    'speed_mps': speed_meters_per_sec,
                    'bbox': current_bbox,
                    'class': current_cls
                }
            else:
                # Objekt nicht gefunden oder zu weit entfernt -> verschwinden lassen
                pass 
        
        # Neue Objekte hinzufügen
        for i, current_centroid in enumerate(current_centroids):
            if i not in matched_current_indices:
                new_tracked_objects[next_object_id] = {
                    'last_centroid': current_centroid,
                    'last_frame_idx': frame_idx,
                    'speed_mps': 0, # Initialgeschwindigkeit 0
                    'bbox': current_bboxes[i],
                    'class': current_classes[i]
                }
                next_object_id += 1
        
        tracked_objects = new_tracked_objects

        # Bounding Boxes und Text auf dem Frame zeichnen
        for obj_id, obj_info in tracked_objects.items():
            bbox = obj_info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            centroid_x, centroid_y = obj_info['last_centroid']
            speed = obj_info['speed_mps']
            cls = obj_info['class']

            # Klasse aus dem Modell extrahieren (Annahme: Modell hat eine names-Property)
            class_name = model.names[cls] if hasattr(model, 'names') else f"Class {cls}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Text über der Bounding Box
            text = f"ID: {obj_id} {class_name}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Geschwindigkeitstext unter der Bounding Box
            speed_text = f"Speed: {speed:.2f} m/s"
            cv2.putText(frame, speed_text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # Frame für Streamlit speichern
        processed_frames.append(frame)

        frame_idx += 1
        progress_bar.progress(min((frame_idx / total_frames), 1.0))

    cap.release()
    os.unlink(video_path) # Temporäre Datei löschen

    st.success("Verarbeitung abgeschlossen!")

    if processed_frames:
        # Erstelle ein temporäres Video aus den verarbeiteten Frames
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec für MP4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        for frame in processed_frames:
            out.write(frame)
        out.release()

        st.subheader("Verarbeitetes Video")
        st.video(output_video_path)
        os.unlink(output_video_path) # Temporäre Ausgabedatei löschen
    else:
        st.warning("Keine Frames verarbeitet oder keine Detektionen gefunden.")

else:
    st.warning("Bitte laden Sie ein Video hoch, um zu beginnen.")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Deine App")

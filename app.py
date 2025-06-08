import os
from collections import defaultdict, deque

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO # Importiere YOLO von ultralytics
import supervision as sv

# --- Konfiguration für Perspektivtransformation (Bleibt gleich) ---
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# --- Streamlit App-Konfiguration ---
st.title("Fahrzeuggeschwindigkeitserkennung")

# Schwellenwerte für Konfidenz und IOU
confidence_threshold = st.sidebar.slider(
    "Konfidenz-Schwellenwert", min_value=0.0, max_value=1.0, value=0.3, step=0.05
)
iou_threshold = st.sidebar.slider(
    "IOU-Schwellenwert", min_value=0.0, max_value=1.0, value=0.7, step=0.05
)

# --- Lokales Modell laden ---
# WICHTIG: Stelle sicher, dass diese Datei (z.B. best.pt) im selben Verzeichnis wie deine app.py liegt
LOCAL_MODEL_PATH = "best.pt"

model = None
if not os.path.exists(LOCAL_MODEL_PATH):
    st.error(f"Fehler: Modell-Datei '{LOCAL_MODEL_PATH}' nicht gefunden.")
    st.info("Bitte lade dein Roboflow-Modell (z.B. 'best.pt' als PyTorch-Export) herunter und platziere es im selben Verzeichnis wie diese App-Datei.")
    st.stop() # Stoppt die Ausführung der App, da das Modell fehlt
else:
    try:
        model = YOLO(LOCAL_MODEL_PATH)
        st.sidebar.success(f"Modell '{LOCAL_MODEL_PATH}' erfolgreich geladen!")
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells von '{LOCAL_MODEL_PATH}': {e}")
        st.info("Stelle sicher, dass es ein gültiges YOLOv8-Modell ist und die 'ultralytics'-Bibliothek korrekt installiert ist.")
        st.stop() # Stoppt die Ausführung bei einem Fehler beim Laden des Modells

# --- Kamera-Steuerung ---
st.sidebar.markdown("---")
st.sidebar.header("Kamera-Einstellungen")

# Initialisiere 'start_camera' im Session State, um den Zustand beizubehalten
if "start_camera" not in st.session_state:
    st.session_state["start_camera"] = False

# Checkbox zum Starten/Stoppen der Kamera
start_camera_checkbox = st.sidebar.checkbox(
    "Kamera starten",
    value=st.session_state["start_camera"],
    key="main_camera_toggle" # Eindeutiger Schlüssel für die Checkbox
)

# Aktualisiere den Session State basierend auf der Checkbox
if start_camera_checkbox != st.session_state["start_camera"]:
    st.session_state["start_camera"] = start_camera_checkbox
    if not st.session_state["start_camera"]:
        st.experimental_rerun() # Erzeugt ein Rerun, um den Video-Loop zu beenden

# --- Live-Videoanalyse ---
if st.session_state["start_camera"] and model:
    st.header("Live-Kamera-Feed")
    st.markdown("---")

    st_frame = st.image([]) # Placeholder für den Kamerastream

    assumed_fps = 30
    byte_track = sv.ByteTrack(
        frame_rate=assumed_fps, track_activation_threshold=confidence_threshold
    )

    # Annotatoren initialisieren (Auflösung kann hier statisch sein, da wir keine VideoInfo haben)
    assumed_resolution_wh = (640, 480)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=assumed_resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=assumed_resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness, trace_length=assumed_fps * 2, position=sv.Position.BOTTOM_CENTER
    )

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=assumed_fps))

    cap = cv2.VideoCapture(0) # 0 für die Standardkamera
    if not cap.isOpened():
        st.error("Fehler: Kamera konnte nicht geöffnet werden.")
        st.info("Bitte stellen Sie sicher, dass keine andere Anwendung die Kamera verwendet und die Berechtigungen erteilt sind.")
        st.stop()
    else:
        # Loop, solange die 'Kamera starten'-Checkbox aktiviert ist
        while st.session_state["start_camera"]:
            ret, frame = cap.read()
            if not ret:
                st.warning("Kamera konnte keinen Frame empfangen. Versuche erneut...")
                continue # Versucht den nächsten Frame

            # Inferenz mit dem lokalen YOLOv8-Modell
            # conf und iou werden direkt an predict() übergeben
            results = model.predict(frame, conf=confidence_threshold, iou=iou_threshold, verbose=False)[0]
            
            # Konvertiere Ultralytics Results zu Supervision Detections
            detections = sv.Detections.from_ultralytics(results)
            
            # Filterung durch Polygon Zone
            detections = detections[polygon_zone.trigger(detections)]
            
            # ByteTrack Update
            detections = byte_track.update_with_detections(detections=detections)

            # Punkte transformieren
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < assumed_fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / assumed_fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            # Frame annotieren
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            # Zeigt den annotierten Frame in Streamlit an
            st_frame.image(annotated_frame, channels="BGR", use_column_width=True)

            # Diese zusätzliche Checkbox in der Sidebar wird benötigt, um den Loop zu unterbrechen.
            # Wenn der Benutzer die Haupt-Checkbox in der Sidebar deaktiviert, wird dies hier
            # erkannt und der Loop beendet.
            if not st.sidebar.checkbox("Kamera läuft (deaktivieren zum Stoppen)", value=st.session_state["start_camera"], key="loop_camera_toggle"):
                st.session_state["start_camera"] = False
                break

        cap.release() # Kamera freigeben
        cv2.destroyAllWindows()
else:
    if not st.session_state["start_camera"]:
        st.info("Klicken Sie auf 'Kamera starten' in der Seitenleiste, um die Geschwindigkeitserkennung zu starten.")

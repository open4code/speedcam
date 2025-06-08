import os
from collections import defaultdict, deque

import cv2
import numpy as np
import streamlit as st
from roboflow import Roboflow

import supervision as sv

# Definieren der Quell- und Zielpunkte für die Perspektivtransformation
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

# Klasse für die Perspektivtransformation
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

## Streamlit App-Konfiguration

st.title("Fahrzeuggeschwindigkeitserkennung")

# Benutzereingaben für Modell-ID und API-Schlüssel
# Für Streamlit Share wird empfohlen, Secrets zu verwenden.
# Siehe: https://docs.streamlit.io/deploy/streamlit-cloud/secrets-management
model_id = st.sidebar.text_input("Roboflow Modell ID (z.B. 'projektname/versionsnummer')", value="yolov8x-640")
roboflow_api_key = st.sidebar.text_input("Roboflow API KEY", type="password")

# Standardwerte für Schwellenwerte
confidence_threshold = st.sidebar.slider(
    "Konfidenz-Schwellenwert", min_value=0.0, max_value=1.0, value=0.3, step=0.05
)
iou_threshold = st.sidebar.slider(
    "IOU-Schwellenwert", min_value=0.0, max_value=1.0, value=0.7, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.header("Kamera-Einstellungen")

# Initialisiere 'start_camera' im Session State, wenn nicht vorhanden
if "start_camera" not in st.session_state:
    st.session_state["start_camera"] = False

# Checkbox, die den Zustand in der Session State speichert
start_camera_checkbox = st.sidebar.checkbox("Kamera starten", value=st.session_state["start_camera"])

# Aktualisiere den Session State basierend auf der Checkbox
if start_camera_checkbox != st.session_state["start_camera"]:
    st.session_state["start_camera"] = start_camera_checkbox
    # Wenn die Kamera ausgeschaltet wird, setze den Kamerastream zurück
    if not st.session_state["start_camera"]:
        st.experimental_rerun() # Nötig, um den Stream sofort zu beenden

# Initialisiere die Variable `model` außerhalb des `if`-Blocks
model = None

if roboflow_api_key:
    try:
        # Roboflow-Objekt initialisieren
        rf = Roboflow(api_key=roboflow_api_key)
        
        # Projekt und Modell laden
        # Annahme: model_id ist im Format "projektname/versionsnummer"
        parts = model_id.split("/")
        if len(parts) == 2:
            project_name = parts[0]
            model_version = parts[1]
            project = rf.workspace().project(project_name)
            model = project.version(model_version).model
            st.sidebar.success(f"Roboflow Modell '{model_id}' erfolgreich geladen!")
        else:
            st.sidebar.error("Ungültiges Modell-ID-Format. Bitte verwenden Sie 'projektname/versionsnummer'.")
            model = None

    except Exception as e:
        st.sidebar.error(f"Fehler beim Laden des Roboflow Modells: {e}")
        model = None
else:
    st.sidebar.warning("Bitte geben Sie Ihren Roboflow API KEY ein.")

## Live-Videoanalyse

# Hauptlogik, wenn die Kamera gestartet wird und das Modell geladen ist
if st.session_state["start_camera"] and model:
    st.header("Live-Kamera-Feed")
    st.markdown("---")

    # Placeholder für den Kamerastream
    st_frame = st.image([])

    assumed_fps = 30 # Angenommene Framerate
    
    # ByteTrack Initialisierung
    byte_track = sv.ByteTrack(
        frame_rate=assumed_fps, track_activation_threshold=confidence_threshold
    )

    # Annotatoren initialisieren
    assumed_resolution_wh = (640, 480) # Angenommene Auflösung
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=assumed_resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh=assumed_resolution_wh
    )
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=assumed_fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=assumed_fps))

    # OpenCV VideoCapture für die Kamera
    cap = cv2.VideoCapture(0)  # 0 steht für die Standardkamera

    if not cap.isOpened():
        st.error("Kamera konnte nicht geöffnet werden. Stellen Sie sicher, dass keine andere Anwendung die Kamera verwendet oder die Berechtigungen erteilt wurden.")
    else:
        while st.session_state["start_camera"]: # Loop, solange die Checkbox aktiviert ist
            ret, frame = cap.read()

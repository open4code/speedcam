import os
from collections import defaultdict, deque

import cv2
import numpy as np
import streamlit as st
# NEUE IMPORTE
from ultralytics import YOLO # Importiere YOLO von ultralytics

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

# Benutzereingaben für Modell-Schwellenwerte
confidence_threshold = st.sidebar.slider(
    "Konfidenz-Schwellenwert", min_value=0.0, max_value=1.0, value=0.3, step=0.05
)
iou_threshold = st.sidebar.slider(
    "IOU-Schwellenwert", min_value=0.0, max_value=1.0, value=0.7, step=0.05
)

# NEU: Pfad zum lokalen Modell
# Passe diesen Pfad an den Namen deiner heruntergeladenen Modell-Datei an!
LOCAL_MODEL_PATH = "best.pt" # Beispiel: wenn du 'best.pt' heruntergeladen hast
# Wenn du ein ONNX-Modell hast, wäre es z.B. "model.onnx"

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
    if not st.session_state["start_camera"]:
        st.experimental_rerun()

# Initialisiere das Modell global (oder lade es einmalig)
# Hier wird das Modell geladen. Dies passiert einmalig beim Start der App
# und bei jeder Neuausführung (z.B. Slider-Änderung), aber nicht im Loop.
try:
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.error(f"Modell-Datei '{LOCAL_MODEL_PATH}' nicht gefunden. Bitte lade dein Roboflow-Modell herunter und platziere es im selben Verzeichnis.")
        model = None
    else:
        model = YOLO(LOCAL_MODEL_PATH)
        st.sidebar.success(f"Modell '{LOCAL_MODEL_PATH}' erfolgreich geladen!")
except Exception as e:
    st.sidebar.error(f"Fehler beim Laden des Modells: {e}")
    model = None

# st.sidebar.warning("Bitte geben Sie Ihren Roboflow API KEY ein.") # Nicht mehr benötigt

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
            if not ret:
                st.warning("Kamera konnte keinen Frame empfangen.")
                break

            # NEU: Inferenz mit dem ultralytics YOLO-Modell
            # Die .predict() Methode gibt eine Liste von Results-Objekten zurück, eines pro Bild.
            # Für ein einzelnes Bild nehmen wir das erste Element.
            results = model.predict(frame, conf=confidence_threshold, iou=iou_threshold, verbose=False)[0]
            
            # Konvertiere ultralytics Results zu Supervision Detections
            detections = sv.Detections.from_ultralytics(results)
            
            # Filterung durch polygon_zone und NMS ist möglicherweise schon durch predict() abgedeckt,
            # aber wir behalten es der Konsistenz halber bei, falls du es spezifisch brauchst.
            # detections = detections[detections.confidence > confidence_threshold] # Bereits in predict() über conf
            detections = detections[polygon_zone.trigger(detections)]
            # detections = detections.with_nms(threshold=iou_threshold) # Bereits in predict() über iou
            
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
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

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            # Zeigt den annotierten Frame in Streamlit an
            st_frame.image(annotated_frame, channels="BGR", use_column_width=True)

            # Wichtig: Diese Zeile ermöglicht es Streamlit, den Loop zu unterbrechen,
            # wenn die Checkbox deaktiviert wird.
            if not st.sidebar.checkbox("Kamera starten", value=st.session_state["start_camera"], key="camera_toggle_in_loop"):
                st.session_state["start_camera"] = False
                break # Beendet den while-Loop

        cap.release() # Kamera freigeben, wenn der Loop beendet ist
        cv2.destroyAllWindows()
else:
    if not st.session_state["start_camera"]:
        st.info("Klicken Sie auf 'Kamera starten' in der Seitenleiste, um die Geschwindigkeitserkennung zu starten.")
    elif model is None: # Füge diese Bedingung hinzu, falls das Modell nicht geladen werden konnte
        st.warning("Modell konnte nicht geladen werden. Bitte überprüfen Sie den Pfad und die Verfügbarkeit der Datei.")

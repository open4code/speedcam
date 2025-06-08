import os
from collections import defaultdict, deque

import cv2
import numpy as np
import streamlit as st
# NEUE IMPORTE
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

---

## Streamlit App-Konfiguration

st.title("Fahrzeuggeschwindigkeitserkennung")

# Benutzereingaben für Modell-ID und API-Schlüssel
# Hinweis: Für ein öffentlicheres Deployment auf Streamlit Share wäre es besser,
# den API-Key über Streamlit Secrets zu verwalten.
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
start_camera = st.sidebar.checkbox("Kamera starten")

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

---

## Live-Videoanalyse

# Hauptlogik, wenn die Kamera gestartet wird und das Modell geladen ist
if start_camera and model:
    st.header("Live-Kamera-Feed")
    st.markdown("---")

    # Placeholder für den Kamerastream
    st_frame = st.image([])

    # VideoInfo ist für die Kamera nicht direkt anwendbar, aber wir können Standardwerte annehmen
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
        st.error("Kamera konnte nicht geöffnet werden. Stellen Sie sicher, dass keine andere Anwendung die Kamera verwendet.")
    else:
        # Loop solange die Checkbox aktiviert ist
        while st.session_state.get("start_camera_state", True): # Überprüfen des Checkbox-Status in Streamlit
            ret, frame = cap.read()
            if not ret:
                st.warning("Kamera konnte keinen Frame empfangen.")
                break

            # Drehen des Frames bei Bedarf, da viele Webcams spiegelverkehrt sind
            # frame = cv2.flip(frame, 1)

            # Frame für die Roboflow-Inferenz auf die erwartete Größe anpassen, falls nötig
            # model.infer() sollte die Größe selbst anpassen, aber wenn Sie spezifische Anforderungen haben,
            # können Sie hier frame = cv2.resize(frame, (W, H)) hinzufügen.

            results = model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)
            detections = detections[detections.confidence > confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=iou_threshold)
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

            # Aktualisieren Sie den Checkbox-Status in Streamlit
            # Streamlit-Widgets geben ihren Zustand zurück. Wir müssen ihn in der Session State speichern,
            # um ihn im nächsten Durchlauf des Loops abzurufen.
            st.session_state["start_camera_state"] = st.sidebar.checkbox("Kamera starten", value=st.session_state.get("start_camera_state", True), key="camera_toggle_loop")

            # Falls der Benutzer die Checkbox deaktiviert hat, beende den Loop
            if not st.session_state["start_camera_state"]:
                break # Beendet den while-Loop

        cap.release() # Kamera freigeben, wenn der Loop beendet ist
        cv2.destroyAllWindows()
else:
    if not start_camera:
        st.info("Klicken Sie auf 'Kamera starten' in der Seitenleiste, um die Geschwindigkeitserkennung zu starten.")

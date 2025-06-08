import streamlit as st
import cv2
from ultralytics import YOLO
from ultralytics import solutions
import numpy as np
import tempfile
import time

# --- Streamlit UI Konfiguration ---
st.set_page_config(
    page_title="YOLOv8 Geschwindigkeitsmessung",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 YOLOv8 Geschwindigkeitsmessung")
st.markdown("""
Diese Anwendung verwendet das Ultralytics YOLOv8-Modell und den `SpeedEstimator`
um die Geschwindigkeit von Fahrzeugen in einem Video zu schätzen.
""")

# --- Modell laden ---
@st.cache_resource
def load_model():
    """Lädt das YOLOv8n Modell nur einmal."""
    st.write("Lade YOLOv8n Modell...")
    model = YOLO("yolov8n.pt")  # Du kannst auch "yolov8s.pt", "yolov8m.pt" usw. verwenden
    st.write("Modell geladen!")
    return model

model = load_model()

# --- Upload des Videos oder Webcam-Auswahl ---
st.sidebar.header("Videoquelle auswählen")
video_source = st.sidebar.radio("Wähle eine Quelle:", ("Videodatei hochladen", "Webcam (Experimentell)"))

video_file = None
if video_source == "Videodatei hochladen":
    video_file = st.sidebar.file_uploader("Lade ein Video hoch (z.B. .mp4, .avi)", type=["mp4", "avi", "mov"])
else:
    st.sidebar.warning("Die Webcam-Funktion in Streamlit kann je nach Systemkonfiguration instabil sein und erfordert `streamlit-webrtc` für eine robuste Lösung.")
    # Für eine robuste Webcam-Lösung in Streamlit wäre `streamlit-webrtc` notwendig.
    # Hier simulieren wir nur den Start.
    st.sidebar.info("Starte die App lokal und drücke 'Strg+C' im Terminal um zu stoppen.")
    st.sidebar.write("Webcam-Support erfordert zusätzliche Konfiguration für Echtzeit-Streaming. Dieses Beispiel konzentriert sich auf Videodateien.")
    st.info("Für Webcam-Echtzeit-Verarbeitung siehe `streamlit-webrtc` Beispiele.")


# --- Konfiguration der Geschwindigkeitsmessung ---
st.sidebar.header("Geschwindigkeitsmessung Konfiguration")
confidence = st.sidebar.slider("Erkennungs-Confidence (0-100)", 25, 100, 40) / 100
iou = st.sidebar.slider("IoU-Schwelle (0-100)", 25, 100, 70) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Linien für die Geschwindigkeitsmessung")
st.sidebar.markdown("""
Definiere die Region (Linien) für die Geschwindigkeitsmessung.
Die Punkte werden als `[(x1, y1), (x2, y2), ..., (xn, yn)]` eingegeben.
""")

# Standardregion für ein 1280x720 Video (anpassen)
# Dies sind zwei horizontale Linien. Die obere Linie ist (0,400) bis (1280,400)
# die untere ist (0,500) bis (1280,500). Der Abstand zwischen den Linien wird geschätzt.
default_region_str = "[(0, 400), (1280, 400), (1280, 500), (0, 500)]"
region_input = st.sidebar.text_area("Region-Punkte (JSON-Format)", default_region_str)

try:
    region = eval(region_input) # eval() ist hier für die Einfachheit verwendet, aber in Prod nicht empfohlen
    # Sicherere Alternative: json.loads wenn Region als String-JSON kommt
    # import json
    # region = json.loads(region_input)
    if not isinstance(region, list) or not all(isinstance(p, tuple) and len(p) == 2 for p in region):
        st.sidebar.error("Ungültiges Region-Format. Bitte `[(x1, y1), (x2, y2), ...]` verwenden.")
        region = []
except Exception as e:
    st.sidebar.error(f"Fehler beim Parsen der Region: {e}")
    region = []

st.sidebar.markdown("""
**Kalibrierung:** Der *reale* Abstand zwischen den beiden horizontalen Linien
(oder den entsprechenden Punkten deiner Polygon-Region) ist entscheidend
für eine genaue Geschwindigkeitsmessung.
""")
# Der 'pixel_per_meter' Wert ist eine entscheidende Kalibrierungskonstante.
# Er muss experimentell für deine Kamera und Szeneneinrichtung ermittelt werden.
# Ein größerer Wert bedeutet, dass ein Pixel eine geringere reale Distanz darstellt (Objekte erscheinen kleiner),
# was zu höheren berechneten Geschwindigkeiten führen würde, wenn der Pixelweg gleich bleibt.
# Beispiel: Wenn ein 1 Meter langes Objekt 100 Pixel hoch ist, dann ist pixel_per_meter = 100.
# Ultralytics SpeedEstimator erwartet `meter_per_pixel`. Wenn 1 Pixel 0.01 Meter ist, dann `0.01`.
# Oder direkt eine Referenzdistanz in Pixeln zu einer Referenzdistanz in Metern.
meter_per_pixel = st.sidebar.number_input("Meter pro Pixel (Kalibrierung)", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f")


st.sidebar.subheader("Zusätzliche Einstellungen")
show_labels = st.sidebar.checkbox("Labels anzeigen", value=True)
line_width = st.sidebar.slider("Linienbreite der Bounding Boxes", 1, 10, 2)
enable_tracking = st.sidebar.checkbox("Objekt-Tracking aktivieren", value=True) # ByteTrack ist Standard
tracker_type = st.sidebar.selectbox("Tracker-Typ", ("bytetrack.yaml", "botsort.yaml")) if enable_tracking else None


# --- Hauptverarbeitung ---
st.header("Videoanalyse")
if video_file:
    st.video(video_file) # Zeigt das Originalvideo an

    # Speichern des hochgeladenen Videos temporär
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Verarbeitetes Video")
        processed_video_placeholder = st.empty() # Placeholder für das verarbeitete Video

    with col2:
        st.subheader("Geschwindigkeitsstatistik")
        speed_stats_placeholder = st.empty() # Placeholder für Statistiken

    start_button = st.button("Analyse starten")

    if start_button:
        if not region:
            st.error("Bitte definiere gültige Region-Punkte für die Geschwindigkeitsmessung.")
        else:
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Fehler: Video konnte nicht geöffnet werden.")
                else:
                    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

                    st.write(f"Video-Info: {w}x{h} @ {fps} FPS")

                    # Initialisiere SpeedEstimator
                    speed_estimator = solutions.SpeedEstimator(
                        reg_pts=region,
                        names=model.names, # Namen der Klassen vom Modell
                        view_img=False, # Nicht direkt anzeigen, Streamlit kümmert sich
                        line_thickness=line_width,
                        font_thickness=line_width,
                        line_color=(0, 255, 0), # Grüne Linie
                        tracker=tracker_type if enable_tracking else None,
                        conf_threshold=confidence,
                        iou_threshold=iou,
                        # meter_per_pixel ist der entscheidende Kalibrierungsfaktor
                        # Er gibt an, wie viele Meter einem Pixel auf der Ebene der Linien entsprechen.
                        meter_per_pixel=meter_per_pixel
                    )

                    # Temporäre Datei für das Ausgabevideo
                    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec für .mp4
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

                    frame_count = 0
                    start_time = time.time()
                    tracked_speeds = {} # Speichert Geschwindigkeiten pro Track ID

                    progress_bar = st.progress(0)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Verarbeite den Frame mit dem SpeedEstimator
                        results = model.track(frame, persist=True, tracker=tracker_type, conf=confidence, iou=iou)[0]
                        # Der SpeedEstimator benötigt die rohen YOLO-Ergebnisse und verarbeitet sie intern
                        processed_frame = speed_estimator.process(frame, results)

                        # Sammle die geschätzten Geschwindigkeiten
                        # Die SpeedEstimator-Klasse aktualisiert intern Geschwindigkeiten in `speed_estimator.spd`
                        for track_id, speed_val in speed_estimator.spd.items():
                            tracked_speeds[track_id] = speed_val

                        # Zeige das verarbeitete Frame an
                        processed_video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

                        # Aktualisiere die Geschwindigkeitsstatistik
                        speed_info = "Aktuelle Geschwindigkeiten:\n"
                        if tracked_speeds:
                            for track_id, speed_val in tracked_speeds.items():
                                speed_info += f"Fahrzeug {track_id}: {speed_val:.1f} km/h\n"
                        else:
                            speed_info += "Warte auf Fahrzeugerkennung und Geschwindigkeitsmessung..."
                        speed_stats_placeholder.text(speed_info)

                        out.write(processed_frame)

                        frame_count += 1
                        progress = min(1.0, frame_count / total_frames)
                        progress_bar.progress(progress)

                    cap.release()
                    out.release()
                    end_time = time.time()
                    st.success(f"Videoanalyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")

                    st.markdown("---")
                    st.subheader("Herunterladen des verarbeiteten Videos")
                    with open(output_video_path, "rb") as file:
                        st.download_button(
                            label="Verarbeitetes Video herunterladen",
                            data=file.read(),
                            file_name="speed_estimation_output.mp4",
                            mime="video/mp4"
                        )
                    # cleanup temp file
                    import os
                    os.remove(video_path)
                    os.remove(output_video_path)

            except Exception as e:
                st.error(f"Ein Fehler ist aufgetreten: {e}")
                st.exception(e) # Zeigt detaillierten Fehler-Trace an
else:
    st.info("Bitte lade ein Video hoch, um die Analyse zu starten.")

st.markdown("---")
st.markdown("Entwickelt mit Ultralytics YOLOv8 und Streamlit.")

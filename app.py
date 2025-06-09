import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Fahrzeug-Erkennung & Zählung",
    layout="wide", # Macht die App so breit wie möglich
    initial_sidebar_state="expanded"
)

st.title("🚗 LKW, Bus und Auto-Erkennung und Zählung aus Video")
st.markdown("Erkennt und zählt **Autos, Lastwagen und Busse** aus einem hochgeladenen Video mit YOLOv8.")

# --- YOLOv8 Modell laden (wird nur einmal geladen) ---
@st.cache_resource
def load_yolo_model():
    """Lädt das YOLOv8-Modell und cached es."""
    st.write("Modell wird geladen (kann einen Moment dauern)...")
    try:
        model = YOLO('yolov8n.pt') # Nutze die nano-Version für bessere Performance
        st.success("Modell erfolgreich geladen!")
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        st.stop() # Stoppt die App, wenn das Modell nicht geladen werden kann

model = load_yolo_model()

# Definiere die Klassen, an denen wir interessiert sind (Namen aus dem COCO-Datensatz)
# Wir verwenden die Klassennamen und werden ihre IDs dynamisch vom Modell abrufen.
TARGET_CLASS_NAMES = ['car', 'truck', 'bus']

# Erstelle ein Set von Ziel-Klassen-IDs basierend auf den Namen aus dem geladenen Modell
# Dies ist robuster, da es direkt die Namen aus dem Modell verwendet.
TARGET_CLASS_IDS = set()
CLASS_ID_TO_NAME = {} # Mapping von ID zu Klassenname für die Anzeige
if hasattr(model, 'names') and model.names:
    for class_id, class_name in model.names.items():
        if class_name in TARGET_CLASS_NAMES:
            TARGET_CLASS_IDS.add(class_id)
            CLASS_ID_TO_NAME[class_id] = class_name
else:
    st.error("Fehler: Konnte Klassennamen aus dem YOLO-Modell nicht laden. Bitte überprüfe das Modell.")
    st.stop()

if not TARGET_CLASS_IDS:
    st.error(f"Fehler: Die gewünschten Klassen {TARGET_CLASS_NAMES} wurden im Modell nicht gefunden. Bitte überprüfe die Klassennamen.")
    st.stop()

# --- Main Content Area for Video Upload ---
st.subheader("Video-Datei hochladen zur Fahrzeug-Erkennung")
uploaded_file = st.file_uploader("Wähle eine Video-Datei aus", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.write("Verarbeite Video...")
    
    # Speichere die hochgeladene Datei temporär
    # tempfile wird verwendet, um eine temporäre Datei sicher zu erstellen und zu verwalten.
    # delete=False sorgt dafür, dass die Datei nach dem Schließen des Descriptors nicht sofort gelöscht wird.
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_file.read())
        temp_file_path = tfile.name
    
    # Streamlit placeholder für die verarbeiteten Video-Frames
    video_placeholder = st.empty()

    # Initialisiere den Video-Capture-Objekt mit der temporären Datei
    cap = cv2.VideoCapture(temp_file_path)
    
    if not cap.isOpened():
        st.error("Fehler: Konnte Video-Datei nicht öffnen.")
    else:
        # Hole die Gesamtanzahl der Frames für den Fortschrittsbalken
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        frame_count = 0

        # Schleife durch jeden Frame des Videos
        while cap.isOpened():
            ret, frame = cap.read() # Lese den nächsten Frame
            if not ret:
                # Wenn keine Frames mehr vorhanden sind oder ein Fehler auftritt, beende die Schleife
                break
            
            # --- Frame-Verarbeitung (Vehicle Detection Logic) ---
            # Konvertiere den Frame von BGR (OpenCV-Standard) zu RGB für YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Führe die Objekterkennung mit dem YOLO-Modell durch
            results = model(rgb_frame, verbose=False) # verbose=False unterdrückt Konsolenausgaben von YOLO

            vehicle_count = 0 # Zähler für erkannte Fahrzeuge
            annotated_frame = frame.copy() # Erstelle eine Kopie des Original-Frames zum Zeichnen
            
            # Überprüfe, ob Ergebnisse vorhanden sind und verarbeite sie
            if results and len(results) > 0:
                detections = results[0].boxes # Greife auf die erkannten Bounding Boxes zu
                for box in detections:
                    # Überprüfe, ob die erkannte Klasse eine unserer Ziel-Fahrzeugklassen ist
                    if int(box.cls) in TARGET_CLASS_IDS:
                        vehicle_count += 1 # Erhöhe den Fahrzeugzähler
                        # Extrahiere die Koordinaten der Bounding Box und die Konfidenz
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        detected_class_name = CLASS_ID_TO_NAME.get(int(box.cls), "Unbekannt") # Hol den Namen

                        # Zeichne ein Rechteck um das erkannte Fahrzeug
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Grün, 2px Dicke
                        # Füge ein Label mit der Klasse und Konfidenz hinzu
                        label = f"{detected_class_name.capitalize()}: {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Zeige die Gesamtzahl der erkannten Fahrzeuge im Frame an
            count_text = f"Fahrzeuge: {vehicle_count}"
            cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # --- Ende der Frame-Verarbeitung ---

            # Konvertiere den annotierten Frame wieder zu RGB für die Anzeige in Streamlit
            processed_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Zeige den verarbeiteten Frame im Streamlit-Platzhalter an
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
            
            frame_count += 1
            # Aktualisiere den Fortschrittsbalken
            progress_bar.progress(frame_count / total_frames)

        # Gib das Video-Capture-Objekt frei und schließe es
        cap.release()
        progress_bar.empty() # Entferne den Fortschrittsbalken nach Abschluss
        st.success("Video-Verarbeitung abgeschlossen!")
        
        # Lösche die temporäre Datei
        os.unlink(temp_file_path)

else:
    st.info("Bitte lade eine Video-Datei hoch, um die Erkennung zu starten.")

# Add a simple footer with custom CSS for better styling
st.markdown("""
<style>
    /* Allgemeine Anpassungen für den Hauptcontainer */
    .reportview-container .main .block-container{
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    /* Stellt sicher, dass das Video gut in den Container passt */
    .stImage > img {
        object-fit: contain;
    }
</style>
""", unsafe_allow_html=True)

#!/usr/bin/env python3
"""
stream_poo.py — Flux vidéo HTTPS avec détection IMX500 (AI Camera)
Version Orientée Objet (Conforme Cahier des Charges BTS CIEL)
"""

import io
import ssl
import time
import threading
import logging
import os               
import requests         
from datetime import datetime 
from http.server import BaseHTTPRequestHandler, HTTPServer
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
import cv2
import numpy as np
from gpiozero import Buzzer
from time import sleep

# --- Masquer l'avertissement SSL pour un terminal propre ---
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Configuration Globale ───────────────────────────────────────────────────
MODEL_PATH   = "/home/projet_camera/Projet_poulailler/modele_coco.rpk"
CERT_PATH    = "/home/projet_camera/Projet_poulailler/cert.pem"
KEY_PATH     = "/home/projet_camera/Projet_poulailler/key.pem"
PORT         = 8443
STREAM_FPS   = 20
STREAM_W     = 640
STREAM_H     = 480
alarme = Buzzer(6)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stream")

# ─── Classes Utilitaires ─────────────────────────────────────────────────────
class FrameBuffer:
    """Classe gérant le partage de l'image vidéo entre les threads (Caméra -> Web)"""
    def __init__(self):
        self._frame = None
        self._lock  = threading.Lock()

    def write(self, frame_bytes):
        with self._lock:
            self._frame = frame_bytes

    def read(self):
        with self._lock:
            return self._frame

# Instanciation du buffer global utilisé par le serveur Web
frame_buffer = FrameBuffer()


# ─── Classe d'Alerte (Demandée au Cahier des Charges) ────────────────────────
class CAlerte:
    """Gère la logique de sécurité : capture d'image, envoi de requêtes HTTPS et alarme sonore"""
    def __init__(self, url_urgence="https://127.0.0.1:5000/api/urgence_predateur", cooldown=10.0, capture_dir="static/captures"):
        self.url = url_urgence
        self.cooldown = cooldown
        self.capture_dir = capture_dir
        self.last_alert_time = 0.0

        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir, exist_ok=True)

    def _declencher_alarme(self):
        """Fait sonner l'alarme en arrière-plan pendant le cooldown"""
        log.info("🔊 Activation de la sirène (10 secondes) !")
        try:
            alarme.on()
            sleep(10) # Utilise le temps défini dans ton alarme.py
            alarme.off()
        except Exception as e:
            log.error(f"❌ Erreur avec l'alarme sonore : {e}")
            alarme.off()

    def verifier_et_declencher(self, frame):
        """Vérifie le cooldown et déclenche l'alerte si nécessaire"""
        now = time.time()
        if now - self.last_alert_time > self.cooldown:
            self.last_alert_time = now
            
            # 1. Sauvegarde de la preuve photographique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predateur_{timestamp}.jpg"
            filepath = os.path.join(self.capture_dir, filename)
            cv2.imwrite(filepath, frame)
            log.info(f"📸 DÉTECTION ! Photo sauvegardée : {filename}")

            # 2. Déclenchement de l'alarme physique (Buzzer) dans un thread séparé
            # Cela permet de ne pas bloquer le flux vidéo HTTPS
            threading.Thread(target=self._declencher_alarme, daemon=True).start()

            # 3. Envoi de l'ordre d'urgence au RPi Central (app.py)
            try:
                requests.get(self.url, timeout=2, verify=False)
                log.info("🚨 Ordre d'urgence envoyé au serveur web !")
            except requests.exceptions.RequestException as e:
                log.error(f"❌ Impossible de joindre Flask : {e}")
# ─── Classe Principale IA (Demandée au Cahier des Charges) ───────────────────
class CCameraAI:
    """Gère le capteur Sony IMX500, l'inférence YOLOv8n et le traitement d'image"""
    def __init__(self, model_path, confidence=0.45):
        self.model_path = model_path
        self.confidence = confidence
        self.alerte_system = CAlerte() # Intégration du module d'alerte
        
        # Configuration des classes et labels (0: Humain, 15: Poule, 16-18: Prédateurs)
        self.classes_predateur = [16, 17, 18]
        self.labels_fr = {
            0:  ("Humain",    (0, 120, 255)),
            15: ("Poule",     (0, 210, 80)),
            16: ("Prédateur", (0, 0, 220)),
            17: ("Prédateur", (0, 0, 220)),
            18: ("Prédateur", (0, 0, 220)),
        }

    def _parse_detections(self, outputs):
        """Extrait les 4 tenseurs de sortie du NPU IMX500"""
        boxes   = outputs[0][0]
        scores  = outputs[1][0]
        classes = outputs[2][0]
        n_dets  = int(outputs[3][0][0]) if outputs[3] is not None else 300

        results = []
        for i in range(min(n_dets, 300)):
            score = float(scores[i])
            if score > 1.0: score = score / 255.0
            if score < self.confidence: continue
            
            cls_id = int(classes[i])
            results.append((cls_id, score, boxes[i]))
        return results

    def _draw_detections(self, frame, outputs):
        """Dessine les boîtes et vérifie la présence de prédateurs"""
        h, w = frame.shape[:2]
        alerte_detectee = False
        detections = self._parse_detections(outputs)

        for cls_id, score, box in detections:
            label_fr, color = self.labels_fr.get(cls_id, (f"cls{cls_id}", (200, 200, 200)))

            # Mise à l'échelle des coordonnées normalisées vers les pixels
            x1, y1 = max(0, int(box[0] / 320.0 * w)), max(0, int(box[1] / 320.0 * h))
            x2, y2 = min(w-1, int(box[2] / 320.0 * w)), min(h-1, int(box[3] / 320.0 * h))

            # Application des rectangles
            if cls_id in self.classes_predateur:
                alerte_detectee = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Affichage du texte
            txt = f"{label_fr} {score:.0%}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Bandeau général si alerte
        if alerte_detectee:
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 200), -1)
            cv2.putText(frame, "  ALERTE PREDATEUR  ", (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

        return frame, alerte_detectee

    def start_stream(self):
        """Boucle principale exécutée dans un thread séparé"""
        log.info("Chargement du modèle IMX500...")
        imx500 = IMX500(self.model_path)
        picam2 = Picamera2(imx500.camera_num)

        config = picam2.create_preview_configuration(
            main={"size": (STREAM_W, STREAM_H), "format": "RGB888"},
            controls={"FrameRate": STREAM_FPS}, buffer_count=4
        )
        picam2.configure(config)
        picam2.start()
        log.info("Caméra démarrée !")

        interval = 1.0 / STREAM_FPS

        while True:
            t0 = time.time()

            # 1. Capture de l'image brute
            request = picam2.capture_request()
            frame_rgb = request.make_array("main")
            metadata  = request.get_metadata()
            request.release()

            # Correction des couleurs (BGRA -> BGR)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_BGRA2BGR) if frame_rgb.shape[2] == 4 else frame_rgb.copy()
            
            # 2. Récupération des tenseurs IA
            try:
                outputs = imx500.get_outputs(metadata, add_batch=True)
                if outputs is not None:
                    frame_bgr, alerte_detectee = self._draw_detections(frame_bgr, outputs)
                    
                    # 3. Logique d'Alerte
                    if alerte_detectee:
                        self.alerte_system.verifier_et_declencher(frame_bgr)

            except Exception as e:
                log.debug("Erreur analyse IA : %s", e)

            # 4. Affichage de l'heure et encodage JPEG
            ts = time.strftime("%d/%m/%Y %H:%M:%S")
            cv2.putText(frame_bgr, ts, (10, frame_bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            frame_buffer.write(jpeg.tobytes())

            # Temporisation pour maintenir le FPS
            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))


# ─── Serveur HTTP MJPEG ──────────────────────────────────────────────────────
BOUNDARY = b"--frame"
class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == "/": self._serve_page()
        elif self.path == "/stream": self._serve_stream()
        else: self.send_error(404)

    def _serve_page(self):
        html = b"<html><body><h1>Flux IA Actif</h1><img src='/stream'></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                frame = frame_buffer.read()
                if frame is None:
                    time.sleep(0.02)
                    continue
                self.wfile.write(BOUNDARY + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / STREAM_FPS)
        except (BrokenPipeError, ConnectionResetError): pass


# ─── Lancement Principal ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Instanciation de l'IA (POO)
    systeme_ia = CCameraAI(MODEL_PATH)
    
    # 2. Démarrage du thread de capture
    t = threading.Thread(target=systeme_ia.start_stream, daemon=True)
    t.start()
    
    log.info("Attente de la première image...")
    while frame_buffer.read() is None: time.sleep(0.1)

    # 3. Démarrage du serveur HTTPS
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_PATH, KEY_PATH)
    server = HTTPServer(("0.0.0.0", PORT), StreamHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)

    log.info("Serveur HTTPS local démarré sur le port %d", PORT)
    try: 
        server.serve_forever()
    except KeyboardInterrupt: 
        log.info("Arrêt du serveur.")
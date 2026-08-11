# poulailler_ia_surveillance
Système IoT Edge AI de vidéosurveillance : détection de prédateurs en temps réel (YOLOv8n) sur Raspberry Pi 5 et capteur NPU IMX500, avec interface web chiffrée (HTTPS).

# 🐔 Poulailler Autonome et Connecté - Module IA & Cybersécurité

*Projet de fin d'études BTS CIEL (Session 2026) - Conception d'un système de supervision Edge AI.*

---

## Présentation du Projet
Ce dépôt contient le code source et la documentation de mon périmètre individuel développé dans le cadre d'un projet d'équipe (5 étudiants)[cite: 9]. L'objectif global était de rendre un poulailler commercial 100 % autonome et connectable à distance[cite: 8, 9]. 

Ma mission s'est concentrée sur la **surveillance vidéo intelligente et la sécurisation des flux réseau** : détecter automatiquement les prédateurs par IA embarquée, déclencher une alerte physique et sécuriser l'intégralité des communications de l'interface web[cite: 8, 9].

---

## ⚙️ Fonctionnalités Principales

*   **Edge AI (YOLOv8n) :** Entraînement en Transfer Learning sur un dataset personnalisé (Renard, Chien/Loup, Poule)[cite: 8, 9]. Le modèle a été quantifié en INT8 et exporté pour s'exécuter directement sur le NPU (Neural Processing Unit) du capteur IMX500, soulageant ainsi le processeur principal[cite: 8, 9].
*   **Cybersécurité (Tunnel HTTPS) :** Refus d'utiliser un flux vidéo HTTP en clair. Mise en place d'un tunnel sécurisé de bout en bout avec OpenSSL, certificat auto-signé RSA 2048 et création d'un proxy Flask pour éviter les erreurs de *mixed content*[cite: 8, 9].
*   **Traitement Multithreading :** Architecture logicielle asynchrone (séparation du thread d'acquisition caméra et du thread serveur web avec `threading.Lock`) garantissant un flux fluide à 20 FPS sans latence, même lors du déclenchement matériel de l'alarme[cite: 8, 9].

---

## 🛠️ Architecture Matérielle

*   **Carte mère :** Raspberry Pi 5[cite: 8, 9].
*   **Capteur IA :** Raspberry Pi AI Camera (Sony IMX500)[cite: 8, 9].
*   **Actionneurs :** Alarme visuelle Velleman 12V pilotée par relais (GPIO)[cite: 8, 9].

---

## Performances du Modèle

L'entraînement du modèle YOLOv8n sur un dataset unifié via Roboflow a donné les résultats suivants :
*   **Poule (Chicken) :** mAP50 de 0.868[cite: 8].
*   **Chien/Loup (Dog & Wolf) :** mAP50 de 0.753[cite: 8].
*   **Renard (Fox) :** mAP50 de 0.260 *(score limité par un jeu de données réduit à 211 images, nécessitant un enrichissement futur)*[cite: 8].

---

## Démonstration Visuelle

Les captures d'écran illustrant l'interface d'historique des intrusions, les schémas réseau et le câblage matériel sont disponibles dans le dossier `images` de ce dépôt.

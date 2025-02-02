import cv2
from ultralytics import YOLO
import sys
import logging

# Désactiver les logs de la bibliothèque Ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Charger le modèle YOLO
model = YOLO("yolov5su.pt")  # Modèle pré-entraîné sur COCO

# Charger une vidéo ou accéder à la webcam
cap = cv2.VideoCapture(0)  # Changez le 0 pour votre caméra ou un fichier vidéo

# Classe cible (par exemple : "sports ball") passée en argument
if len(sys.argv) < 2:
    print("Veuillez fournir le nom de la classe cible comme argument.")
    sys.exit(1)

class_target = sys.argv[1]

# Liste des classes du modèle
class_names = model.names  # Récupère la liste des noms de classes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Détection YOLO sur chaque image
    results = model(frame)

    # Annoter uniquement la classe cible
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())  # Index de la classe détectée
            class_name = class_names[cls]  # Nom de la classe
            if class_name == class_target.lower():  # Comparer avec le nom de la classe cible
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordonnées de la boîte
                confidence = box.conf[0].item()  # Confiance
                label = f"{class_name} ({confidence:.2f})"

                # Calcul du centre de la boîte
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Affichage de la position (x, y) ou du centre
                print(f"Position du centre : ({center_x}, {center_y})")

                # Dessiner la boîte englobante et le centre sur l'image
                color = (0, 255, 0)  # Couleur de la boîte
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Afficher le centre (un petit cercle)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Cercle rouge au centre

    # Afficher le flux vidéo
    cv2.imshow("YOLOv5 Detection", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

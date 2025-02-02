import numpy as np
import cv2
from ultralytics import YOLO
import glob
import logging

# Désactiver les logs de la bibliothèque Ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def calibrate_cameras():
    # Step 1: Settings for calibration
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chessboard dimensions
    chessboard_size = (9, 7)  # Number of inner corners per a chessboard row and column

    # Prepare object points, e.g., (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * 2  # Si chaque case fait 3 cm

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real-world space
    imgpoints_left = []  # 2D points in image plane of left camera
    imgpoints_right = []  # 2D points in image plane of right camera

    # Initialize the camera
    cap_left = cv2.VideoCapture(2)  # ID de la caméra gauche
    cap_right = cv2.VideoCapture(3)  # ID de la caméra droite

    # Check if the cameras opened correctly
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open one or both cameras.")
        exit()

    # Capture images from the cameras
    image_count = 0
    max_images = 20  # Number of images to capture for calibration

    while image_count < max_images:
        # Capture frames from both cameras
        ret_left, img_left = cap_left.read()
        ret_right, img_right = cap_right.read()

        if not (ret_left and ret_right):
            print("Erreur lors de la capture des images.")
            break

        # Convert to grayscale
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

        if ret_left and ret_right:
            # Refine the corners
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            # Append object points and image points
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

            image_count += 1

            # Draw and display the corners
            cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
            cv2.drawChessboardCorners(img_right, chessboard_size, corners_right, ret_right)

            cv2.imshow('Camera Gauche', img_left)
            cv2.imshow('Camera Droite', img_right)

            cv2.waitKey(500)  # Wait for a short time before capturing the next image
            print("Damier capturé !")
        else:
            # Affichage sans coins détectés
            cv2.imshow('Camera Gauche', img_left)
            cv2.imshow('Camera Droite', img_right)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

    # Step 2: Calibrate each camera individually
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, gray_left.shape[::-1], None, None
    )
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, gray_right.shape[::-1], None, None
    )

    # Step 3: Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
        gray_left.shape[::-1],
        criteria=criteria,
        flags=flags
    )

    # Step 4: Stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], R, T
    )

    # Save the calibration data as .npz file
    np.savez('stereo_calibration.npz',
            mtx_left=mtx_left, dist_left=dist_left,
            mtx_right=mtx_right, dist_right=dist_right,
            R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

    # Save results in a .txt file
    with open("stereo_calibration_results.txt", "w") as f:
        f.write("Left Camera Matrix (mtx_left):\n")
        f.write(np.array2string(mtx_left, precision=4, separator=' ') + "\n\n")
        f.write("Left Camera Distortion Coefficients (dist_left):\n")
        f.write(np.array2string(dist_left, precision=4, separator=' ') + "\n\n")
        f.write("Right Camera Matrix (mtx_right):\n")
        f.write(np.array2string(mtx_right, precision=4, separator=' ') + "\n\n")
        f.write("Right Camera Distortion Coefficients (dist_right):\n")
        f.write(np.array2string(dist_right, precision=4, separator=', ') + "\n\n")
        f.write("Rotation Matrix (R):\n")
        f.write(np.array2string(R, precision=4, separator=', ') + "\n\n")
        f.write("Translation Vector (T):\n")
        f.write(np.array2string(T, precision=4, separator=', ') + "\n\n")
        f.write("Stereo Rectification Matrix (R1):\n")
        f.write(np.array2string(R1, precision=4, separator=', ') + "\n\n")
        f.write("Stereo Rectification Matrix (R2):\n")
        f.write(np.array2string(R2, precision=4, separator=', ') + "\n\n")
        f.write("Projection Matrix (P1):\n")
        f.write(np.array2string(P1, precision=4, separator=', ') + "\n\n")
        f.write("Projection Matrix (P2):\n")
        f.write(np.array2string(P2, precision=4, separator=', ') + "\n\n")
        f.write("Q Matrix (Q):\n")
        f.write(np.array2string(Q, precision=4, separator=', ') + "\n\n")

        print("Stereo calibration complete. Calibration data saved in 'stereo_calibration.npz' and 'calibration_results.txt'.")

calibrate_cameras()

# Charger les paramètres de calibration stéréo depuis le fichier .npz
calibration_data = np.load('stereo_calibration.npz')
mtx_left = calibration_data['mtx_left']
dist_left = calibration_data['dist_left']
mtx_right = calibration_data['mtx_right']
dist_right = calibration_data['dist_right']
R = calibration_data['R']
T = calibration_data['T']
R1 = calibration_data['R1']
R2 = calibration_data['R2']
P1 = calibration_data['P1']
P2 = calibration_data['P2']
Q = calibration_data['Q']

# Baseline entre les caméras (distance sur l'axe X)
baseline = T[0, 0]

# Charger le modèle YOLO (v5 small)
model = YOLO("yolov5su.pt")  # Charger le modèle pré-entraîné YOLOv5s

# Configurer les flux vidéo
cap_left = cv2.VideoCapture(2)
cap_right = cv2.VideoCapture(3)

if not (cap_left.isOpened() and cap_right.isOpened()):
    print("Erreur : Impossible d'ouvrir les flux des caméras.")
    exit()

# Paramètres pour la détection
threshold = 0.5
target_class = 39  # Détecter la classe 39 (selon le modèle YOLOv5)

print("Appuyez sur 'q' pour quitter.")

while True:
    # Lire les images des deux caméras
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not (ret_left and ret_right):
        print("Erreur : Impossible de capturer les images.")
        break

    # Redimensionner les images pour le modèle
    frame_left_resized = cv2.resize(frame_left, (640, 480))
    frame_right_resized = cv2.resize(frame_right, (640, 480))

    # Préparer les images pour la détection
    frame_left_rgb = cv2.cvtColor(frame_left_resized, cv2.COLOR_BGR2RGB)
    frame_right_rgb = cv2.cvtColor(frame_right_resized, cv2.COLOR_BGR2RGB)

    # Detect objects directly for the target class
    results_left = model(frame_left, classes=[target_class])
    results_right = model(frame_right, classes=[target_class])

    # Extract bounding boxes from results
    boxes_left = []
    boxes_right = []

    for result, boxes in [(results_left, boxes_left), (results_right, boxes_right)]:
        for box in result[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()

            if confidence > threshold:
                # Append box to the corresponding list
                boxes.append([x1, y1, x2, y2])

    # # Draw bounding boxes for detected objects
    # for box in boxes_left:
    #     x1, y1, x2, y2 = map(int, box)
    #     cv2.rectangle(frame_left_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for left camera

    # for box in boxes_right:
    #     x1, y1, x2, y2 = map(int, box)
    #     cv2.rectangle(frame_right_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for right camera

    # Si un objet est détecté dans les deux caméras
    if len(boxes_left) > 0 and len(boxes_right) > 0:
        # Utiliser le premier objet détecté
        box_left = boxes_left[0]
        box_right = boxes_right[0]

        x1_l, y1_l, x2_l, y2_l = box_left
        x1_r, y1_r, x2_r, y2_r = box_right

        ul, vl = (x1_l + x2_l) / 2, (y1_l + y2_l) / 2  # Center of the object in the left frame
        ur, vr = (x1_r + x2_r) / 2, (y1_r + y2_r) / 2  # Center of the object in the right frame

        # Display 2D coordinates
        print(f"2D Coordinates (Left Camera): ({ul:.2f}, {vl:.2f})")
        print(f"2D Coordinates (Right Camera): ({ur:.2f}, {vr:.2f})")


        # Dessiner les rectangles des objets détectés
        cv2.rectangle(frame_left_resized, (int(x1_l), int(y1_l)), (int(x2_l), int(y2_l)), (255, 0, 0), 2)
        cv2.rectangle(frame_right_resized, (int(x1_r), int(y1_r)), (int(x2_r), int(y2_r)), (0, 0, 255), 2)

        # Convertir en format numpy pour la triangulation
        pts_left = np.array([ul, vl]).reshape(1, 1, 2)
        pts_right = np.array([ur, vr]).reshape(1, 1, 2)

        # Rectification et undistortion des points
        pts_left_undist = cv2.undistortPoints(pts_left, mtx_left, dist_left, R=R1, P=P1)
        pts_right_undist = cv2.undistortPoints(pts_right, mtx_right, dist_right, R=R2, P=P2)

        # Calculer les coordonnées 3D à partir des équations
        ul, vl = pts_left_undist[0, 0]
        ur, vr = pts_right_undist[0, 0]

        disparity = ul - ur
        if abs(disparity) > 1e-6:  # Éviter la division par zéro
            z = (baseline * mtx_left[0, 0]) / abs(disparity)
            x = (ul - mtx_left[0, 2]) * z / mtx_left[0, 0]
            y = (vl - mtx_left[1, 2]) * z / mtx_left[1, 1]

            # Coordonnées 3D dans le repère de la caméra gauche
            P_cam = np.array([x, y, z])
            print(f"Position 3D de l'objet dans le repère de la caméra gauche : x={x:.2f}, y={y:.2f}, z={z:.2f}")
            
            #Effectue une transformation de repère. Elle permet de passer des coordonnées dans le repère de la caméra gauche P_cam
            #au repère du chessboard Pcb
            P_cb = R @ P_cam + T

            x_cb, y_cb, z_cb = P_cb[:, 0]  # Premier point
            print(f"Position 3D de l'objet dans le repère du chessboard : x={x_cb:.2f}, y={y_cb:.2f}, z={z_cb:.2f}")
            
            # Calcul du point médian dans le repère de la caméra gauche
            midpoint_cam = np.array([baseline / 2, 0, 0])
            print(f"Point médian entre les caméras : {midpoint_cam}")
            
            # Transformation du point médian vers le repère du chessboard
            midpoint_cb = R @ midpoint_cam + T
            # Calcul des moyennes
            x_median = np.mean(P_cb[0, :])
            y_median = np.mean(P_cb[1, :])
            z_median =  abs(np.mean(P_cb[2, :]))

            # Affichage du point médian
            print(f"Point médian dans le repère du chessboard : x={x_median:.2f}, y={y_median:.2f}, z={z_median:.2f}")

            # Afficher les coordonnées sur l'image de la caméra gauche
            cv2.putText(frame_left_resized, f"3D Chessboard: {x_cb:.2f}, {y_cb:.2f}, {z_cb:.2f}", 
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.putText(frame_left_resized, f"Midpoint CB: {x_median:.2f}, {y_median:.2f}, {z_median:.2f}",
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            # Afficher le point médian sur l'image
            cv2.putText(frame_left_resized, f"3D camera: {x:.2f}, {y:.2f}, {z:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_left_resized, f"Midpoint camera: {midpoint_cam[0]:.2f}, {midpoint_cam[1]:.2f}, {midpoint_cam[2]:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Mark the centers on the frames
            cv2.circle(frame_left, (int(ul), int(vl)), 5, (0, 255, 0), -1)
            cv2.putText(frame_left, f"({ul:.2f}, {vl:.2f})", (int(ul) + 10, int(vl)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.circle(frame_right, (int(ur), int(vr)), 5, (255, 0, 0), -1)
            cv2.putText(frame_right, f"({ur:.2f}, {vr:.2f})", (int(ur) + 10, int(vr)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            print("Erreur de triangulation : disparity trop faible.")
        
    # Mettre à jour les images de suivi
    prev_left_gray = cv2.cvtColor(frame_left_resized, cv2.COLOR_BGR2GRAY)
    prev_right_gray = cv2.cvtColor(frame_right_resized, cv2.COLOR_BGR2GRAY)

    # Afficher les flux vidéo
    cv2.imshow('Camera Gauche', frame_left_resized)
    cv2.imshow('Camera Droite', frame_right_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()


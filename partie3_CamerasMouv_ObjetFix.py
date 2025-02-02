import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Désactiver les logs de la bibliothèque Ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load the YOLO model
model = YOLO("yolov5su.pt")

# Checkerboard parameters
CHECKERBOARD = (9, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)*2

# Configure cameras
cap_left = cv2.VideoCapture(2)
cap_right = cv2.VideoCapture(3)

if not (cap_left.isOpened() and cap_right.isOpened()):
    print("Error: Unable to open cameras.")
    exit()

# Variables for motion detection
prev_frame_left = None
prev_frame_right = None
motion_threshold = 5000  # Tolerance for detecting significant motion
motion_detected = False
consecutive_motion_frames = 0
motion_frames_threshold = 5  # Minimum consecutive frames with motion before recalibration

# Paramètres pour la détection
threshold = 0.5
target_class = 39  # Détecter la classe 39 (selon le modèle YOLOv5)

# Charger les paramètres de calibration stéréo depuis le fichier .npz
calibration_data = np.load('stereo_calibration.npz')
# Calibration storage
last_mtx_left = calibration_data['mtx_left']
last_dist_left = calibration_data['dist_left']
last_mtx_right = calibration_data['mtx_right']
last_dist_right = calibration_data['dist_right']
last_R = calibration_data['R']
last_T = calibration_data['T']

print("Press 'q' to quit.")

# Main loop
while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not (ret_left and ret_right):
        print("Error: Unable to capture frames.")
        break

    # Convert frames to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Detect motion
    if prev_frame_left is not None and prev_frame_right is not None:
        diff_left = cv2.absdiff(prev_frame_left, gray_left)
        diff_right = cv2.absdiff(prev_frame_right, gray_right)

        motion_score_left = np.sum(diff_left > 25)  # Count significant pixel changes
        motion_score_right = np.sum(diff_right > 25)

        if motion_score_left > motion_threshold or motion_score_right > motion_threshold:
            consecutive_motion_frames += 1
        else:
            consecutive_motion_frames = 0

        # Check if motion persists over several frames
        if consecutive_motion_frames >= motion_frames_threshold:
            motion_detected = True
        else:
            motion_detected = False

    prev_frame_left = gray_left
    prev_frame_right = gray_right

    # Detect chessboard
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

    if ret_left and ret_right:
        # Refine corners and calibrate
        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

        frame_left = cv2.drawChessboardCorners(frame_left, CHECKERBOARD, corners2_left, ret_left)
        frame_right = cv2.drawChessboardCorners(frame_right, CHECKERBOARD, corners2_right, ret_right)

        # Calibrate cameras dynamically if motion is detected
        if motion_detected:
            objpoints = [objp]
            imgpoints_left = [corners_left]
            imgpoints_right = [corners_right]

            _, mtx_left, dist_left, R_left, T_left = cv2.calibrateCamera(
                objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
            _, mtx_right, dist_right, R_right, T_right = cv2.calibrateCamera(
                objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

            flags = cv2.CALIB_FIX_INTRINSIC
            _, mtx_left, dist_left, mtx_right, dist_right, R, T, _, _ = cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, 
                mtx_right, dist_right, gray_left.shape[::-1], criteria=criteria, flags=flags)
            
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], R, T
            )

            last_mtx_left = mtx_left
            last_dist_left = dist_left
            last_R_left = R_left
            last_T_left = T_left
            last_mtx_right = mtx_right
            last_dist_right = dist_right
            last_R_right = R_right
            last_T_right = T_right
            last_R = R
            last_T = T

            print("Motion detected! Recalibrating...")
        else:
            print("Chessboard detected, but no motion. Skipping recalibration.")

    elif motion_detected:
        print("Motion detected! Chessboard not visible. Skipping calibration.")

    else:
        print("No motion and chessboard not visible. Using last known calibration.")
    
    baseline = last_T[0,0]
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
        pts_left_undist = cv2.undistortPoints(pts_left, last_mtx_left, last_dist_left, R=R1, P=P1)
        pts_right_undist = cv2.undistortPoints(pts_right, last_mtx_right, last_dist_right, R=R2, P=P2)

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
            z_median = np.mean(P_cb[2, :])

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

# Release resources
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

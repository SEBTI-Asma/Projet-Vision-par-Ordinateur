import cv2
import numpy as np
import glob
import time

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
cap_left = cv2.VideoCapture(0)  # ID de la caméra gauche
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

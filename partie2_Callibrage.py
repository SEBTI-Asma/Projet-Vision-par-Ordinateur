import cv2
import numpy as np

# OpenCV VideoCapture to access the iVCam or local camera (camera index 3, modify as needed)
cap = cv2.VideoCapture(3)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Checkerboard dimensions (number of inner corners per row and column)
checkerboard = (9, 7)  # Modify based on your checkerboard

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Prepare object points (like (0,0,0), (1,0,0), (2,0,0), ...)
objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

print("Press 'c' to capture the checkerboard or 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    # Rotate the frame 90 degrees clockwise
    (h, w) = frame.shape[:2]  # Dimensions de l'image
    center = (w // 2, h // 2)  # Centre de rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, -90, 1.0)  # Rotation de 90°
    frame = cv2.warpAffine(frame, rotation_matrix, (w, h))  # Appliquer la rotation

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

    if ret:
        # Refine corner locations for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        frame = cv2.drawChessboardCorners(frame, checkerboard, corners2, ret)

        # Show the frame with detected corners
        cv2.imshow('Real-Time Chessboard Calibration', frame)

        # Press 'c' to capture the current frame
        if cv2.waitKey(1) & 0xFF == ord('c'):
            objpoints.append(objp)
            imgpoints.append(corners2)
            print("Checkerboard captured!")
    else:
        # Show the original frame
        cv2.imshow('Real-Time Chessboard Calibration', frame)

    # Press 'q' to quit the calibration process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calibration after capturing multiple images
if len(objpoints) > 0:
    print("Calibrating the camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        print("\nCamera calibration successful!")
        print("Results are being saved in 'calibration_results.txt'...")

        # Open a file to save the results
        with open("calibration_results.txt", "w") as file:
            file.write("Camera Calibration Results\n")
            file.write("===========================\n\n")

            # Camera matrix
            file.write("1. Camera Matrix \n")
            file.write(f"{mtx}\n\n")

            # Distortion coefficients
            file.write("2. Distortion Coefficients:\n")
            file.write(f"{dist}\n\n")

            # Rotation vectors
            file.write("3. Rotation Vectors \n")
            for i, rvec in enumerate(rvecs):
                file.write(f"Image {i + 1}: {np.array2string(rvec.ravel(), precision=4)}\n")

            file.write("\n")

            # Translation vectors
            file.write("4. Translation Vectors \n")
            for i, tvec in enumerate(tvecs):
                file.write(f"Image {i + 1}: {np.array2string(tvec.ravel(), precision=4)}\n")

        print("Calibration results have been saved successfully!")
    else:
        print("Calibration failed. Make sure enough checkerboard images were captured.")
else:
    print("No checkerboards were captured. Calibration not possible.")
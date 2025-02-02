import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
from ultralytics import YOLO
from collections import deque
import time

# Charger les paramètres de calibration stéréo depuis le fichier .npz
calibration_data = np.load('stereo_calibration.npz')

mtx_left = calibration_data['mtx_left']
dist_left = calibration_data['dist_left']
mtx_right = calibration_data['mtx_right']
dist_right = calibration_data['dist_right']
R = calibration_data['R']
T = calibration_data['T']

baseline = T[0, 0]

# Charger le modèle YOLOv8 pour la détection d'objets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov5su.pt')  # Utiliser le modèle nano qui est le plus rapide
target_class = 39  # ID de classe pour "bottle" dans COCO

# Configurer les flux vidéo
cap_left = cv2.VideoCapture(2)
cap_right = cv2.VideoCapture(3)

if not (cap_left.isOpened() and cap_right.isOpened()):
    print("Erreur : Impossible d'ouvrir les flux des caméras.")
    exit()

# Transformation pour le modèle
transform = transforms.Compose([transforms.ToTensor()])

# Paramètres pour la détection
threshold = 0.5

print("Appuyez sur 'q' pour quitter.")

# Position tracking variables
last_known_position = None
position_history = deque(maxlen=10)  # Store last 10 positions
last_velocity = np.zeros(3)  # Store velocity for prediction
last_update_time = None
prediction_timeout = 10.0  # Increase prediction timeout to 5 seconds
predicted_positions = deque(maxlen=5)  # Store last 5 predicted positions

# Add SGBM parameters
window_size = 11
min_disp = 0
num_disp = 128 - min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Add function to compute depth from disparity
def compute_depth_map(left_img, right_img):
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Compute disparity
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # Calculate depth map
    depth_map = (baseline * mtx_left[0, 0]) / (disparity + 1e-7)  # Avoid division by zero
    
    # Normalize depth map for visualization
    depth_map_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map_color = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_JET)
    
    return depth_map, depth_map_color

def get_3d_point_from_depth(x, y, depth_map):
    Z = depth_map[y, x]
    # Convert image coordinates to world coordinates
    X = (x - mtx_left[0, 2]) * Z / mtx_left[0, 0]
    Y = (y - mtx_left[1, 2]) * Z / mtx_left[1, 1]
    # Calculate Euclidean distance
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    return np.array([X, Y, Z]), distance  # Return both position and distance

def predict_position(current_time):
    global last_known_position, last_velocity, last_update_time
    
    if last_known_position is None or last_update_time is None:
        return None, None  # Return None for both position and distance
    
    # If too much time has passed, stop predicting
    if current_time - last_update_time > prediction_timeout:
        return None, None
    
    # Calculate time delta
    dt = current_time - last_update_time
    
    # Predict new position based on last known position and velocity
    predicted_position = last_known_position + last_velocity * dt
    predicted_distance = np.sqrt(np.sum(predicted_position**2))  # Calculate distance for predicted position
    
    return predicted_position, predicted_distance

def update_tracking(new_position):
    global last_known_position, last_velocity, last_update_time, position_history
    
    current_time = time.time()
    
    if last_known_position is not None and last_update_time is not None:
        # Calculate velocity
        dt = current_time - last_update_time
        if dt > 0:
            velocity = (new_position - last_known_position) / dt
            # Smooth velocity using exponential moving average
            alpha = 0.3
            last_velocity = alpha * velocity + (1 - alpha) * last_velocity
    
    last_known_position = new_position
    last_update_time = current_time
    position_history.append(new_position)

def draw_3d_marker(image, position, distance, color=(0, 255, 0), is_predicted=False):
    x, y, z = position
    scale = 1000
    # Ensure coordinates are within image bounds
    image_x = int(x * scale / z + image.shape[1] / 2)
    image_y = int(y * scale / z + image.shape[0] / 2)

    # Different appearance for predicted positions
    if is_predicted:
        color = (0, 165, 255)  # Orange for predicted positions
        cv2.putText(image, "Predicted Position", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw crosshair marker
    size = max(20, int(50 / z))  # Adjust size based on distance
    cv2.line(image, (image_x - size, image_y), (image_x + size, image_y), color, 2)
    cv2.line(image, (image_x, image_y - size), (image_x, image_y + size), color, 2)
    
    # Draw position information
    cv2.putText(image, f"X: {x:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(image, f"Y: {y:.2f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(image, f"Z: {z:.2f}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(image, f"Distance: {distance:.2f}", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
    frame_left_tensor = transform(cv2.cvtColor(frame_left_resized, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    frame_right_tensor = transform(cv2.cvtColor(frame_right_resized, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    with torch.no_grad():
        results_left = model(frame_left_resized, verbose=False)
        results_right = model(frame_right_resized, verbose=False)

    # Extraire les détections
    boxes_left = []
    boxes_right = []
    
    # Filtrer les bouteilles dans l'image gauche
    for r in results_left:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if int(cls) == target_class and conf > threshold:
                boxes_left.append(box.cpu().numpy())
    
    # Filtrer les bouteilles dans l'image droite
    for r in results_right:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if int(cls) == target_class and conf > threshold:
                boxes_right.append(box.cpu().numpy())
    
    boxes_left = np.array(boxes_left)
    boxes_right = np.array(boxes_right)

    current_time = time.time()
    object_detected = False

    # Compute depth map
    depth_map, depth_map_color = compute_depth_map(frame_left_resized, frame_right_resized)
    
    if len(boxes_left) > 0:
        # Get the center of the detected object in left image
        box_left = boxes_left[0]
        x1_l, y1_l, x2_l, y2_l = box_left.tolist()
        center_x = int((x1_l + x2_l) / 2)
        center_y = int((y1_l + y2_l) / 2)
        
        # Calculate 3D position and distance
        position_3d, distance = get_3d_point_from_depth(center_x, center_y, depth_map)
        
        # Update tracking with the new 3D position
        update_tracking(position_3d)
        
        # Draw markers and trajectory with distance
        draw_3d_marker(frame_left_resized, position_3d, distance, color=(0, 255, 0), is_predicted=False)
        
        print(f"Distance to object: {distance:.2f}")  # Debug print
        
        # Draw trajectory
        if len(position_history) > 1:
            for i in range(len(position_history)-1):
                pt1 = position_history[i]
                pt2 = position_history[i+1]
                
                # Convert 3D points to 2D image coordinates
                x1 = int(pt1[0] * 1000 / pt1[2] + frame_left_resized.shape[1]/2)
                y1 = int(pt1[1] * 1000 / pt1[2] + frame_left_resized.shape[0]/2)
                x2 = int(pt2[0] * 1000 / pt2[2] + frame_left_resized.shape[1]/2)
                y2 = int(pt2[1] * 1000 / pt2[2] + frame_left_resized.shape[0]/2)
                
                # Draw lines with decreasing intensity
                alpha = (i + 1) / len(position_history)
                color = (int(255 * alpha), 0, 0)
                cv2.line(frame_left_resized, (x1, y1), (x2, y2), color, 2)
    else:
        # When object is not detected, predict position
        predicted_result = predict_position(current_time)
        if predicted_result is not None:
            predicted_position, predicted_distance = predicted_result
            if predicted_position is not None:
                # Add the predicted position to our predicted positions queue
                predicted_positions.append(predicted_position)
                
                # Draw the current predicted marker
                draw_3d_marker(frame_left_resized, predicted_position, predicted_distance, 
                             color=(0, 165, 255), is_predicted=True)
                
                # Draw predicted trajectory
                if len(predicted_positions) > 1:
                    for i in range(len(predicted_positions)-1):
                        pt1 = predicted_positions[i]
                        pt2 = predicted_positions[i+1]
                        
                        # Convert 3D points to 2D image coordinates
                        x1 = int(pt1[0] * 1000 / pt1[2] + frame_left_resized.shape[1]/2)
                        y1 = int(pt1[1] * 1000 / pt1[2] + frame_left_resized.shape[0]/2)
                        x2 = int(pt2[0] * 1000 / pt2[2] + frame_left_resized.shape[1]/2)
                        y2 = int(pt2[1] * 1000 / pt2[2] + frame_left_resized.shape[0]/2)
                        
                        # Draw predicted trajectory lines in bright orange with increased thickness
                        alpha = (i + 1) / len(predicted_positions)
                        color = (0, 255, 255)  # Bright yellow color for better visibility
                        cv2.line(frame_left_resized, (x1, y1), (x2, y2), color, 4)  # Increased thickness

    # Display frames including depth map
    depth_display = cv2.resize(depth_map_color, (frame_left_resized.shape[1], frame_left_resized.shape[0]))
    combined_frame = np.hstack((frame_left_resized, frame_right_resized, depth_display))
    cv2.imshow('Stereo View with Depth', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        last_known_position = None
        last_velocity = np.zeros(3)
        last_update_time = None
        position_history.clear()
        predicted_positions.clear()

# Cleanup
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
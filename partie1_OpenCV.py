import cv2
import numpy as np
import sys

# Fonction pour détecter l'objet basé sur la couleur

def detect_object_by_color(image_path, lower_color, upper_color):
    # Vérifiez si les limites de couleur sont valides
    frame = cv2.imread(image_path)

    if lower_color is None or upper_color is None:
        print("Erreur : Les limites de couleur (lower_color, upper_color) ne sont pas définies.")
        return frame

    # Convertir l'image en espace de couleur HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Appliquer un masque basé sur la plage de couleurs
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Trouver les contours dans le masque
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Ajuster ce seuil si nécessaire
            continue

        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        # Dessiner un rectangle et un point central
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # Afficher les coordonnées du centre sur l'image
        cv2.putText(frame, f"Centre: ({center[0]}, {center[1]})", (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame
    return frame
# Fonction pour détecter l'objet basé sur la forme
def detect_object_by_shape(image_path, selected_shape):
    # Load the image
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Could not load the image.")
        return

    # Convert the frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=150,
    )

    if circles is not None:
        # Convert the coordinates and radius to integers
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the outer circle (green)
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            # Draw the center of the circle (red)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), 5)
            # Afficher les coordonnées du centre
            cv2.putText(frame, f"Centre: ({x}, {y})", (x + 10, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        shape_name = "Cercle"
    else:
        # Detect edges using Canny for shape detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Identify shapes based on the number of vertices
            vertices = len(approx)
            print(vertices)
            x, y, w, h = cv2.boundingRect(approx)
            cx, cy = x + w // 2, y + h // 2  # Center of the bounding rectangle

            # Check if the shape is a triangle
            if vertices == 3:
                shape_name = "Triangle"
            elif vertices == 4:
                aspect_ratio = w / float(h)
                shape_name = "Carre" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif vertices == 5:
                shape_name = "Pentagon"
            elif vertices == 6:
                shape_name = "Hexagon"
            elif vertices == 7:
                shape_name = "Heptagon"
            elif vertices == 8:
                shape_name = "Octagon"
            else:
                shape_name = "Polygon"

            # Annotate the shape on the frame
            cv2.putText(frame, shape_name, (cx - 50, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw the contours
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

            # Draw the center of the shape (red dot)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # -1 to fill the circle

            # Afficher les coordonnées du centre
            cv2.putText(frame, f"Centre: ({cx}, {cy})", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if shape_name==selected_shape:

        return frame

# Fonction principale pour gérer la logique
def main(image_path, lower_color, upper_color, object_color, selected_shape):

    if object_color == "Couleur" and object_shape != "Forme":
        # Détection par forme uniquement
        processed_frame = detect_object_by_shape(image_path, selected_shape)
        print("shape seulement")
    elif object_shape == "Forme" and object_color != "Couleur":
        # Détection par couleur uniquement
        print("couleur seulement")
        processed_frame = detect_object_by_color(image_path, lower_color, upper_color)

    cv2.imshow("Détection d'Objet", processed_frame)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Exécution du script
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python detecter_Objet_OpenCV.py <object_color> <object_shape>")
        sys.exit(1)

    object_color = sys.argv[1]
    object_shape = sys.argv[2]
    image_path = sys.argv[3]
    print(object_color)
    print(object_shape)
    print(image_path)

    color_ranges = {
        "Rouge": (np.array([0, 150, 50]), np.array([10, 255, 255])),
        "Vert": (np.array([40, 100, 50]), np.array([80, 255, 255])),
        "Bleu": (np.array([100, 150, 50]), np.array([140, 255, 255])),
        "Jaune": (np.array([20, 100, 100]), np.array([40, 255, 255])),  # Plage pour Jaune
        "Orange": (np.array([10, 100, 100]), np.array([20, 255, 255]))  # Plage pour Orange
    }

    if object_color not in color_ranges and object_color != "Couleur":
        print(f"Couleur '{object_color}' non reconnue. Utilisez Rouge, Vert, Bleu ou Couleur.")
        sys.exit(1)

    # Définir les limites de couleur
    lower_color, upper_color = color_ranges.get(object_color, (None, None))

    main(image_path, lower_color, upper_color, object_color, object_shape)


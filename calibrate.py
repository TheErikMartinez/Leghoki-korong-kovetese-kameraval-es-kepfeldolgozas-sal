import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Coordinates of the click are stored in the points list
        points.append((x, y))
        print(f"Click registered: ({x}, {y})")
        
        # Draw a circle and the point number on the image for visual feedback
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, str(len(points)), (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Calibration', img)

cap = cv2.VideoCapture('test_video.mp4')
ret, img = cap.read()

if not ret:
    print("Error: Failed to read video frame.")
    exit()

print("TABLE CALIBRATION: ")
print("Click on the 4 corners of the table (where the puck can bounce) in this order:")
print("1. Top-left corner")
print("2. Top-right corner")
print("3. Bottom-right corner")
print("4. Bottom-left corner")
print("After clicking all 4 points, press any key to finish calibration!")

# Set up the window and mouse callback for calibration
cv2.imshow('Calibration', img)
cv2.setMouseCallback('Calibration', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

if len(points) == 4:
    np.save("calibration.npy", np.array(points, dtype=np.float32))
    print("Calibration saved to calibration.npy")
else:
    print(f"Warning: {len(points)} point(s) selected, need exactly 4. Not saved.")

print("\nCalibration points saved to 'calibration.npy'.")

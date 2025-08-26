from ultralytics import YOLO
import cv2

# --- DEFINE YOUR MODEL AND IMAGE HERE ---
MODEL_PATH = 'runs/detect/yolov8n_crop_weed_model3/weights/best.pt'
IMAGE_PATH = 'C:/Users/surya/Downloads/image5.jpeg'  # <-- IMPORTANT: Change this to your image path!
# -----------------------------------------

# Load your custom-trained YOLOv8 model
model = YOLO(MODEL_PATH)

# Run inference on the image
results = model(IMAGE_PATH)

# The 'results' object contains the processed image with detections
# We can display it directly using its plot() method
annotated_image = results[0].plot()

# Display the image with detections
cv2.imshow("Crop and Weed Detection", annotated_image)
cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()

print(f"Prediction complete. Results displayed for '{IMAGE_PATH}'.")
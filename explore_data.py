import cv2
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt

def get_yolo_labels(label_path):
    """Reads a YOLO format label file and returns a list of labels."""
    labels = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                # Bbox coords are normalized: x_center, y_center, width, height
                bbox = [float(p) for p in parts[1:]]
                labels.append((class_id, bbox))
    return labels

def draw_boxes(image, labels, class_names):
    """Draws bounding boxes on an image."""
    h, w, _ = image.shape
    for class_id, bbox in labels:
        x_center, y_center, width, height = bbox
        
        # Denormalize coordinates
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        # Get class name and color
        label_name = class_names.get(class_id, "Unknown")
        color = (0, 255, 0) if label_name == 'crop' else (0, 0, 255) # Green for crop, Red for weed
        
        # Draw rectangle and text
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    return image

# --- Main Script ---
# Define class names mapping
CLASS_NAMES = {0: 'crop', 1: 'weed'} # Adjust if your class IDs are different

# Define paths
base_path = Path('datasets/crop-weed-data/')
image_dir = base_path / 'images/train'
label_dir = base_path / 'labels/train'

# Get a list of all images
image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.png'))


# Display 4 random images
plt.figure(figsize=(12, 12))
for i in range(4):
    # Choose a random image
    image_path = random.choice(image_files)
    label_path = label_dir / f"{image_path.stem}.txt"

    # Read image and labels
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert for matplotlib
    labels = get_yolo_labels(label_path)

    # Draw bounding boxes
    image_with_boxes = draw_boxes(image.copy(), labels, CLASS_NAMES)
    
    # Display the image
    plt.subplot(2, 2, i + 1)
    plt.imshow(image_with_boxes)
    plt.title(image_path.name)
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Displayed 4 random images from '{image_dir}'. Check if boxes and labels are correct.")
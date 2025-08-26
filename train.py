from ultralytics import YOLO

# Load a pre-trained YOLOv8 model. 'yolov8n.pt' is the smallest and fastest version.
# The model will download automatically on the first run.
model = YOLO('yolov8n.pt') 

# Train the model using your custom dataset
if __name__ == '__main__':
    results = model.train(
        data='datasets/crop-weed-data/data.yaml',  # Path to your data.yaml file
        epochs=50,                                # Number of training epochs (50 is a good start)
        imgsz=512,                                # Image size (should match your preprocessed images)
        batch=16,                                 # Number of images to process at a time
        name='yolov8n_crop_weed_model'            # A name for your trained model folder
    )

    print("Training complete!")
    print("Your trained model and results are saved in the 'runs/detect/' directory.")
from ultralytics import YOLO
import cv2
import os

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "runs/detect/train/weights/best.pt"
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "output/all_plate_crops"

# Your 12 vehicle images
IMAGE_FILES = [
    "bus1.jpg",
    "car.jpg",
    "car2.jpg",
    "car3.jpg",
    "car4.jpg",
    "car5.jpg",
    "car6.jpg",
    "car7.jpg",
    "car8.jpg",
    "car9.jpg",
    "car10.jpg",
    "truck1.jpg"
]

# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# LOAD YOLO
# ==========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO loaded successfully!")

# ==========================================
# PROCESS ALL IMAGES
# ==========================================

total_images = 0
detected_images = 0
failed_images = 0

for image_name in IMAGE_FILES:

    print("\n==========================================")
    print("Processing:", image_name)
    print("==========================================")

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    # Check image exists
    if not os.path.exists(image_path):

        print("ERROR: Image not found!")
        failed_images += 1
        continue

    total_images += 1

    # Read image
    image = cv2.imread(image_path)

    if image is None:

        print("ERROR: Could not read image!")
        failed_images += 1
        continue

    # YOLO detection
    results = model(image)

    plate_found = False
    plate_count = 0

    for result in results:

        if result.boxes is None:
            continue

        for i, box in enumerate(result.boxes):

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(
                box.conf[0]
            )

            # Make sure coordinates are inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            # Crop number plate
            plate = image[y1:y2, x1:x2]

            if plate.size == 0:

                print("Invalid plate crop")
                continue

            # Save filename
            base_name = os.path.splitext(
                image_name
            )[0]

            output_name = (
                f"{base_name}_plate_{plate_count}.jpg"
            )

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_name
            )

            cv2.imwrite(
                output_path,
                plate
            )

            print("Plate detected!")
            print(
                "Confidence:",
                round(confidence * 100, 2),
                "%"
            )

            print(
                "Coordinates:",
                x1,
                y1,
                x2,
                y2
            )

            print(
                "Saved:",
                output_path
            )

            plate_count += 1
            plate_found = True

    if plate_found:

        detected_images += 1

    else:

        print("No number plate detected!")

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n")
print("==========================================")
print("       ALL PLATE CROPS COMPLETED")
print("==========================================")

print(
    "Total images processed :",
    total_images
)

print(
    "Images with plate      :",
    detected_images
)

print(
    "Images without plate   :",
    total_images - detected_images
)

print(
    "File/read failures     :",
    failed_images
)

print("==========================================")

print("\nSaved crops in:")

print(OUTPUT_FOLDER)
from ultralytics import YOLO
import cv2
import easyocr

# Load trained number plate model
model = YOLO("runs/detect/train/weights/best.pt")

# Load vehicle image
image = cv2.imread("images/car2.jpg")

if image is None:
    print("Image not found!")
    exit()

# YOLO detection
results = model(image)

# OCR reader
reader = easyocr.Reader(['en'], gpu=False)

for box in results[0].boxes:

    # Get plate coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Crop plate
    plate = image[y1:y2, x1:x2]

    # Resize for OCR
    plate = cv2.resize(
        plate,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Convert to grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # OCR
    results_ocr = reader.readtext(
        processed,
        detail=0,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    plate_number = "".join(results_ocr)

    print("--------------------------------")
    print("Number Plate Detected")
    print("Plate Number:", plate_number)
    print("--------------------------------")

    # Draw box and text
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        plate_number,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Processed Number Plate", processed)

# Show final result
cv2.imshow("Final ANPR Result", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
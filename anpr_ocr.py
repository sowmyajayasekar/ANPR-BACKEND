from ultralytics import YOLO
import easyocr
import cv2
import os
import re


# ==========================================
# SETTINGS
# ==========================================

IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "output"

CONFIDENCE_THRESHOLD = 0.80


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(
    "runs/detect/train/weights/best.pt"
)

print("YOLO loaded successfully!")


# ==========================================
# LOAD EASY OCR
# ==========================================

print("Loading EasyOCR...")

reader = easyocr.Reader(["en"])

print("EasyOCR loaded successfully!")


# ==========================================
# FIND VEHICLE IMAGES
# ==========================================

image_files = sorted([
    file
    for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])

print("\nImages found:", image_files)


# ==========================================
# PROCESS EACH IMAGE
# ==========================================

for image_file in image_files:

    print("\n\n========================================")
    print("PROCESSING:", image_file)
    print("========================================")

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_file
    )

    # ======================================
    # LOAD IMAGE
    # ======================================

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Could not read image")
        continue

    print("Vehicle image loaded!")

    # ======================================
    # YOLO DETECTION
    # ======================================

    results = model(image)

    print("YOLO detection completed!")

    plate_found = False

    # ======================================
    # PROCESS DETECTED PLATE
    # ======================================

    for result in results:

        for box in result.boxes:

            plate_found = True

            # ==================================
            # COORDINATES
            # ==================================

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            yolo_confidence = float(
                box.conf[0]
            )

            print("\nNumber plate detected!")

            print(
                "YOLO Confidence:",
                round(
                    yolo_confidence * 100,
                    2
                ),
                "%"
            )

            print(
                "Coordinates:",
                x1,
                y1,
                x2,
                y2
            )

            # ==================================
            # CROP NUMBER PLATE
            # ==================================

            plate = image[
                y1:y2,
                x1:x2
            ]

            if plate.size == 0:
                print("Invalid plate crop!")
                continue

            print(
                "Original plate size:",
                plate.shape
            )

            base_name = os.path.splitext(
                image_file
            )[0]

            # Save original crop
            cv2.imwrite(
                f"{OUTPUT_FOLDER}/{base_name}_cropped.jpg",
                plate
            )

            # ==================================
            # GRAYSCALE
            # ==================================

            gray = cv2.cvtColor(
                plate,
                cv2.COLOR_BGR2GRAY
            )

            # ==================================
            # UPSCALE
            # ==================================

            upscaled = cv2.resize(
                gray,
                None,
                fx=5,
                fy=5,
                interpolation=cv2.INTER_CUBIC
            )

            # ==================================
            # CONTRAST ENHANCEMENT
            # ==================================

            enhanced = cv2.equalizeHist(
                upscaled
            )

            # ==================================
            # OTSU THRESHOLD
            # ==================================

            blurred = cv2.GaussianBlur(
                enhanced,
                (3, 3),
                0
            )

            otsu = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            # ==================================
            # ADAPTIVE THRESHOLD
            # ==================================

            adaptive = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            # ==================================
            # MORPHOLOGY
            # ==================================

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 3)
            )

            morphology = cv2.morphologyEx(
                enhanced,
                cv2.MORPH_CLOSE,
                kernel
            )

            print(
                "Multiple preprocessing methods completed!"
            )

            # ==================================
            # SAVE PREPROCESSED IMAGES
            # ==================================

            cv2.imwrite(
                f"{OUTPUT_FOLDER}/{base_name}_upscaled.jpg",
                upscaled
            )

            cv2.imwrite(
                f"{OUTPUT_FOLDER}/{base_name}_enhanced.jpg",
                enhanced
            )

            cv2.imwrite(
                f"{OUTPUT_FOLDER}/{base_name}_otsu.jpg",
                otsu
            )

            cv2.imwrite(
                f"{OUTPUT_FOLDER}/{base_name}_adaptive.jpg",
                adaptive
            )

            cv2.imwrite(
                f"{OUTPUT_FOLDER}/{base_name}_morphology.jpg",
                morphology
            )

            # ==================================
            # OCR METHODS
            # ==================================

            methods = {
                "Upscaled": upscaled,
                "Enhanced": enhanced,
                "Otsu": otsu,
                "Adaptive": adaptive,
                "Morphology": morphology
            }

            print("\n========================================")
            print("             OCR COMPARISON")
            print("========================================")

            all_results = []

            # ==================================
            # RUN OCR
            # ==================================

            for method_name, processed_image in methods.items():

                ocr_result = reader.readtext(
                    processed_image,
                    allowlist=(
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        "0123456789"
                    )
                )

                if not ocr_result:

                    print(
                        method_name,
                        "-> No text detected"
                    )

                    continue

                for detection in ocr_result:

                    text = detection[1]

                    confidence = float(
                        detection[2]
                    )

                    # ==================================
                    # CLEAN TEXT
                    # ==================================

                    clean_text = re.sub(
                        r"[^A-Z0-9]",
                        "",
                        text.upper()
                    )

                    print(
                        method_name,
                        "->",
                        clean_text,
                        "| Confidence:",
                        round(
                            confidence * 100,
                            2
                        ),
                        "%"
                    )

                    # ==================================
                    # FILTER INVALID OCR RESULTS
                    # ==================================

                    if not clean_text:
                        continue

                    # Ignore IND
                    if clean_text == "IND":
                        continue

                    # Ignore very short results
                    if len(clean_text) < 2:
                        continue

                    # Save valid result
                    all_results.append(
                        (
                            clean_text,
                            confidence,
                            method_name
                        )
                    )

            # ==================================
            # FINAL ANPR RESULT
            # ==================================

            print("\n========================================")
            print("             ANPR RESULT")
            print("========================================")

            print(
                "Vehicle Image :",
                image_file
            )

            if all_results:

                # ==================================
                # IMPROVED OCR RESULT SELECTION
                # ==================================

                reliable_results = [
                    result
                    for result in all_results
                    if len(result[0]) >= 4
                ]

                # If there are no 4+ character results,
                # use all valid OCR results
                if not reliable_results:
                    reliable_results = all_results

                # Select highest-confidence result
                best = max(
                    reliable_results,
                    key=lambda x: x[1]
                )

                best_text = best[0]
                best_confidence = best[1]
                best_method = best[2]

                print(
                    "Plate Number  :",
                    best_text
                )

                print(
                    "OCR Confidence:",
                    round(
                        best_confidence * 100,
                        2
                    ),
                    "%"
                )

                print(
                    "Best Method   :",
                    best_method
                )

                # ==================================
                # CONFIDENCE CHECK
                # ==================================

                if best_confidence >= CONFIDENCE_THRESHOLD:

                    print(
                        "Status        : Recognized"
                    )

                else:

                    print(
                        "Status        : Low Confidence"
                    )

            else:

                print(
                    "Plate Number  : Not Recognized"
                )

                print(
                    "OCR Confidence: 0%"
                )

                print(
                    "Best Method   : None"
                )

                print(
                    "Status        : OCR Failed"
                )

            print(
                "========================================"
            )

            # ==================================
            # PROCESS ONLY FIRST DETECTED PLATE
            # ==================================

            break

    # ======================================
    # NO PLATE DETECTED
    # ======================================

    if not plate_found:

        print("\n========================================")
        print("             ANPR RESULT")
        print("========================================")

        print(
            "Vehicle Image :",
            image_file
        )

        print(
            "Plate Number  : Not Detected"
        )

        print(
            "OCR Confidence: 0%"
        )

        print(
            "Status        : Detection Failed"
        )

        print(
            "========================================"
        )


# ==========================================
# COMPLETION
# ==========================================

print("\n\n========================================")
print("       ALL IMAGES PROCESSED")
print("========================================")
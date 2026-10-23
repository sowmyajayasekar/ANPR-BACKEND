from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

image = cv2.imread("images/car.jpg")

print("Image loaded:", image is not None)

results = model(image)
output = results[0].plot()

cv2.imshow("YOLO Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
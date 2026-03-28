from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

results = model("5.jpg")

count = len(results[0].boxes)
print("Số lượng:", count)

annotated = results[0].plot(
	boxes=True,
	labels=True,
	conf=True,
	line_width=1,
	font_size=0.35,
)

cv2.imwrite("test_bbox.jpg", annotated)
cv2.imshow("Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
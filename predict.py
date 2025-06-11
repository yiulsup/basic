import cv2
from ultralytics import YOLO

# 모델 로드 (fine-tuned best.pt)
model = YOLO('./best.pt')

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 또는 /dev/video0

# 라벨 이름 가져오기
class_names = model.names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론 (이미지를 numpy 배열로 직접 입력 가능)
    results = model.predict(source=frame, conf=0.25, verbose=False)

    # 결과 처리
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = r.boxes.conf.cpu().numpy()  # confidence
        classes = r.boxes.cls.cpu().numpy().astype(int)  # class indices

        for box, score, cls_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[cls_id]} {score:.2f}"

            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 프레임 보여주기
    cv2.imshow("YOLOv5 Webcam Detection", frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


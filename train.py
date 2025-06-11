from ultralytics import YOLO

model = YOLO('yolov5s.pt')  # yolov5n.pt, yolov5m.pt, yolov5l.pt 등도 가능

model.train(
    data='/home/soo/github_work/yiulsup/basic/data.yaml',      # 클래스 및 경로 정의 파일
    epochs=100,             # 학습 epoch 수
    imgsz=640,             # 입력 이미지 크기
    batch=16,              # 배치 크기
    name='yolov5s_custom'  # 결과 저장 폴더 이름 (runs/train/...)
)

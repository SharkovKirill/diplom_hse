from ultralytics import YOLO

model = YOLO("yolo11s-cls.pt")

results = model.train(
    data="/home/kirill/diplom_hse/train_models/data_yolo_03_01",
    epochs=100,
    imgsz=640,
    batch=0.9,
)  # , classes=[0,1,2,3,4]

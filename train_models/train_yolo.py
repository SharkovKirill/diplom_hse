import comet_ml
from ultralytics import YOLO
import torch

comet_ml.login(
    project_name="classification_by_aesthetics", api_key="your_api_key"
)

if __name__ == "__main__":
    models = [
        "yolo11n-cls.pt",
        "yolo11s-cls.pt",
        "yolo11m-cls.pt",
        "yolo11l-cls.pt",
        "yolo11x-cls.pt",
    ]
    for model_name in models:
        experiment = comet_ml.Experiment(experiment_name=f"{model_name} 50_epochs 640")
        model = YOLO(model_name)

        torch.multiprocessing.freeze_support()

        result = model.train(
            data=r"D:\MY_PROJECTS\diplom_hse\train_models\prepared_data_2",
            epochs=50,
            imgsz=640,
            batch=0.9,
        )  
        experiment.end()

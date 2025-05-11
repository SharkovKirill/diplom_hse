import os
import random
import shutil
from loguru import logger
from PIL import Image
import json


def process_image(src_path, dst_path, target_size):
    """Жесткий ресайз изображения до 640x480"""
    try:
        img = Image.open(src_path)
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img.save(dst_path)
        return True
    except Exception as e:
        print(f"Ошибка обработки {src_path}: {str(e)}")
        return False


def copy_and_resize_folder(src_folder, dst_folder, target_size):
    """Рекурсивно копирует папку с ресайзом изображений"""
    for root, dirs, files in os.walk(src_folder):
        # Создаем аналогичную структуру папок
        rel_path = os.path.relpath(root, src_folder)
        dest_dir = os.path.join(dst_folder, rel_path)
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dest_dir, file)

            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                process_image(src_path, dst_path, target_size)
            else:
                shutil.copy2(src_path, dst_path)


def make_val_test_samples(
    source_folder: str,
    subsets: list = ["val", "test"],
):

    # Путь к новой папке, куда будут скопированы данные
    for subset in subsets:
        dir_path, _ = os.path.split(source_folder)
        # Заменяем последний сегмент
        destination_folder = os.path.join(
            dir_path, subset
        ) 
        os.makedirs(destination_folder, exist_ok=True)
        shutil.copy2(source_folder + r"\labels.json" , destination_folder + r"\labels.json")

        for root, dirs, files in os.walk(source_folder):
            print(root, dirs, len(files))
            relative_path = os.path.relpath(root, source_folder)
            new_folder = os.path.join(destination_folder, relative_path)
            os.makedirs(new_folder, exist_ok=True)

            if files:
                if root.endswith(r"\1"):
                    photos_to_sample = 10
                elif root.endswith(r"\5"):
                    photos_to_sample = 15
                else:
                    photos_to_sample = int(0.1 * len(files))
                files_to_copy = random.sample(files, min(photos_to_sample, len(files)))

                for file_name in files_to_copy:
                    source_file = os.path.join(root, file_name)
                    destination_file = os.path.join(new_folder, file_name)
                    shutil.move(source_file, destination_file)


def filter_train_dataset(
    sources_train: list,
    dest_train: str,
    classes_to_keep={"1", "2", "3", "4", "5"},
):
    os.makedirs(dest_train, exist_ok=True)

    combined_json = {}

    for source_train in sources_train:
        with open(source_train + "\labels.json", "r") as f:
            combined_json.update(json.load(f))

        for class_name in os.listdir(source_train):
            if class_name in classes_to_keep:
                src_folder = os.path.join(source_train, class_name)
                dest_folder = os.path.join(dest_train, class_name)
                copy_and_resize_folder(src_folder, dest_folder, target_size=(640, 480))

    with open(dest_train + "/labels.json", "w", encoding="utf-8") as f:
        json.dump(combined_json, f)

    print("Dataset filtered successfully!")


if __name__ == "__main__":
    sources_train = [
        r"D:\MY_PROJECTS\diplom_hse\train_models\data_yolo_03_01\train",
        r"D:\MY_PROJECTS\diplom_hse\train_models\data_yolo_05_04\train",
        r"D:\MY_PROJECTS\diplom_hse\train_models\data_yolo_12_04\train", 
    ]
    dest_train = r"D:\MY_PROJECTS\diplom_hse\train_models\prepared_data_2\train"

    filter_train_dataset(sources_train, dest_train)
    make_val_test_samples(dest_train)
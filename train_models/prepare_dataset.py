import os
import random
import shutil
from loguru import logger


def make_val_test_samples(
    source_folder: str = "/home/kirill/diplom_hse/train_models/data_yolo_03_01/train",
    photos_to_sample: int = 5,
    subsets: list = ["val", "test"],
):
    # Путь к новой папке, куда будут скопированы данные
    for subset in subsets:
        dir_path, _ = os.path.split(source_folder)
        # Заменяем последний сегмент
        destination_folder = os.path.join(dir_path, subset)

        os.makedirs(destination_folder, exist_ok=True)

        for root, dirs, files in os.walk(source_folder):
            relative_path = os.path.relpath(root, source_folder)
            new_folder = os.path.join(destination_folder, relative_path)
            os.makedirs(new_folder, exist_ok=True)

            if files:
                files_to_copy = random.sample(files, min(photos_to_sample, len(files)))

                for file_name in files_to_copy:
                    source_file = os.path.join(root, file_name)
                    destination_file = os.path.join(new_folder, file_name)
                    shutil.copy2(source_file, destination_file)
                    # logger.debug(
                    #     f"Скопирован файл: {source_file} -> {destination_file}"
                    # )


def filter_train_dataset(
    source_train: str,
    dest_train: str,
    classes_to_keep={"1", "2", "3", "4", "5"},
):
    os.makedirs(dest_train, exist_ok=True)

    for class_name in os.listdir(source_train):
        if class_name in classes_to_keep:
            src_folder = os.path.join(source_train, class_name)
            dest_folder = os.path.join(dest_train, class_name)
            shutil.copytree(src_folder, dest_folder)

    shutil.copyfile(source_train + "/labels.json", dest_train + "/labels.json")

    print("Dataset filtered successfully!")


if __name__ == "__main__":
    source_train = "/home/kirill/diplom_hse/train_models/data_yolo_03_01/train"
    suffix = "_filtered_data"

    path_parts = source_train.split(os.sep)
    path_parts[-2] += suffix
    dest_train = os.sep.join(path_parts)

    filter_train_dataset(source_train, dest_train)
    make_val_test_samples(dest_train)

from ultralytics import YOLO
import cv2


def get_person_count_and_image(cv2_image):
    model = YOLO("yolo11x.pt")
    model.to("cpu")
    output_image = cv2_image.copy()
    results = model.predict(source=cv2_image, save=False)

    person_count = 0

    for result in results:
        # Фильтруем только детекции людей (класс 0 в COCO)
        for box, cls, conf in zip(
            result.boxes.xyxy, result.boxes.cls, result.boxes.conf
        ):
            if cls == 0:  # Класс 'person'
                person_count += 1
                x1, y1, x2, y2 = map(int, box[:4])

                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                label = f"Person {conf:.2f}"
                cv2.putText(
                    output_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

    return person_count, output_image

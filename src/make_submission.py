import os
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

TEST_IMAGES_DIR = Path(r"Z:\data_cv\VisDrone2019-DET-test-challenge\VisDrone2019-DET-test-challenge\images")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "yolov8m_custom.pt"
OUTPUT_CSV = BASE_DIR / "submission.csv"

def main():
    print("Загрузка обученной YOLOv8 Medium...")
    model = YOLO(str(MODEL_PATH))

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    test_images = [p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in image_extensions]
    
    if not test_images:
        print(f"Изображения в папке {TEST_IMAGES_DIR} не найдены!")
        return

    print(f"Найдено картинок для инференса: {len(test_images)}")

    submission_rows = []

    for img_path in tqdm(test_images, desc="Генерация сабмита"):
        results = model.predict(source=str(img_path), imgsz=1024, conf=0.25, augment=True, verbose=False)
        
        image_id = img_path.name 
        
        boxes = results[0].boxes

        for box in boxes:
            xywh = box.xywh[0].tolist()
            x_center, y_center, width, height = xywh
            
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            submission_rows.append({
                "image_id": image_id,
                "bbox_x": round(x_min, 2),
                "bbox_y": round(y_min, 2),
                "bbox_width": round(width, 2),
                "bbox_height": round(height, 2),
                "score": round(confidence, 4),
                "category_id": class_id
            })

    print("Запись результатов в файл...")
    df = pd.DataFrame(submission_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Файл сохранен: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
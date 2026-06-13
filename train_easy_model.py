import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

def train_model(model_type='yolov8n', epochs=30, batch=16, imgsz=640, name='yolo_visdrone'):
    print(f"\n--- Начинаем обучение модели {model_type} на GPU ---")
    
    torch.cuda.empty_cache()
    
    model = YOLO(f"{model_type}.pt")
    
    results = model.train(
        data=os.path.abspath('data/layout.yaml'),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=0,         
        workers=1, 
        name=name,
        plots=True,
        amp=False 
    )
    
    best_weights_path = Path(f"runs/detect/{name}/weights/best.pt")
    target_path = Path(f"models/{model_type}_custom.pt")
    
    if best_weights_path.exists():
        shutil.copy(best_weights_path, target_path)
        print(f"Обучение {model_type} завершено! Веса сохранены в: {target_path}")
    else:
        print(f"Внимание: не удалось найти обученные веса по пути {best_weights_path}")

if __name__ == '__main__':
    Path('models').mkdir(exist_ok=True)

    train_model(model_type='yolov8n', epochs=30, batch=2, imgsz=640, name='yolov8n_fast')

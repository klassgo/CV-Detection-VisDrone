import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_box(size, box):
    
    dw = 1/size[0]
    dh = 1/size[1]
    
    x_center = box[0] + box[2] / 2.0
    y_center = box[1] + box[3] / 2.0
    
    x = x_center * dw
    y = y_center * dh
    w = box[2] * dw
    h = box[3] * dh
    return x, y, w, h

def process_visdrone(base_dir, output_dir):
    base_path = Path(base_dir)
    out_path = Path(output_dir)
    

    valid_categories = set(range(1, 11))

    for split in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val']:
        split_name = 'train' if 'train' in split else 'val'
        
    
        check_path = base_path / split / split
        if check_path.exists():
            current_split_path = check_path
        else:
            current_split_path = base_path / split
            
        img_dir = current_split_path / 'images'
        anno_dir = current_split_path / 'annotations'
        
        dest_img_dir = out_path / 'images' / split_name
        dest_lbl_dir = out_path / 'labels' / split_name
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        if not anno_dir.exists():
            print(f"Папка {anno_dir} не найдена, пропускаем split {split_name}")
            continue

        print(f"Конвертация сплита: {split_name}...")
        txt_files = list(anno_dir.glob('*.txt'))
        
        for txt_path in tqdm(txt_files):
            img_path = img_dir / f"{txt_path.stem}.jpg"
            if not img_path.exists():
                continue
                
            with Image.open(img_path) as img:
                img_size = img.size 
                
            import shutil
            shutil.copy(img_path, dest_img_dir / img_path.name)
            
            yolo_annotations = []
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = [int(x) for x in line.strip().split(',') if x.strip()]
                    if len(parts) < 6:
                        continue
                        
                    bbox = parts[0:4]
                    category = parts[5]
                    
                    if category in valid_categories:
                        yolo_class = category - 1 
                        x, y, w, h = convert_box(img_size, bbox)
                        yolo_annotations.append(f"{yolo_class} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            
            out_txt_path = dest_lbl_dir / f"{txt_path.stem}.txt"
            with open(out_txt_path, 'w') as out_f:
                out_f.write('\n'.join(yolo_annotations))

if __name__ == '__main__':

    raw_dataset_dir = r'Z:\data_cv' 
    
    target_dir = './data/raw'
    
    if os.path.exists(raw_dataset_dir):
        process_visdrone(raw_dataset_dir, target_dir)
        print("Конвертация успешно завершена! Данные готовы к обучению.")
    else:
        print(f"Пожалуйста, скачай датасет VisDrone и укажи правильный путь к нему в переменной raw_dataset_dir. Текущий путь '{raw_dataset_dir}' не найден.")
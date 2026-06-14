import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Drone Vision MVP", layout="wide")

st.title("Задача детекции объектов с дронов — VisDrone MVP")
st.write("Прототип ML-приложения для распознавания объектов на базе YOLOv8")

st.sidebar.header("Настройки модели")

model_choice = st.sidebar.selectbox(
    "Выберите ML модель:",
    ["Быстрая (YOLOv8 Nano)", "Точная (YOLOv8 Medium)"]
)

conf_threshold = st.sidebar.slider("Порог уверенности", 0.1, 1.0, 0.25, 0.05)

VISDRONE_CLASSES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}

model_mapping = {
    "Быстрая (YOLOv8 Nano)": MODELS_DIR / "yolov8n_custom.pt",
    "Точная (YOLOv8 Medium)": MODELS_DIR / "yolov8m_custom.pt"
}

target_model_path = model_mapping[model_choice]

@st.cache_resource
def load_yolo_model(model_path_str):
    return YOLO(model_path_str)

model_loaded = False

try:
    model = load_yolo_model(str(target_model_path))
    st.sidebar.success(f"Модель {model_choice} успешно подключена!")
    model_loaded = True
except Exception as e:
    st.sidebar.error(f"Ошибка при загрузке модели: {e}")

st.sidebar.header("Фильтрация объектов")

available_classes = list(VISDRONE_CLASSES.values())

selected_classes = st.sidebar.multiselect(
    "Какие объекты детектировать?",
    options=available_classes,
    default=available_classes,
    help="Выберите классы объектов, которые хотите обнаружить"
)

uploaded_file = st.file_uploader("Загрузите изображение для анализа", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Результат детекции")
        if model_loaded:
            if st.button("Запустить распознавание"):
                with st.spinner("Нейросеть обрабатывает кадр..."):
                    try:
                        img_size = 1024 if "Medium" in model_choice else 640
                        results = model(image, conf=conf_threshold, imgsz=img_size)
                        
                        if selected_classes and len(selected_classes) < len(available_classes):
                            selected_indices = [idx for idx, name in VISDRONE_CLASSES.items() if name in selected_classes]
                            
                            boxes = results[0].boxes
                            if len(boxes) > 0:
                                mask = [int(box.cls[0]) in selected_indices for box in boxes]
                                results[0].boxes = boxes[mask]
                        
                        res_plotted = results[0].plot()
                        st.image(res_plotted, channels="BGR", use_column_width=True)
                        st.success("Готово!")
                        
                        st.write("Обнаруженные объекты:")
                        boxes = results[0].boxes
                        if len(boxes) > 0:
                            class_counts = {}
                            for box in boxes:
                                cls_name = model.names[int(box.cls[0])]
                                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                            
                            for cls_name, count in class_counts.items():
                                st.write(f"- **{cls_name}**: {count}")
                        else:
                            st.info("Объекты не найдены.")
                            
                    except Exception as e:
                        st.error(f"Ошибка в процессе детекции: {e}")
        else:
            st.warning("Распознавание заблокировано, так как модель не загружена.")

if st.button("Вывод графиков обучения моделей"):
    st.image(
        [str(MODELS_DIR / "results_8n.png"), str(MODELS_DIR / "results_8m.png"), str(MODELS_DIR / "results_8m_add60tpochs.png")], 
        caption=["**Обучение Nano (GTX 3050 4gb Laptop)**", "**Обучение Medium (T4 x2 15gb Kaggle)**", "**Дообучение Medium (T4 x2 15gb Kaggle) + 60 эпох**"],
        use_column_width=True
    )
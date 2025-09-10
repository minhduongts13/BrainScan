# 📦 Cài đặt: pip install fastapi uvicorn tensorflow pillow python-multipart

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import shutil
import os
import uvicorn

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    # hoặc ["*"] cho dev nhanh (không recommended prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔧 Huấn luyện mô hình MobileNetV1 nếu chưa có file .h5
def train_model():
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        'dataset/validation_set',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, validation_data=val_generator, epochs=10)
    model.save('my_model.h5')

# 🧠 Load mô hình
if not os.path.exists('my_model.h5'):
    print("🔄 Đang huấn luyện mô hình MobileNetV1...")
    train_model()
    print("✅ Huấn luyện xong và đã lưu mô hình.")
model = load_model('my_model.h5')

# 📸 Xử lý ảnh
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# 🌐 API dự đoán
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        processed_img = preprocess_image(temp_path)
        prediction = model.predict(processed_img)
        probability = float(prediction[0][0])
        result = "hemmorhage_data" if probability > 0.5 else "non_hemmorhage_data"

        return JSONResponse(content={
            "prediction": result,
            "prob": round(probability, 4)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# 🚀 Tự động chạy server khi file được chạy trực tiếp
if __name__ == "__main__":
    uvicorn.run("v1:app", host="0.0.0.0", port=8000, reload=True)
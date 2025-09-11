from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import os

app = FastAPI()

origins = [
    "http://localhost:5173",
    "https://minhduongts13.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    # hoặc ["*"] cho dev nhanh (không recommended prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model MobilenetV1 đã train (sigmoid output)
model = tf.keras.models.load_model("my_model.h5")

# Danh sách class
class_names = ["hemmorhage_data", "non_hemmorhage_data"]

# Tiền xử lý ảnh
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Input size chuẩn cho MobileNetV1
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Hàm dự đoán
def predict(image: Image.Image):
    processed_image = preprocess_image(image)

    # Sigmoid output (chỉ có 1 node)
    prob = model.predict(processed_image, verbose=0)[0][0]  # ra 1 giá trị [0-1]
    prob = 1 - prob
    # Xác suất 2 class (hemorrhage / non-hemorrhage)
    probs = [prob, 1 - prob]

    # Lấy class có xác suất cao nhất

    pred_index = int(prob <= 0.5)  # 0: hemorrhage, 1: non-hemorrhage
    max_prob = probs[pred_index]

    # Quy tắc uncertain 40–60%
    if 0.4 <= max_prob <= 0.6:
        prediction = "uncertain"
    else:
        prediction = class_names[pred_index]

    return {
        "prediction": prediction,
        "confidence": round(float(max_prob), 4),
        "probabilities": {
            "hemmorrage_data": round(float(probs[0]), 4),
            "non_hemmorrage_data": round(float(probs[1]), 4)
        }
    }

# API upload ảnh
@app.post("/predict")
async def predict_hemorrhage(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        result = predict(image)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("v1:app", host="0.0.0.0", port=port, reload=False)
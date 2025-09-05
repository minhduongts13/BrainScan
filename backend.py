from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models

app = FastAPI()

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình đã huấn luyện
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 2)  # 2 lớp: xuất huyết và không xuất huyết
)
model.load_state_dict(torch.load("mobilenetv2_brain_tumor.pt", map_location=device))
model = model.to(device)
model.eval()

# ⚠️ Mapping nhãn đúng với thư mục training_set_augmented
class_names = ["hemmorhage_data", "non_hemmorhage_data"]

# Transform ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# Hàm dự đoán
def predict(image: Image.Image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item()

    return {
        "prediction": class_names[pred_class],
        "confidence": round(confidence, 4),
        "probabilities": {
            class_names[i]: round(probs[i].item(), 4) for i in range(len(class_names))
        }
    }

# Endpoint upload ảnh
@app.post("/predict")
async def predict_hemorrhage(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        result = predict(image)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
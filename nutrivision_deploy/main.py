"""
NutriVision Deployment API
---------------------------
FastAPI backend that serves the trained ResNet50 food classifier,
looks up nutrition info for the predicted class, and serves the
dashboard UI.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Requires in the same folder:
    - best_model_resnet50_finetune.pth   (trained model weights)
    - food101_nutrition.csv              (nutrition lookup table)
"""

import io
import os
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms

# =====================================================
# Config
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_resnet50_finetune.pth")
NUTRITION_CSV = os.path.join(BASE_DIR, "food101_nutrition.csv")
IMG_SIZE = 224
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

# Food-101 class names (alphabetical order - matches torchvision.datasets.ImageFolder's
# sorting, which is what the model was trained against). Keep this list in sync if you
# retrain on a different dataset.
CLASS_NAMES = sorted([
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
    "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
    "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
    "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
    "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
])

ALLERGY_KEYWORDS = {
    "shellfish": ["shrimp", "crab", "lobster", "scallops", "mussels", "oysters", "clam", "calamari"],
    "dairy": ["cheese", "cream", "milk", "cheesecake", "tiramisu", "panna_cotta"],
    "nuts": ["baklava", "macarons"],
    "gluten": ["bread", "pizza", "pasta", "spaghetti", "lasagna", "waffles", "pancakes", "donuts", "churros"],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Load model
# =====================================================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. Place your trained "
            f"best_model_resnet50_finetune.pth in this folder."
        )
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_nutrition():
    if not os.path.exists(NUTRITION_CSV):
        raise FileNotFoundError(f"Nutrition file '{NUTRITION_CSV}' not found.")
    df = pd.read_csv(NUTRITION_CSV).set_index("class_name")
    return df


model = None
nutrition_df = None
startup_error: Optional[str] = None

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# =====================================================
# App
# =====================================================
app = FastAPI(title="NutriVision API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    global model, nutrition_df, startup_error
    startup_error = None
    try:
        model = load_model()
        nutrition_df = load_nutrition()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        model = None
        nutrition_df = None
        startup_error = str(exc)
        print(f"⚠️ Startup warning: {exc}")
        return

    print(f"✅ Model loaded on {device}, {len(CLASS_NAMES)} classes, "
          f"{len(nutrition_df)} nutrition entries.")


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    with open(os.path.join(BASE_DIR, "templates", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "degraded",
        "device": str(device),
        "classes": len(CLASS_NAMES),
        "startup_error": startup_error,
    }


def check_allergies(food_name: str):
    flagged = []
    for allergy, keywords in ALLERGY_KEYWORDS.items():
        if any(k in food_name for k in keywords):
            flagged.append(allergy)
    return flagged


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or nutrition_df is None:
        raise HTTPException(status_code=503, detail=startup_error or "Model is not ready.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.") from exc

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large. Please use an image smaller than 5MB.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image_obj:
            image_obj.verify()
        with Image.open(io.BytesIO(image_bytes)) as image_obj:
            image = image_obj.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image. Please upload a valid JPG or PNG file.") from exc

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        top5_probs, top5_idx = probs.topk(5)

    top5 = [
        {"class": CLASS_NAMES[idx.item()], "confidence": round(prob.item() * 100, 2)}
        for prob, idx in zip(top5_probs, top5_idx)
    ]

    predicted_class = top5[0]["class"]

    if predicted_class not in nutrition_df.index:
        raise HTTPException(status_code=500, detail=f"No nutrition entry for '{predicted_class}'.")

    nutrition_row = nutrition_df.loc[predicted_class]
    nutrition = {
        "calories_kcal": float(nutrition_row["calories_kcal"]),
        "protein_g": float(nutrition_row["protein_g"]),
        "carbs_g": float(nutrition_row["carbs_g"]),
        "fat_g": float(nutrition_row["fat_g"]),
    }

    return {
        "predicted_class": predicted_class,
        "confidence": top5[0]["confidence"],
        "top5": top5,
        "nutrition": nutrition,
        "allergy_flags": check_allergies(predicted_class),
    }


static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# NutriVision Deployment API

FastAPI backend that serves the trained ResNet50 food classifier and the
scan-result dashboard UI, wired together end to end.

## What's in this folder

```
nutrivision_deploy/
├── main.py                  # FastAPI app: model loading + /predict endpoint
├── templates/
│   └── index.html           # Dashboard UI (upload, live results, recommendations)
├── static/                  # (empty - reserved for future assets)
├── requirements.txt
└── README.md
```

## Setup

1. Copy your trained model weights into this folder, named exactly:
   ```
   best_model_resnet50_finetune.pth
   ```

2. Copy the nutrition table into this folder, named exactly:
   ```
   food101_nutrition.csv
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Open your browser at:
   ```
   http://localhost:8000
   ```
   Click the scan frame to upload a food photo. The page calls `/predict`,
   which runs the model and returns the class, confidence, top-5
   alternatives, nutrition breakdown, and allergy flags — all rendered live
   in the dashboard.

## API reference

### `POST /predict`
Multipart form upload, field name `file` (an image).

Response:
```json
{
  "predicted_class": "pad_thai",
  "confidence": 94.2,
  "top5": [
    {"class": "pad_thai", "confidence": 94.2},
    {"class": "fried_rice", "confidence": 3.1},
    ...
  ],
  "nutrition": {
    "calories_kcal": 250.0,
    "protein_g": 9.0,
    "carbs_g": 33.0,
    "fat_g": 9.0
  },
  "allergy_flags": ["gluten"]
}
```

### `GET /health`
Quick check that the model loaded and which device it's running on.

## Notes

- `CLASS_NAMES` in `main.py` must stay in the same order the model was
  trained with (alphabetical, matching `torchvision.datasets.ImageFolder`'s
  default sort). If you retrain on a different class set, update this list.
- The allergy check is a simple keyword match on the class name — good
  enough for a demo, not a substitute for a real ingredient database.
- CORS is wide open (`allow_origins=["*"]`) for easy local testing. Lock
  this down before any public deployment.

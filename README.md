# 🍽️ NutriVision

**AI-powered food recognition, nutritional analysis, and personalized meal recommendations — from a single photo.**

Snap a photo of any dish. NutriVision identifies it, breaks down its nutrition, checks it against your goals and allergies, and tells you what to do next — all in real time, no manual logging required.

---

## ⚡ Why NutriVision is different

Most food-tracking apps stop at "here's a database, search for your meal." NutriVision removes that step entirely.

| USP | What it means |
|---|---|
| **🔍 Zero manual entry** | No searching a food database, no guessing portion names — point a camera, get an answer. |
| **🧠 Real transfer learning, not a toy model** | Built on a fine-tuned ResNet50 (ImageNet → Food-101, ~100K images across 101 classes), not a from-scratch network that overfits in 5 epochs. The difference is measurable: **30% → 80.8%** Top-1 accuracy across the project's iterations. |
| **🎯 Goal-aware recommendations** | The same detected dish gets a *different* recommendation depending on whether you're cutting, bulking, or maintaining — this isn't just a nutrition label, it's a decision layer on top of it. |
| **⚠️ Built-in allergy safety net** | Automatic keyword-based allergen flagging (shellfish, dairy, nuts, gluten) on every prediction — a layer most food-recognition demos skip entirely. |
| **📊 Rigorously validated, not just "trained"** | Confusion matrix, Top-5 accuracy, per-class breakdown, and Grad-CAM visual explanations back every accuracy claim — the model's mistakes are documented and explained, not hidden. |
| **🚀 Two production-ready deployment paths** | Ships with both a **FastAPI** backend (for integration into a larger product) and a **Streamlit** app (for instant standalone demos) — not locked into one deployment story. |
| **🍱 101 dishes, fully covered** | Every single class in Food-101 has a matching nutrition profile — no "sorry, we don't recognize that" gaps. |

---

## 🧭 How it works

```
   📸 Photo                🧠 Model                 📋 Result
┌───────────┐        ┌──────────────────┐      ┌─────────────────────┐
│  Upload a │  --->  │  ResNet50 (fine- │ ---> │  Dish + confidence   │
│ food photo│        │  tuned, 101-way) │      │  Nutrition breakdown │
└───────────┘        └──────────────────┘      │  Goal recommendation │
                                                │  Allergy flags       │
                                                └─────────────────────┘
```

1. **Recognition** — a ResNet50 backbone, pretrained on ImageNet and fully fine-tuned on the full Food-101 dataset (~100,000 images, 101 categories), classifies the dish across 101 categories.
2. **Nutrition lookup** — the predicted class is matched instantly against a curated per-100g nutrition table (calories, protein, carbs, fat).
3. **Personalization** — the same nutrition data is re-interpreted through the lens of the user's goal (maintain / lose weight / gain muscle).
4. **Safety check** — the dish name is cross-checked against common allergen keyword groups, and flags are surfaced immediately, not buried in a settings page.

---

## 📈 Model performance

| Metric | Score |
|---|---|
| Top-1 Accuracy (test set) | **80.81%** |
| Top-5 Accuracy (test set) | **94.70%** |
| Validation Accuracy (best checkpoint) | 80.63% |
| Classes | 101 (Food-101) |
| Dataset size	~100,000 images (1,000 per class) |

**The path to get here matters as much as the number itself:**

| Stage | Approach | Val Accuracy |
|---|---|---|
| 1 | Custom MobileNetV1/V4 trained from scratch | ~30% |
| 2 | MobileNetV3-Small, transfer learning (head-only) | ~48–58% |
| 3 | ResNet50, transfer learning (head-only) | ~74.6% |
| 4 | **ResNet50, full fine-tuning + discriminative LR + cosine schedule** | **80.6%+** |

This progression is deliberate, not accidental — it's a documented case study in why transfer learning and full fine-tuning outperform training from scratch, especially under limited compute (the entire pipeline was trained on free-tier Colab GPUs).

**Where the model still struggles, and why it makes sense:** the lowest-accuracy classes (`steak`, `chocolate_mousse`, `cheesecake`) are confused almost exclusively with visually near-identical dishes (`filet_mignon`, `chocolate_cake`, `strawberry_shortcake`) — confirmed via confusion-matrix analysis and Grad-CAM attention maps. The model isn't making random mistakes; it's making the same mistakes a human would.

---

## 🛠️ Tech stack

- **Model:** PyTorch, torchvision (ResNet50, transfer learning + full fine-tuning)
- **Training:** Mixed-precision (AMP), OneCycle/Cosine LR scheduling, discriminative learning rates, label smoothing, checkpoint-resume training
- **Evaluation:** Top-1/Top-5 accuracy, confusion matrix, per-class accuracy, Grad-CAM
- **Nutrition engine:** Pandas-backed lookup table, 101/101 classes covered
- **Deployment:** FastAPI (REST API) + Streamlit (interactive UI), both wired to the same trained model
- **Frontend:** Custom dark "AI-scan" themed dashboard (not a default template) — image upload, live confidence bar, macro breakdown rings, goal-based recommendation panel, allergy checklist

---

## 📂 Project structure

```
nutrivision_deploy/
├── main.py                  # FastAPI backend — /predict endpoint
├── streamlit_app.py         # Streamlit app — single-file, no backend needed
├── templates/index.html     # Dashboard UI for the FastAPI version
├── food101_nutrition.csv    # Nutrition table, all 101 classes
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

```bash
pip install -r requirements.txt

# Option A — Streamlit (fastest to demo)
streamlit run streamlit_app.py

# Option B — FastAPI (for integration into a larger system)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Place your trained `best_model_resnet50_finetune.pth` in the same folder before running either option.

---

## 🗺️ Roadmap

- [ ] Swap the keyword-based allergen check for an ingredient-level model
- [ ] Portion-size estimation from the image (currently nutrition is per-100g only)
- [ ] Expand beyond Food-101 to region-specific cuisines
- [ ] On-device deployment via TFLite/ONNX export for a true mobile experience

---

*Built as an end-to-end deep learning project: from architecture comparison (MobileNetV1/V4 vs. ResNet50 vs. EfficientNet-B3) through training optimization, rigorous evaluation, and full-stack deployment.*

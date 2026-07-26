"""
NutriVision - Streamlit App (styled to match the FastAPI dashboard)
--------------------------------------------------------------------
Run with:
    streamlit run streamlit_app.py

Requires in the same folder:
    - best_model_resnet50_finetune.pth   (trained model weights)
    - food101_nutrition.csv              (nutrition lookup table)
"""

import base64
import io

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# =====================================================
# Config
# =====================================================
MODEL_PATH = "best_model_resnet50_finetune.pth"
NUTRITION_CSV = "food101_nutrition.csv"
IMG_SIZE = 224

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
ALLERGY_LABELS = {"shellfish": "Shellfish", "dairy": "Dairy", "nuts": "Nuts", "gluten": "Gluten"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =====================================================
# Cached loaders
# =====================================================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@st.cache_data
def load_nutrition():
    return pd.read_csv(NUTRITION_CSV).set_index("class_name")


def check_allergies(food_name: str):
    return [a for a, keywords in ALLERGY_KEYWORDS.items() if any(k in food_name for k in keywords)]


def get_recommendation_lines(row, goal, flags):
    calories, protein = row["calories_kcal"], row["protein_g"]
    lines = []

    if goal == "lose_weight":
        if calories > 300:
            lines.append(("🔴", f"Relatively high calories ({calories:.0f} kcal/100g) for a weight-loss goal — consider a smaller portion."))
        elif calories > 200:
            lines.append(("🟡", f"Moderate calories ({calories:.0f} kcal/100g) — fine in a reasonable portion."))
        else:
            lines.append(("🟢", f"Low calories ({calories:.0f} kcal/100g) — a good choice for your goal."))
    elif goal == "gain_muscle":
        if protein >= 20:
            lines.append(("🟢", f"High protein ({protein:.0f}g/100g) — great for building muscle."))
        elif protein >= 10:
            lines.append(("🟡", f"Moderate protein ({protein:.0f}g/100g) — consider adding another protein source."))
        else:
            lines.append(("🔴", f"Low protein ({protein:.0f}g/100g) — not the best choice for muscle-building alone."))
    else:
        lines.append(("ℹ️", f"{calories:.0f} kcal · {protein:.0f}g protein · {row['carbs_g']:.0f}g carbs · {row['fat_g']:.0f}g fat per 100g."))

    if flags:
        lines.append(("⚠️", f"Contains possible allergens: {', '.join(flags)}."))

    return lines


def img_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def ring_svg(color: str, fraction: float, label: str) -> str:
    circumference = 201
    offset = circumference * (1 - max(0.0, min(fraction, 1.0)))
    return f"""
    <div class="ring">
      <svg width="76" height="76" viewBox="0 0 76 76">
        <circle cx="38" cy="38" r="32" fill="none" stroke="#2c2f21" stroke-width="7"/>
        <circle cx="38" cy="38" r="32" fill="none" stroke="{color}" stroke-width="7"
          stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"/>
      </svg>
      <div class="ring-val">{label}</div>
    </div>
    """


def clean_html(html: str) -> str:
    """Strip leading whitespace from every line, and drop blank lines
    entirely, before handing HTML/CSS to st.markdown.

    Two Markdown quirks were breaking our raw HTML/CSS:
    1. 4+ leading spaces on a line = an indented code block (renders as
       literal text instead of being parsed).
    2. A blank line inside a raw-HTML block makes Markdown think the HTML
       block ended there, so everything after it gets treated as a new
       paragraph of plain text - which is exactly what was happening
       between our CSS rule groups.
    """
    lines = [line.strip() for line in html.strip().split("\n")]
    return "\n".join(line for line in lines if line)


# =====================================================
# Page config + theme CSS (matches the FastAPI dashboard)
# =====================================================
st.set_page_config(page_title="NutriVision", page_icon="🍽️", layout="centered")

theme_css = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
--bg:#101109; --panel:#181a12; --panel-2:#1f2216; --line:#2c2f21;
--saffron:#F2A93B; --scan:#63E6D4; --ink:#F4F1E6; --muted:#9C9A85;
--faint:#5c5b4c; --danger:#E8694A;
}
#MainMenu, footer, header {visibility: hidden;}
.stApp{
background:
radial-gradient(1200px 600px at 15% -10%, rgba(242,169,59,0.10), transparent 60%),
radial-gradient(900px 500px at 100% 0%, rgba(99,230,212,0.07), transparent 55%),
var(--bg);
}
.block-container{padding-top:2.2rem;max-width:900px;}
html, body, [class*="css"]{font-family:'Inter',sans-serif;color:var(--ink);}

.brand{display:flex;align-items:center;gap:10px;font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:16px;margin-bottom:6px;}
.brand-mark{width:22px;height:22px;border-radius:5px;background:linear-gradient(135deg,var(--saffron),var(--scan));flex-shrink:0;}
.status-pill{
display:inline-flex;align-items:center;gap:7px;font-family:'JetBrains Mono',monospace;font-size:11px;
letter-spacing:0.04em;color:var(--scan);background:rgba(99,230,212,0.08);
border:1px solid rgba(99,230,212,0.25);padding:5px 12px;border-radius:100px;margin-bottom:28px;
}
.dot{width:6px;height:6px;border-radius:50%;background:var(--scan);box-shadow:0 0 8px var(--scan);}

.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin-bottom:16px;}
.panel h3{font-family:'Space Grotesk',sans-serif;font-size:15px;margin:0 0 12px;font-weight:600;}
.section-label{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin:22px 0 12px 2px;}

.hero{display:flex;gap:24px;background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin-bottom:6px;flex-wrap:wrap;}
.scan-frame{position:relative;width:220px;height:220px;border-radius:12px;overflow:hidden;background:#0c0d08;flex-shrink:0;}
.scan-frame img{width:100%;height:100%;object-fit:cover;display:block;}
.corner{position:absolute;width:20px;height:20px;border-color:var(--saffron);opacity:0.9;}
.corner.tl{top:10px;left:10px;border-top:2px solid;border-left:2px solid;}
.corner.tr{top:10px;right:10px;border-top:2px solid;border-right:2px solid;}
.corner.bl{bottom:10px;left:10px;border-bottom:2px solid;border-left:2px solid;}
.corner.br{bottom:10px;right:10px;border-bottom:2px solid;border-right:2px solid;}
.scanline{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--scan),transparent);box-shadow:0 0 12px 2px var(--scan);animation:sweep 3.2s ease-in-out infinite;opacity:0.85;}
@keyframes sweep{0%{top:6%;}50%{top:92%;}100%{top:6%;}}
.scan-tag{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.08em;color:var(--bg);background:var(--saffron);padding:3px 9px;border-radius:5px;font-weight:600;}

.hero-info{flex:1;min-width:220px;display:flex;flex-direction:column;justify-content:center;}
.eyebrow{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;}
.food-name{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:30px;line-height:1.05;margin:0 0 14px;text-transform:capitalize;}
.confidence-row{display:flex;align-items:center;gap:12px;margin-bottom:16px;}
.conf-bar-track{flex:1;height:6px;border-radius:100px;background:var(--line);overflow:hidden;max-width:200px;}
.conf-bar-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--saffron),var(--scan));}
.conf-value{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--scan);font-weight:600;}
.alt-row{display:flex;gap:10px;font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--faint);margin-bottom:4px;}
.alt-row .name{color:var(--muted);}

.macro-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:6px;}
.macro-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 10px;display:flex;flex-direction:column;align-items:center;gap:8px;}
.macro-card.kcal{background:linear-gradient(135deg, rgba(242,169,59,0.10), rgba(99,230,212,0.05));border:1px solid rgba(242,169,59,0.28);}
.ring{position:relative;width:76px;height:76px;}
.ring svg{transform:rotate(-90deg);}
.ring-val{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-weight:600;font-size:13px;color:var(--ink);}
.macro-label{font-size:12px;color:var(--muted);font-weight:500;}
.macro-grams{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--faint);}

.rec-line{display:flex;gap:9px;align-items:flex-start;font-size:13.5px;line-height:1.5;padding:8px 0;border-top:1px solid var(--line);}
.rec-line:first-of-type{border-top:none;padding-top:0;}

.allergy-item{display:flex;align-items:center;justify-content:space-between;font-size:13px;padding:9px 12px;border-radius:9px;background:var(--panel-2);border:1px solid var(--line);margin-bottom:6px;}
.allergy-item.flagged{border-color:rgba(232,105,74,0.4);background:rgba(232,105,74,0.07);}
.allergy-item .tag{font-family:'JetBrains Mono',monospace;font-size:10px;padding:3px 8px;border-radius:5px;background:var(--line);color:var(--muted);}
.allergy-item.flagged .tag{background:var(--danger);color:#fff;}

/* goal buttons -> pill look */
div.stButton > button{
border-radius:100px !important;font-family:'Inter',sans-serif;font-size:12.5px;font-weight:500;
border:1px solid var(--line) !important;background:var(--panel-2) !important;color:var(--muted) !important;
padding:6px 16px !important;
}
div.stButton > button[kind="primary"]{
background:var(--saffron) !important;border-color:var(--saffron) !important;color:#191a10 !important;
}

[data-testid="stFileUploader"]{
background:var(--panel) !important;border:1px dashed var(--line) !important;border-radius:14px !important;padding:8px !important;
}
footer, .footnote{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--faint);text-align:center;letter-spacing:0.04em;margin-top:20px;}
</style>
"""
st.markdown(clean_html(theme_css), unsafe_allow_html=True)

st.markdown("""
<div class="brand"><div class="brand-mark"></div>NutriVision</div>
<div class="status-pill"><span class="dot"></span>MODEL: RESNET50-FT · READY</div>
""", unsafe_allow_html=True)


try:
    model = load_model()
    nutrition_df = load_nutrition()
except FileNotFoundError as e:
    st.error(f"⚠️ {e}\n\nMake sure `{MODEL_PATH}` and `{NUTRITION_CSV}` are in this folder.")
    st.stop()

if "goal" not in st.session_state:
    st.session_state.goal = "maintain"

uploaded_file = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing…"):
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            top5_probs, top5_idx = probs.topk(5)

    top5 = [(CLASS_NAMES[idx.item()], prob.item() * 100) for prob, idx in zip(top5_probs, top5_idx)]
    predicted_class, confidence = top5[0]
    row = nutrition_df.loc[predicted_class]

    img_b64 = img_to_base64(image)
    alt_rows_html = "".join(
        f'<div class="alt-row"><span>0{i+2}</span><span class="name">{name}</span><span>{pct:.1f}%</span></div>'
        for i, (name, pct) in enumerate(top5[1:3])
    )

    # ---------------- Hero ----------------
    hero_html = f"""
    <div class="hero">
      <div class="scan-frame">
        <img src="data:image/jpeg;base64,{img_b64}" />
        <div class="corner tl"></div><div class="corner tr"></div>
        <div class="corner bl"></div><div class="corner br"></div>
        <div class="scanline"></div>
        <div class="scan-tag">SCANNED</div>
      </div>
      <div class="hero-info">
        <div class="eyebrow">Detected dish</div>
        <div class="food-name">{predicted_class.replace('_',' ')}</div>
        <div class="confidence-row">
          <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{confidence}%;"></div></div>
          <span class="conf-value">{confidence:.1f}%</span>
        </div>
        {alt_rows_html}
      </div>
    </div>
    """
    st.markdown(clean_html(hero_html), unsafe_allow_html=True)

    # ---------------- Macro rings ----------------
    st.markdown('<div class="section-label">Nutrition — per 100g</div>', unsafe_allow_html=True)

    kcal_from_macros = row["protein_g"] * 4 + row["carbs_g"] * 4 + row["fat_g"] * 9
    protein_pct = round(row["protein_g"] * 4 / kcal_from_macros * 100) if kcal_from_macros else 0
    carbs_pct = round(row["carbs_g"] * 4 / kcal_from_macros * 100) if kcal_from_macros else 0
    fat_pct = round(row["fat_g"] * 9 / kcal_from_macros * 100) if kcal_from_macros else 0

    macro_html = f"""
    <div class="macro-grid">
      <div class="macro-card kcal">
        {ring_svg("#F2A93B", min(row['calories_kcal']/500, 1), f"{row['calories_kcal']:.0f}")}
        <div class="macro-label">Calories</div><div class="macro-grams">kcal/100g</div>
      </div>
      <div class="macro-card">
        {ring_svg("#63E6D4", protein_pct/100, f"{protein_pct}%")}
        <div class="macro-label">Protein</div><div class="macro-grams">{row['protein_g']:.0f}g</div>
      </div>
      <div class="macro-card">
        {ring_svg("#E8B84A", carbs_pct/100, f"{carbs_pct}%")}
        <div class="macro-label">Carbs</div><div class="macro-grams">{row['carbs_g']:.0f}g</div>
      </div>
      <div class="macro-card">
        {ring_svg("#E8694A", fat_pct/100, f"{fat_pct}%")}
        <div class="macro-label">Fat</div><div class="macro-grams">{row['fat_g']:.0f}g</div>
      </div>
    </div>
    """
    st.markdown(clean_html(macro_html), unsafe_allow_html=True)

    # ---------------- Goal chips + recommendation / allergy ----------------
    col_rec, col_allergy = st.columns(2)

    with col_rec:
        st.markdown('<div class="panel"><h3>Recommendation</h3>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        goal_options = [("maintain", "Maintain"), ("lose_weight", "Lose weight"), ("gain_muscle", "Gain muscle")]
        for col, (key, label) in zip([c1, c2, c3], goal_options):
            with col:
                if st.button(label, key=f"goal_{key}",
                             type="primary" if st.session_state.goal == key else "secondary"):
                    st.session_state.goal = key
                    st.rerun()

        flags = check_allergies(predicted_class)
        rec_lines = get_recommendation_lines(row, st.session_state.goal, flags)
        rec_html = "".join(f'<div class="rec-line"><span>{icon}</span><span>{text}</span></div>' for icon, text in rec_lines)
        st.markdown(rec_html + "</div>", unsafe_allow_html=True)

    with col_allergy:
        allergy_html = "".join(
            f'<div class="allergy-item {"flagged" if key in flags else ""}">'
            f'<span>{label}</span><span class="tag">{"FLAGGED" if key in flags else "CLEAR"}</span></div>'
            for key, label in ALLERGY_LABELS.items()
        )
        st.markdown(f'<div class="panel"><h3>Allergy check</h3>{allergy_html}</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="panel" style="text-align:center;color:var(--muted);">Upload a food photo above to get started.</div>', unsafe_allow_html=True)

st.markdown('<div class="footnote">NUTRIVISION · FOOD-101 · 101 CLASSES · TOP-1 80.8% / TOP-5 94.7%</div>', unsafe_allow_html=True)
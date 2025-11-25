###############################################################
# CHANFUI — OCR FACTURE PRO (GOOGLE VISION + UI PREMIUM)
# VERSION 2025 — FICHIER COMPLET (prêt à coller & exécuter)
#
# Remarques :
# - Place les fichiers JSON de credentials (chanfuiocr-478317-46a916fc6992.json
#   et Credentials.json) dans le même dossier que ce script.
# - SAMPLE_IMAGE_PATH pointe vers l'exemple que tu as uploadé : /mnt/data/Teste.jpg
###############################################################

import streamlit as st
import cv2
import numpy as np
import re
import os
import json
import time
from datetime import datetime, date
from io import BytesIO
from PIL import Image
from google.cloud import vision
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import pandas as pd

# ---------------------------
# Config / chemins / constantes
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fichiers JSON (mets-les dans le même dossier)
VISION_CREDENTIALS = os.path.join(BASE_DIR, "chanfuiocr-478317-46a916fc6992.json")
SHEETS_CREDENTIALS = os.path.join(BASE_DIR, "Credentials.json")  # fallback local

# Exemple d'image fournie (chemin local)
SAMPLE_IMAGE_PATH = "/mnt/data/Teste.jpg"   # <-- fichier envoyé précédemment

st.set_page_config(page_title="ChanFui OCR PRO", layout="centered", page_icon="🍷")

SCAN_STATE_FILENAME = "scan_state.json"

COLORS = [
    {"red": 0.8, "green": 1.0, "blue": 0.8},
    {"red": 0.15, "green": 0.15, "blue": 0.15},
    {"red": 1.0, "green": 0.7, "blue": 0.7},
]

AUTHORIZED_USERS = {
    "haina": "A1234",
    "paul": "B5531",
    "lina": "C9910",
    "admin": "D2201"
}

# ---------------------------
# CSS UI
# ---------------------------
st.markdown("""
<style>
:root{
  --wine:#6d071a;
  --muted:#7a4b4b;
  --card-bg:rgba(255,255,255,0.96);
}
html, body, [data-testid='stAppViewContainer']{
  background:linear-gradient(180deg,#fffaf8,#fff6f6);
}
.wine-title{
  font-family:'Georgia';
  font-size:34px;
  color:var(--wine);
  font-weight:700;
  text-align:center;
}
.wine-sub{color:var(--muted);text-align:center}
.chancard{
  border-radius:14px;
  padding:16px;
  background:var(--card-bg);
  box-shadow:0 8px 30px rgba(110,20,25,0.06);
  border:1px solid rgba(109,7,26,0.06);
}
.btn-wine{
  background-color:var(--wine);
  color:white;
  border-radius:10px;
  padding:8px 12px;
  border:none;
  font-weight:600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='wine-title'>ChanFui & Fils </div><div class='wine-sub'>Google Vision Premium Edition</div>", unsafe_allow_html=True)

# ---------------------------
# Helpers - scan state
# ---------------------------
APP_FOLDER = os.getcwd()

def load_scan_state():
    p = os.path.join(APP_FOLDER, SCAN_STATE_FILENAME)
    if os.path.exists(p):
        return json.load(open(p, "r", encoding="utf-8"))
    return {"scan_index": 0}

def save_scan_state(data):
    p = os.path.join(APP_FOLDER, SCAN_STATE_FILENAME)
    json.dump(data, open(p, "w", encoding="utf-8"))

scan_state = load_scan_state()

# ---------------------------
# Login
# ---------------------------
def do_logout():
    for k in ["auth", "user_nom", "user_matricule"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

def login_block():
    st.markdown("### 🔐 Connexion")
    nom = st.text_input("Nom")
    mat = st.text_input("Matricule", type="password")
    if st.button("Se connecter"):
        if nom.lower() in AUTHORIZED_USERS and AUTHORIZED_USERS[nom.lower()] == mat:
            st.session_state.auth = True
            st.session_state.user_nom = nom
            st.session_state.user_matricule = mat
            st.success("Connexion OK")
            time.sleep(0.3)
            st.rerun()
        else:
            st.error("Accès refusé")

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    login_block()
    st.stop()

# ---------------------------
# Preprocessing image
# ---------------------------
def preprocess_image(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Contrast via LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # Sharpen
    sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, sharp)

    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes()

# ---------------------------
# Google Vision OCR (explicit credentials)
# ---------------------------
def google_vision_ocr(img_bytes):
    if not os.path.exists(VISION_CREDENTIALS):
        raise FileNotFoundError(f"Fichier de credentials Vision introuvable: {VISION_CREDENTIALS}")
    client = vision.ImageAnnotatorClient.from_service_account_file(VISION_CREDENTIALS)
    image = vision.Image(content=img_bytes)
    response = client.text_detection(image=image)
    if response.error and response.error.message:
        raise Exception(f"Google Vision Error: {response.error.message}")
    raw_text = ""
    if response.text_annotations:
        raw_text = response.text_annotations[0].description
    return raw_text or ""

def clean_text(text):
    text = text.replace("\r", "\n")
    text = text.replace("\n ", "\n")
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = text.replace("’", "'")
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()

# ---------------------------
# Extraction helpers
# ---------------------------
def extract_invoice_number(text):
    p = r"FACTURE\s+EN\s+COMPTE.*?N[°o]?\s*([0-9]{3,})"
    m = re.search(p, text, flags=re.I)
    if m:
        return m.group(1).strip()
    patterns = [r"FACTURE.*?N[°o]\s*([0-9]{3,})", r"FACTURE.*?N\s*([0-9]{3,})"]
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return m.group(1).strip()
    m = re.search(r"N°\s*([0-9]{3,})", text)
    if m:
        return m.group(1)
    return ""

def extract_delivery_address(text):
    p = r"Adresse de livraison\s*[:\-]\s*(.+)"
    m = re.search(p, text, flags=re.I)
    if m:
        return m.group(1).strip().rstrip(".")
    p2 = r"Adresse(?:\s+de\s+livraison)?\s*[:\-]?\s*\n?\s*(.+)"
    m2 = re.search(p2, text, flags=re.I)
    if m2:
        return m2.group(1).strip().split("\n")[0]
    return ""

def extract_doit(text):
    p = r"\bDOIT\s*[:\-]?\s*([A-Z0-9]{2,6})"
    m = re.search(p, text, flags=re.I)
    if m:
        return m.group(1).strip()
    candidates = ["S2M", "ULYS", "DLP"]
    for c in candidates:
        if c in text:
            return c
    return ""

def extract_month(text):
    months = {
        "janvier":"Janvier", "février":"Février", "fevrier":"Février", "mars":"Mars", "avril":"Avril",
        "mai":"Mai", "juin":"Juin", "juillet":"Juillet", "août":"Août", "aout":"Août",
        "septembre":"Septembre", "octobre":"Octobre",
        "novembre":"Novembre", "décembre":"Décembre", "decembre":"Décembre"
    }
    for mname in months:
        if re.search(r"\b" + re.escape(mname) + r"\b", text, flags=re.I):
            return months[mname]
    return ""

def extract_bon_commande(text):
    m = re.search(r"Suivant votre bon de commande\s*[:\-]?\s*([0-9A-Za-z\-\/]+)", text, flags=re.I)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"bon de commande\s*[:\-]?\s*(.+)", text, flags=re.I)
    if m2:
        return m2.group(1).strip().split()[0]
    return ""

def extract_items(text):
    items = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    pattern = re.compile(r"(.+?(?:75\s*cls?|75\s*cl|75cl|75))\s+\d+\s+\d+\s+(\d+)", flags=re.I)
    for l in lines:
        m = pattern.search(l)
        if m:
            name = m.group(1).strip()
            nb_btls = int(m.group(2))
            name = re.sub(r"\s{2,}", " ", name)
            items.append({"article": name, "bouteilles": nb_btls})
    if not items:
        for l in lines:
            if "75" in l or "cls" in l.lower():
                nums = re.findall(r"(\d{1,4})", l)
                if nums:
                    nb_btls = int(nums[-1])
                    name = re.sub(r"\d+", "", l).strip()
                    items.append({"article": name, "bouteilles": nb_btls})
    return items

# ---------------------------
# Pipeline
# ---------------------------
def invoice_pipeline(image_bytes):
    clean_img = preprocess_image(image_bytes)
    raw = google_vision_ocr(clean_img)
    raw = clean_text(raw)
    return {
        "raw": raw,
        "facture": extract_invoice_number(raw),
        "adresse": extract_delivery_address(raw),
        "doit": extract_doit(raw),
        "mois": extract_month(raw),
        "bon_commande": extract_bon_commande(raw),
        "articles": extract_items(raw)
    }

# ---------------------------
# Google Sheets helpers
# ---------------------------
def get_worksheet():
    creds_path = None
    if "GCP_CREDENTIALS_PATH" in st.secrets:
        creds_path = st.secrets["GCP_CREDENTIALS_PATH"]
    elif os.path.exists(SHEETS_CREDENTIALS):
        creds_path = SHEETS_CREDENTIALS
    else:
        raise FileNotFoundError("Credentials Google Sheets introuvable. Mettez st.secrets['GCP_CREDENTIALS_PATH'] ou Credentials.json dans le dossier.")

    sheet_id = st.secrets.get("SHEET_ID", None)
    if not sheet_id:
        raise KeyError("Mettez 'SHEET_ID' dans st.secrets pour l'ID du Google Sheet.")

    creds = Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    return sh.sheet1

def get_sheets_service():
    creds_path = None
    if "GCP_CREDENTIALS_PATH" in st.secrets:
        creds_path = st.secrets["GCP_CREDENTIALS_PATH"]
    elif os.path.exists(SHEETS_CREDENTIALS):
        creds_path = SHEETS_CREDENTIALS
    else:
        raise FileNotFoundError("Credentials Google Sheets introuvable.")

    creds = Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return build("sheets", "v4", credentials=creds)

def color_rows(spreadsheet_id, sheet_id, start, end, color):
    service = get_sheets_service()
    body = {
        "requests":[
            {
                "repeatCell":{
                    "range":{
                        "sheetId":sheet_id,
                        "startRowIndex":start,
                        "endRowIndex":end
                    },
                    "cell":{"userEnteredFormat":{"backgroundColor":color}},
                    "fields":"userEnteredFormat.backgroundColor"
                }
            }
        ]
    }
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body
    ).execute()

# ---------------------------
# UI - Upload / camera
# ---------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='chancard'>", unsafe_allow_html=True)

uploaded = st.file_uploader("Importer une facture (jpg/png)", type=["jpg","jpeg","png"])
take = st.button("📸 Prendre une photo maintenant")

camera_image = None
if take:
    st.info("Active la caméra puis capture.")
    camera_image = st.camera_input("Photo")

st.markdown("</div>", unsafe_allow_html=True)

img = None
if camera_image:
    img = Image.open(camera_image)
elif uploaded:
    img = Image.open(uploaded)
else:
    if os.path.exists(SAMPLE_IMAGE_PATH):
        with st.expander("Exemple facture (démo)"):
            st.image(SAMPLE_IMAGE_PATH, use_container_width=True)

# session state for edited table
if "edited_articles_df" not in st.session_state:
    st.session_state["edited_articles_df"] = None

if img:
    st.image(img, caption="Aperçu", use_container_width=True)

    buf = BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    st.info("Traitement OCR Google Vision...")
    p = st.progress(10)
    try:
        res = invoice_pipeline(img_bytes)
    except Exception as e:
        st.error(f"Erreur OCR: {e}")
        p.empty()
        st.stop()
    p.progress(100)
    p.empty()

    # Editable detection fields
    st.subheader("Informations détectées (modifiable)")
    col1, col2 = st.columns(2)
    facture_val = col1.text_input("🔢 Numéro de facture", value=res.get("facture", ""))
    bon_commande_val = col1.text_input("📦 Suivant votre bon de commande", value=res.get("bon_commande", ""))
    adresse_val = col2.text_input("📍 Adresse de livraison", value=res.get("adresse", ""))
    doit_val = col2.text_input("👤 DOIT", value=res.get("doit", ""))
    # mois as selectbox with detected default if available
    month_detected = res.get("mois", "")
    mois_val = col2.selectbox("📅 Mois", [""] + ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"], index= (0 if not month_detected else ["","Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"].index(month_detected)) )

    # prepare articles df
    detected_articles = res.get("articles", [])
    if not detected_articles:
        detected_articles = [{"article": "", "bouteilles": 0}]

    df_articles = pd.DataFrame(detected_articles)
    if "article" not in df_articles.columns:
        df_articles["article"] = ""
    if "bouteilles" not in df_articles.columns:
        df_articles["bouteilles"] = 0
    df_articles["bouteilles"] = pd.to_numeric(df_articles["bouteilles"].fillna(0), errors="coerce").fillna(0).astype(int)

    st.subheader("Articles détectés (modifiable)")

    edited_df = st.data_editor(
        df_articles,
        num_rows="dynamic",
        column_config={
            "article": st.column_config.TextColumn(label="Article"),
            "bouteilles": st.column_config.NumberColumn(label="Nb bouteilles", min_value=0)
        },
        use_container_width=True
    )

    # Add line button
    if st.button("➕ Ajouter une ligne"):
        new_row = pd.DataFrame([{"article": "", "bouteilles": 0}])
        edited_df = pd.concat([edited_df, new_row], ignore_index=True)
        st.session_state["edited_articles_df"] = edited_df
        st.rerun()

    # store edited df
    st.session_state["edited_articles_df"] = edited_df.copy()

    st.subheader("Texte brut (résultat OCR)")
    st.code(res["raw"])

# ---------------------------
# Prepare worksheet (attempt, but non-blocking)
# ---------------------------
try:
    ws = get_worksheet()
    sheet_id = ws.id
    spreadsheet_id = st.secrets.get("SHEET_ID", None)
except Exception as e:
    ws = None
    sheet_id = None
    spreadsheet_id = None

# ---------------------------
# ENVOI -> Google Sheets (format EXACT selon ton modèle)
# ---------------------------
if img and st.session_state.get("edited_articles_df") is not None and ws and st.button("📤 Envoyer vers Google Sheets"):
    try:
        edited = st.session_state["edited_articles_df"].copy()
        # remove empty lines
        edited = edited[~((edited["article"].astype(str).str.strip() == "") & (edited["bouteilles"] == 0))]
        edited["bouteilles"] = pd.to_numeric(edited["bouteilles"].fillna(0), errors="coerce").fillna(0).astype(int)

        # model columns:
        # A MOIS | B DOIT | C Date | D Suivant votre bon de commande | E Adresse de livraison
        # F Article | G Nb bouteilles | H editeur

        # compute start row (1-based)
        start_row = len(ws.get_all_values()) + 1

        # date string: use today
        today_str = datetime.now().strftime("%d/%m/%Y")

        for _, row in edited.iterrows():
            ws.append_row([
                mois_val or "",                  # A
                doit_val or "",                  # B
                today_str,                       # C  (or use a date input if you prefer)
                bon_commande_val or "",          # D
                adresse_val or "",               # E
                row.get("article", ""),          # F
                int(row.get("bouteilles", 0)),   # G
                st.session_state.user_nom        # H editor
            ])

        end_row = len(ws.get_all_values())

        # color rows (startRowIndex & endRowIndex are 0-based in API)
        color = COLORS[scan_state["scan_index"] % len(COLORS)]
        if spreadsheet_id and sheet_id is not None:
            # start_row - 1 because API uses 0-index
            color_rows(spreadsheet_id, sheet_id, start_row-1, end_row, color)

        scan_state["scan_index"] += 1
        save_scan_state(scan_state)

        # Report
        st.success("✅ Données insérées avec succès !")
        st.info(f"📌 Lignes insérées dans le sheet : {start_row} → {end_row}")
        st.write("🧾 Récapitulatif envoyé :")
        st.json({
            "mois": mois_val,
            "doit": doit_val,
            "date_envoye": today_str,
            "bon_de_commande": bon_commande_val,
            "adresse": adresse_val,
            "nb_lignes_envoyees": len(edited),
            "editeur": st.session_state.user_nom
        })

    except Exception as e:
        st.error(f"❌ Erreur envoi Sheets: {e}")

# ---------------------------
# Aperçu du Google Sheet
# ---------------------------
if ws and st.button("👀 Aperçu du Google Sheet"):
    try:
        # show first 200 rows to keep it responsive
        records = ws.get_all_records()
        df_sheet = pd.DataFrame(records)
        if df_sheet.shape[0] > 200:
            st.warning("⚠ Le sheet contient plus de 200 lignes — affichage des 200 premières lignes.")
            st.dataframe(df_sheet.head(200), use_container_width=True)
        else:
            st.dataframe(df_sheet, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors du chargement du sheet : {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.button("🚪 Déconnexion", on_click=do_logout)

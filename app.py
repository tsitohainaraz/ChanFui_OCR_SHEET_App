# app_streamlit_cloud.py
# ChanFui OCR — Streamlit Cloud ready (no OpenCV, uses st.secrets for credentials)
# - Keep your secrets in .streamlit/secrets.toml (see template at the end of this file)
# - This file replaces cv2 preprocessing by PIL-based preprocessing and uses Google Vision
# - Google Sheets coloration is preserved (uses googleapiclient)

import streamlit as st
import numpy as np
import re
import time
import json
import os
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
from google.cloud import vision
from google.oauth2.service_account import Credentials as SA_Credentials
import gspread
from googleapiclient.discovery import build
import pandas as pd

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="ChanFui OCR PRO", layout="centered", page_icon="🍷")

# ---------------------------
# Constants & UI styles (same as your original)
# ---------------------------
COLORS = [
    {"red": 0.8, "green": 1.0, "blue": 0.8},
    {"red": 0.15, "green": 0.15, "blue": 0.15},
    {"red": 1.0, "green": 0.7, "blue": 0.7},
]

AUTHORIZED_USERS = {
    "CFADMIN": "A1234",
    "CFCOMERCIALE": "B5531",
    "CFSTOCK": "C9910",
    "CFDIRECTION": "D2201"
}

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
# Session-state backed scan index (instead of local file)
# ---------------------------
if "scan_index" not in st.session_state:
    # try to initialize from optional scan_state in secrets for migration
    try:
        st.session_state.scan_index = int(st.secrets.get("SCAN_STATE", {}).get("scan_index", 0))
    except Exception:
        st.session_state.scan_index = 0

# ---------------------------
# Login
# ---------------------------

def do_logout():
    for k in ["auth", "user_nom", "user_matricule"]:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()


def login_block():
    st.markdown("### 🔐 Connexion")
    nom = st.text_input("Nom")
    mat = st.text_input("Matricule", type="password")
    if st.button("Se connecter"):
        if nom.upper() in AUTHORIZED_USERS and AUTHORIZED_USERS[nom.upper()] == mat:
            st.session_state.auth = True
            st.session_state.user_nom = nom
            st.session_state.user_matricule = mat
            st.success("Connexion OK")
            time.sleep(0.3)
            st.experimental_rerun()
        else:
            st.error("Accès refusé")

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    login_block()
    st.stop()

# ---------------------------
# Image preprocessing (PIL-based, Cloud-friendly)
# ---------------------------

def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Replace OpenCV preprocessing with PIL-based operations:
    - open image
    - convert to RGB
    - auto-contrast, median filter (denoise), sharpen, optional resize
    - return JPEG bytes
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Optional resize to a max width to speed up processing (keeps aspect)
    max_width = 2600
    if img.width > max_width:
        ratio = max_width / img.width
        new_h = int(img.height * ratio)
        img = img.resize((max_width, new_h), Image.LANCZOS)

    # Auto contrast
    img = ImageOps.autocontrast(img)
    # Median filter for light denoise
    img = img.filter(ImageFilter.MedianFilter(size=3))
    # Slight sharpen
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

    # Save back to JPEG bytes
    out = BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()

# ---------------------------
# Google Vision OCR using st.secrets
# ---------------------------

def get_vision_client():
    # support flexible secrets keys: 'gcp_vision' or 'gcp_sheet' or 'google_service_account'
    if "gcp_vision" in st.secrets:
        sa_info = dict(st.secrets["gcp_vision"])  # preferred
    elif "google_service_account" in st.secrets:
        sa_info = dict(st.secrets["google_service_account"])  # older template
    else:
        raise RuntimeError("Credentials Google Vision introuvables dans st.secrets (gcp_vision)")

    creds = SA_Credentials.from_service_account_info(sa_info)
    client = vision.ImageAnnotatorClient(credentials=creds)
    return client


def google_vision_ocr(img_bytes: bytes) -> str:
    client = get_vision_client()
    image = vision.Image(content=img_bytes)
    response = client.text_detection(image=image)
    if response.error and response.error.message:
        raise Exception(f"Google Vision Error: {response.error.message}")
    raw_text = ""
    if response.text_annotations:
        raw_text = response.text_annotations[0].description
    return raw_text or ""

# ---------------------------
# Text cleaning & extraction (same helpers)
# ---------------------------

def clean_text(text):
    text = text.replace("\r", "\n")
    text = text.replace("\n ", "\n")
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = text.replace("’", "'")
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()

# Extraction helpers unchanged (copy your functions)
# ... (we keep the same extraction helpers as in your original code)

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
# Pipeline (uses PIL preprocess + Google Vision)
# ---------------------------

def invoice_pipeline(image_bytes: bytes):
    cleaned = preprocess_image(image_bytes)
    raw = google_vision_ocr(cleaned)
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
# Google Sheets helpers (use secrets)
# ---------------------------
def _get_sheet_id():
    # flexible lookup for sheet id
    if "settings" in st.secrets and "sheet_id" in st.secrets["settings"]:
        return st.secrets["settings"]["sheet_id"]
    if "SHEET_ID" in st.secrets:
        return st.secrets["SHEET_ID"]
    raise KeyError("Mettez 'sheet_id' dans st.secrets['settings'] ou 'SHEET_ID' dans st.secrets")


def get_worksheet():
    # prefer gcp_sheet, fallback to google_service_account
    if "gcp_sheet" in st.secrets:
        sa_info = dict(st.secrets["gcp_sheet"])
    elif "google_service_account" in st.secrets:
        sa_info = dict(st.secrets["google_service_account"])
    else:
        raise FileNotFoundError("Credentials Google Sheets introuvables dans st.secrets")

    client = gspread.service_account_from_dict(sa_info)
    sheet_id = _get_sheet_id()
    sh = client.open_by_key(sheet_id)
    return sh.sheet1


def get_sheets_service():
    if "gcp_sheet" in st.secrets:
        sa_info = dict(st.secrets["gcp_sheet"])
    elif "google_service_account" in st.secrets:
        sa_info = dict(st.secrets["google_service_account"])
    else:
        raise FileNotFoundError("Credentials Google Sheets introuvables dans st.secrets")

    creds = SA_Credentials.from_service_account_info(sa_info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    service = build("sheets", "v4", credentials=creds)
    return service


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
    service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()

# ---------------------------
# UI - Upload (no camera as requested)
# ---------------------------
st.markdown("<div class='chancard'>", unsafe_allow_html=True)
uploaded = st.file_uploader("Importer une facture (jpg/png)", type=["jpg","jpeg","png"])
st.markdown("</div>", unsafe_allow_html=True)

img = None
if uploaded:
    img = Image.open(uploaded)

# prepare edited df storage
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

    month_detected = res.get("mois", "")
    months_list = ["","Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
    mois_val = col2.selectbox("📅 Mois", months_list, index=(0 if not month_detected else months_list.index(month_detected)))

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

    if st.button("➕ Ajouter une ligne"):
        new_row = pd.DataFrame([{"article": "", "bouteilles": 0}])
        edited_df = pd.concat([edited_df, new_row], ignore_index=True)
        st.session_state["edited_articles_df"] = edited_df
        st.experimental_rerun()

    st.session_state["edited_articles_df"] = edited_df.copy()

    st.subheader("Texte brut (résultat OCR)")
    st.code(res["raw"])

# ---------------------------
# Prepare worksheet (attempt, non-blocking)
# ---------------------------
try:
    ws = get_worksheet()
    sheet_id = ws.id
    spreadsheet_id = _get_sheet_id()
except Exception as e:
    ws = None
    sheet_id = None
    spreadsheet_id = None

# ---------------------------
# ENVOI -> Google Sheets
# ---------------------------
if img and st.session_state.get("edited_articles_df") is not None and ws and st.button("📤 Envoyer vers Google Sheets"):
    try:
        edited = st.session_state["edited_articles_df"].copy()
        edited = edited[~((edited["article"].astype(str).str.strip() == "") & (edited["bouteilles"] == 0))]
        edited["bouteilles"] = pd.to_numeric(edited["bouteilles"].fillna(0), errors="coerce").fillna(0).astype(int)

        start_row = len(ws.get_all_values()) + 1
        today_str = datetime.now().strftime("%d/%m/%Y")

        for _, row in edited.iterrows():
            ws.append_row([
                mois_val or "",
                doit_val or "",
                today_str,
                bon_commande_val or "",
                adresse_val or "",
                row.get("article", ""),
                int(row.get("bouteilles", 0)),
                st.session_state.user_nom
            ])

        end_row = len(ws.get_all_values())

        color = COLORS[st.session_state.get("scan_index", 0) % len(COLORS)]
        if spreadsheet_id and sheet_id is not None:
            color_rows(spreadsheet_id, sheet_id, start_row-1, end_row, color)

        st.session_state["scan_index"] = st.session_state.get("scan_index", 0) + 1

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


# ---------------------------
# requirements.txt (paste this into your requirements.txt file)
# ---------------------------
#
# streamlit
# pillow
# numpy
# google-cloud-vision
# gspread
# google-api-python-client
# google-auth
# pandas
#

# ---------------------------
# .streamlit/secrets.toml TEMPLATE
# ---------------------------
#
# [gcp_vision]
# type = "service_account"
# project_id = "chanfuiocr-478317"
# private_key_id = "..."
# private_key = """-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"""
# client_email = "ocr-chanfui@chanfuiocr-478317.iam.gserviceaccount.com"
# token_uri = "https://oauth2.googleapis.com/token"
#
# [gcp_sheet]
# type = "service_account"
# project_id = "chanfuishett"
# private_key_id = "..."
# private_key = """-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"""
# client_email = "python-api-chanfui@chanfuishett.iam.gserviceaccount.com"
# token_uri = "https://oauth2.googleapis.com/token"
#
# [settings]
# sheet_id = "TON_GOOGLE_SHEET_KEY"

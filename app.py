# app_streamlit_cloud.py — Version sans LOGIN
# ChanFui OCR — Streamlit Cloud ready

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
# Constants & UI styles
# ---------------------------
COLORS = [
    {"red": 0.8, "green": 1.0, "blue": 0.8},
    {"red": 0.15, "green": 0.15, "blue": 0.15},
    {"red": 1.0, "green": 0.7, "blue": 0.7},
]

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

st.markdown("<div class='wine-title'>ChanFui & Fils</div><div class='wine-sub'>Google Vision Premium Edition</div>", unsafe_allow_html=True)

# ---------------------------
# Scan index (persistant)
# ---------------------------
if "scan_index" not in st.session_state:
    try:
        st.session_state.scan_index = int(st.secrets.get("SCAN_STATE", {}).get("scan_index", 0))
    except Exception:
        st.session_state.scan_index = 0

# ---------------------------
# Image preprocessing
# ---------------------------
def preprocess_image(image_bytes: bytes) -> bytes:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    max_width = 2600
    if img.width > max_width:
        ratio = max_width / img.width
        new_h = int(img.height * ratio)
        img = img.resize((max_width, new_h), Image.LANCZOS)

    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

    out = BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()

# ---------------------------
# Google Vision OCR
# ---------------------------
def get_vision_client():
    if "gcp_vision" in st.secrets:
        sa_info = dict(st.secrets["gcp_vision"])
    elif "google_service_account" in st.secrets:
        sa_info = dict(st.secrets["google_service_account"])
    else:
        raise RuntimeError("Credentials Google Vision introuvables")

    creds = SA_Credentials.from_service_account_info(sa_info)
    return vision.ImageAnnotatorClient(credentials=creds)

def google_vision_ocr(img_bytes):
    client = get_vision_client()
    response = client.text_detection(image=vision.Image(content=img_bytes))
    if response.error and response.error.message:
        raise Exception(response.error.message)
    if response.text_annotations:
        return response.text_annotations[0].description
    return ""

# ---------------------------
# Cleaning & extraction helpers
# ---------------------------
def clean_text(t):
    t = t.replace("\r", "\n")
    t = t.replace("\n ", "\n")
    t = re.sub(r"[^\S\r\n]+", " ", t)
    t = t.replace("’", "'")
    t = re.sub(r"\s+\n", "\n", t)
    return t.strip()

def extract_invoice_number(text):
    p = r"FACTURE\s+EN\s+COMPTE.*?N[°o]?\s*([0-9]{3,})"
    m = re.search(p, text, flags=re.I)
    if m:
        return m.group(1)
    patterns = [r"FACTURE.*?N[°o]\s*([0-9]{3,})", r"FACTURE.*?N\s*([0-9]{3,})"]
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return m.group(1)
    m = re.search(r"N°\s*([0-9]{3,})", text)
    return m.group(1) if m else ""

def extract_delivery_address(text):
    m = re.search(r"Adresse de livraison\s*[:\-]\s*(.+)", text, flags=re.I)
    if m:
        return m.group(1).strip().rstrip(".")
    m2 = re.search(r"Adresse(?:\s+de\s+livraison)?\s*[:\-]?\s*\n?\s*(.+)", text, flags=re.I)
    if m2:
        return m2.group(1).split("\n")[0].strip()
    return ""

def extract_doit(text):
    p = r"\bDOIT\s*[:\-]?\s*([A-Z0-9]{2,6})"
    m = re.search(p, text, flags=re.I)
    if m:
        return m.group(1)
    for c in ["S2M", "ULYS", "DLP"]:
        if c in text:
            return c
    return ""

def extract_month(text):
    months = {
        "janvier":"Janvier","février":"Février","fevrier":"Février","mars":"Mars",
        "avril":"Avril","mai":"Mai","juin":"Juin","juillet":"Juillet","août":"Août",
        "aout":"Août","septembre":"Septembre","octobre":"Octobre","novembre":"Novembre",
        "décembre":"Décembre","decembre":"Décembre"
    }
    for m in months:
        if re.search(r"\b"+re.escape(m)+r"\b", text, flags=re.I):
            return months[m]
    return ""

def extract_bon_commande(text):
    m = re.search(r"Suivant votre bon de commande\s*[:\-]?\s*([0-9A-Za-z\-\/]+)", text, flags=re.I)
    if m:
        return m.group(1)
    m2 = re.search(r"bon de commande\s*[:\-]?\s*(.+)", text, flags=re.I)
    if m2:
        return m2.group(1).split()[0]
    return ""

def extract_items(text):
    items = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    pattern = re.compile(r"(.+?(?:75\s*cls?|75\s*cl|75cl|75))\s+\d+\s+\d+\s+(\d+)", flags=re.I)
    for l in lines:
        m = pattern.search(l)
        if m:
            items.append({"article": m.group(1).strip(), "bouteilles": int(m.group(2))})
    if not items:
        for l in lines:
            if "75" in l:
                nums = re.findall(r"(\d{1,4})", l)
                if nums:
                    items.append({"article": re.sub(r"\d+", "", l).strip(), "bouteilles": int(nums[-1])})
    return items

# ---------------------------
# Pipeline
# ---------------------------
def invoice_pipeline(image_bytes):
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
# Google Sheets helpers
# ---------------------------
def _get_sheet_id():
    if "settings" in st.secrets and "sheet_id" in st.secrets["settings"]:
        return st.secrets["settings"]["sheet_id"]
    if "SHEET_ID" in st.secrets:
        return st.secrets["SHEET_ID"]
    raise KeyError("Sheet ID introuvable")

def get_worksheet():
    if "gcp_sheet" in st.secrets:
        sa_info = dict(st.secrets["gcp_sheet"])
    else:
        sa_info = dict(st.secrets["google_service_account"])
    client = gspread.service_account_from_dict(sa_info)
    return client.open_by_key(_get_sheet_id()).sheet1

def get_sheets_service():
    if "gcp_sheet" in st.secrets:
        sa_info = dict(st.secrets["gcp_sheet"])
    else:
        sa_info = dict(st.secrets["google_service_account"])
    creds = SA_Credentials.from_service_account_info(sa_info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    return build("sheets", "v4", credentials=creds)

def color_rows(spreadsheet_id, sheet_id, start, end, color):
    service = get_sheets_service()
    body = {
        "requests":[
            {
                "repeatCell":{
                    "range":{"sheetId":sheet_id,"startRowIndex":start,"endRowIndex":end},
                    "cell":{"userEnteredFormat":{"backgroundColor":color}},
                    "fields":"userEnteredFormat.backgroundColor"
                }
            }
        ]
    }
    service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()

# ---------------------------
# Upload UI
# ---------------------------
st.markdown("<div class='chancard'>", unsafe_allow_html=True)
uploaded = st.file_uploader("Importer une facture (jpg/png)", type=["jpg","jpeg","png"])
st.markdown("</div>", unsafe_allow_html=True)

img = Image.open(uploaded) if uploaded else None

if "edited_articles_df" not in st.session_state:
    st.session_state["edited_articles_df"] = None

# ---------------------------
# OCR Processing
# ---------------------------
if img:
    st.image(img, caption="Aperçu", use_container_width=True)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    st.info("Traitement OCR Google Vision...")
    p = st.progress(10)
    try:
        res = invoice_pipeline(img_bytes)
        p.progress(100)
    except Exception as e:
        st.error(f"Erreur OCR: {e}")
        st.stop()
    p.empty()

    st.subheader("Informations détectées (modifiable)")
    col1, col2 = st.columns(2)
    facture_val = col1.text_input("🔢 Numéro de facture", value=res["facture"])
    bon_commande_val = col1.text_input("📦 Suivant bon de commande", value=res["bon_commande"])
    adresse_val = col2.text_input("📍 Adresse de livraison", value=res["adresse"])
    doit_val = col2.text_input("👤 DOIT", value=res["doit"])

    month_detected = res["mois"]
    months_list = ["","Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
    mois_val = col2.selectbox("📅 Mois", months_list, index=(0 if month_detected=="" else months_list.index(month_detected)))

    # Articles
    df_articles = pd.DataFrame(res["articles"] if res["articles"] else [{"article":"", "bouteilles":0}])
    st.subheader("Articles détectés (modifiable)")
    edited_df = st.data_editor(df_articles, num_rows="dynamic", use_container_width=True)

    if st.button("➕ Ajouter une ligne"):
        edited_df.loc[len(edited_df)] = {"article":"", "bouteilles":0}
        st.session_state["edited_articles_df"] = edited_df
        st.experimental_rerun()

    st.session_state["edited_articles_df"] = edited_df.copy()

    st.subheader("Texte brut (résultat OCR)")
    st.code(res["raw"])

# ---------------------------
# Google Sheet init
# ---------------------------
try:
    ws = get_worksheet()
    sheet_id = ws.id
    spreadsheet_id = _get_sheet_id()
except:
    ws = None
    sheet_id = None
    spreadsheet_id = None

# ---------------------------
# ENVOI Google Sheets
# ---------------------------
if img and st.session_state["edited_articles_df"] is not None and ws and st.button("📤 Envoyer vers Google Sheets"):
    try:
        edited = st.session_state["edited_articles_df"].copy()
        edited = edited[~((edited["article"].str.strip()=="") & (edited["bouteilles"]==0))]

        start_row = len(ws.get_all_values()) + 1
        today = datetime.now().strftime("%d/%m/%Y")

        for _, row in edited.iterrows():
            ws.append_row([
                mois_val,
                doit_val,
                today,
                bon_commande_val,
                adresse_val,
                row["article"],
                int(row["bouteilles"]),
                "SYSTEM"   # remplacement du user_nom
            ])

        end_row = len(ws.get_all_values())
        color = COLORS[st.session_state.scan_index % len(COLORS)]
        color_rows(spreadsheet_id, sheet_id, start_row-1, end_row, color)

        st.session_state.scan_index += 1

        st.success("Données envoyées avec succès !")
        st.json({
            "mois": mois_val,
            "doit": doit_val,
            "date": today,
            "bon_de_commande": bon_commande_val,
            "adresse": adresse_val,
            "nb_lignes_envoyees": len(edited)
        })

    except Exception as e:
        st.error(f"Erreur Sheets: {e}")

# ---------------------------
# Aperçu du Sheet
# ---------------------------
if ws and st.button("👀 Aperçu du Google Sheet"):
    try:
        df_sheet = pd.DataFrame(ws.get_all_records())
        st.dataframe(df_sheet.head(200), use_container_width=True)
    except Exception as e:
        st.error(f"Erreur sheet : {e}")

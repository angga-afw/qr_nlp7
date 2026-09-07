import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import google.generativeai as genai
import qrcode
import os
import base64
import csv
import json
import re
import time
from io import BytesIO
from datetime import datetime
from geopy.distance import geodesic
import logging
import traceback
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# 0. LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MedisAI Pro: Integrated Health Ecosystem",
    page_icon="🩺",
    layout="wide"
)

# Database Files
DATA_FILE = "medical_registry_v3.csv"
ENCOUNTER_FILE = "medical_encounters.csv"
COMPARISON_LOG_FILE = "model_comparison_results.csv"  # New file for paper data
MASTER_PASSWORD = "access123"

# Ensure Data Structures
DATA_STRUCTURE = [
    "User_ID", "BPJS_ID", "Name", "Birth_Date", "Gender",
    "Blood_Type", "Weight_kg", "Height_cm", "Chronic_Diseases",
    "Current_Medication", "Allergies", "Emergency_Contact_Name",
    "Emergency_Contact_Phone", "Blood_Pressure", "Oxygen_Saturation",
    "Hospitalization_History", "Responsible_Doctor", "Medical_History", "Last_Update"
]

ENCOUNTER_STRUCTURE = ["Timestamp", "User_ID", "S", "O", "A", "P"]
COMPARISON_STRUCTURE = [
    "Timestamp", "User_ID", "Model_Name", "Input_Narrative", 
    "S_Result", "O_Result", "A_Result", "P_Result", 
    "RR", "SpO2", "BPS", "HR", "AVPU", "Latency_sec",
    "Knowledge_Source", "RAG_History_Used", "RAG_Guidelines_Used", "RAG_Used"
]

if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=DATA_STRUCTURE).to_csv(DATA_FILE, index=False)

if not os.path.exists(ENCOUNTER_FILE):
    pd.DataFrame(columns=ENCOUNTER_STRUCTURE).to_csv(ENCOUNTER_FILE, index=False)

def normalize_comparison_log_file(path):
    if not os.path.exists(path):
        pd.DataFrame(columns=COMPARISON_STRUCTURE).to_csv(path, index=False)
        return

    try:
        df = pd.read_csv(path)
        expected_cols = list(COMPARISON_STRUCTURE)
        missing = [c for c in expected_cols if c not in df.columns]
        if not missing:
            return
        for col in missing:
            df[col] = ""
        df = df[expected_cols]
        df.to_csv(path, index=False)
        return
    except Exception:
        pass

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            pd.DataFrame(columns=COMPARISON_STRUCTURE).to_csv(path, index=False)
            return

        header = [h.strip() for h in rows[0]]
        if header and header[:len(COMPARISON_STRUCTURE)] == COMPARISON_STRUCTURE[:len(header)]:
            df_norm = pd.DataFrame(rows[1:], columns=COMPARISON_STRUCTURE[:len(header)])
            for col in COMPARISON_STRUCTURE:
                if col not in df_norm.columns:
                    df_norm[col] = ""
            df_norm = df_norm[COMPARISON_STRUCTURE]
            df_norm.to_csv(path, index=False)
            return

        normalized_rows = []
        base_cols = ["Timestamp", "Model_Name", "Input_Narrative", "S_Result", "O_Result", "A_Result", "P_Result", "RR", "SpO2", "BPS", "HR", "AVPU", "Latency_sec"]
        for row in rows[1:]:
            if not row or all((cell is None or str(cell).strip() == "") for cell in row):
                continue
            norm_row = row[:len(base_cols)] + [""] * (len(base_cols) - len(row[:len(base_cols)]))
            if len(row) > len(base_cols):
                extras = row[len(base_cols):]
                norm_row += extras[:4]
                norm_row += [""] * (len(COMPARISON_STRUCTURE) - len(norm_row))
            else:
                norm_row += [""] * (len(COMPARISON_STRUCTURE) - len(norm_row))
            normalized_rows.append(norm_row)

        if normalized_rows:
            pd.DataFrame(normalized_rows, columns=COMPARISON_STRUCTURE).to_csv(path, index=False)
        else:
            pd.DataFrame(columns=COMPARISON_STRUCTURE).to_csv(path, index=False)
    except Exception:
        pd.DataFrame(columns=COMPARISON_STRUCTURE).to_csv(path, index=False)

normalize_comparison_log_file(COMPARISON_LOG_FILE)

HOSPITALS = [
    {"name": "RSUP Dr. Sardjito", "lat": -7.7684, "lon": 110.3737},
    {"name": "RS JIH Yogyakarta", "lat": -7.7569, "lon": 110.4021},
    {"name": "RSA UGM", "lat": -7.7423, "lon": 110.3492},
]

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 15px; padding: 15px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .soap-container { background-color: white; padding: 25px; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 5px solid #007bff; }
    .soap-header { color: #007bff; font-weight: bold; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; margin-bottom: 20px; }
    .soap-label { font-weight: bold; color: #495057; font-size: 0.85em; text-transform: uppercase; }
    .soap-content { color: #212529; font-size: 1.05em; line-height: 1.4; margin-bottom: 15px; }
    .metric-card { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center; color: #212529; }
    .profile-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #212529;
    }
    .profile-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #f1f1f1;
    }
    .profile-label {
        font-weight: bold;
        color: #6c757d;
        font-size: 0.85em;
        text-transform: uppercase;
    }
    .profile-value {
        color: #212529;
        font-weight: 500;
        text-align: right;
        max-width: 60%;
    }
    .new-data {
        color: #d32f2f;
        font-weight: bold;
        background: #fff5f5;
        padding: 2px 6px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state: st.session_state.messages = []
if "soap_record" not in st.session_state: 
    st.session_state.soap_record = {"S": "-", "O": "-", "A": "-", "P": "-"}
if "extracted_vitals" not in st.session_state:
    st.session_state.extracted_vitals = {}
if "retrieved_history" not in st.session_state:
    st.session_state.retrieved_history = []
if "retrieved_guidelines" not in st.session_state:
    st.session_state.retrieved_guidelines = []
if "rag_debug_history" not in st.session_state:
    st.session_state.rag_debug_history = []

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

# --- RAG HELPERS ---
GUIDELINES_FILE = "clinical_guidelines.txt"

def get_embedding(text, api_key, task_type="retrieval_document"):
    if not text or not text.strip():
        return None
    try:
        genai.configure(api_key=api_key)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def cosine_similarity_val(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_prod / (norm_v1 * norm_v2))

def load_guideline_chunks():
    if not os.path.exists(GUIDELINES_FILE):
        return []
    with open(GUIDELINES_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    sections = content.split("=== PANDUAN KLINIS: ")
    chunks = []
    for sec in sections:
        if not sec.strip():
            continue
        parts = sec.split("===", 1)
        if len(parts) == 2:
            title = parts[0].strip()
            body = parts[1].strip()
            chunks.append({
                "title": f"Panduan Klinis: {title}",
                "content": body
            })
        else:
            chunks.append({
                "title": "Panduan Umum",
                "content": sec.strip()
            })
    return chunks

def get_cached_guidelines(api_key):
    if "guidelines_cache" not in st.session_state:
        chunks = load_guideline_chunks()
        cached = []
        for chunk in chunks:
            text_to_embed = f"{chunk['title']}\n{chunk['content']}"
            emb = get_embedding(text_to_embed, api_key, task_type="retrieval_document")
            cached.append({"chunk": chunk, "embedding": emb})
        st.session_state.guidelines_cache = cached
    return st.session_state.guidelines_cache

def retrieve_guidelines(query, api_key, top_k=2):
    query_emb = get_embedding(query, api_key, task_type="retrieval_query")
    if not query_emb:
        return []
    
    cached_data = get_cached_guidelines(api_key)
    results = []
    for item in cached_data:
        if item["embedding"]:
            sim = cosine_similarity_val(query_emb, item["embedding"])
            results.append((sim, item["chunk"]))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in results[:top_k] if r[0] > 0.35]

def retrieve_patient_history(query, user_id, api_key, top_k=2):
    if not os.path.exists(ENCOUNTER_FILE):
        return []
    df_e = pd.read_csv(ENCOUNTER_FILE)
    p_encounters = df_e[df_e["User_ID"].astype(str) == str(user_id)]
    if p_encounters.empty:
        return []
    
    query_emb = get_embedding(query, api_key, task_type="retrieval_query")
    if not query_emb:
        return []
    
    results = []
    for _, row in p_encounters.iterrows():
        enc_text = f"Tanggal: {row.get('Timestamp', '-')}\n" \
                   f"Subjective: {row.get('S', '-')}\n" \
                   f"Objective: {row.get('O', '-')}\n" \
                   f"Assessment: {row.get('A', '-')}\n" \
                   f"Plan: {row.get('P', '-')}"
        
        enc_emb = get_embedding(enc_text, api_key, task_type="retrieval_document")
        if enc_emb:
            sim = cosine_similarity_val(query_emb, enc_emb)
            results.append((sim, {
                "timestamp": row.get('Timestamp', '-'),
                "S": row.get('S', '-'),
                "O": row.get('O', '-'),
                "A": row.get('A', '-'),
                "P": row.get('P', '-'),
                "full_text": enc_text
            }))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in results[:top_k] if r[0] > 0.35]

def calculate_news2(rr, spo2, bps, hr, avpu):
    score = 0
    if rr <= 8 or rr >= 25: score += 3
    elif 21 <= rr <= 24: score += 2
    if spo2 <= 91: score += 3
    elif 92 <= spo2 <= 93: score += 2
    if bps <= 90 or bps >= 220: score += 3
    elif 91 <= bps <= 100: score += 2
    if hr <= 40 or hr >= 131: score += 3
    elif 111 <= hr <= 130: score += 2
    if avpu != "Alert": score += 3
    
    if score >= 7: return score, "RED (CRITICAL)", "#d32f2f"
    elif score >= 5: return score, "ORANGE (URGENT)", "#f57c00"
    return score, "GREEN (STABLE)", "#388e3c"


def parse_int_value(value, default=None):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ["nan", "none", "null"]:
            return default
        match = re.search(r"(\d{1,3})", cleaned)
        if match:
            return int(match.group(1))

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_systolic_bp(value, default=120):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ["nan", "none", "null"]:
            return default
        if "/" in cleaned:
            cleaned = cleaned.split("/")[0].strip()
        match = re.search(r"(\d{2,3})", cleaned)
        if match:
            return int(match.group(1))

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def get_triage_default_vitals(active_patient=None, extracted=None):
    extracted = extracted or {}
    defaults = {
        "rr": extracted.get("rr", 20),
        "spo2": extracted.get("spo2", 95),
        "bps": extracted.get("bps", 120),
        "hr": extracted.get("hr", 80),
        "avpu": extracted.get("avpu", "Alert"),
    }

    if active_patient is not None:
        patient_spo2 = parse_int_value(active_patient.get("Oxygen_Saturation"), defaults["spo2"])
        patient_bps = parse_systolic_bp(active_patient.get("Blood_Pressure"), defaults["bps"])
        defaults["spo2"] = patient_spo2
        defaults["bps"] = patient_bps

    # Preserve AI-extracted values only when patient data is absent or not available.
    for key, value in extracted.items():
        if value is None or value == "":
            continue
        if key in ["spo2", "bps"] and active_patient is not None:
            continue
        if key == "avpu" and active_patient is not None and active_patient.get("Name"):
            continue
        defaults[key] = value

    return defaults


def build_rag_debug_summary(query, retrieved_history, retrieved_guidelines, patient_data=None):
    reasons = []
    if not query or not str(query).strip():
        reasons.append("Query kosong.")
    if patient_data is None:
        reasons.append("Pasien belum dipilih / profil tidak tersedia.")
    if retrieved_history:
        reasons.append(f"Riwayat pasien relevan ditemukan: {len(retrieved_history)} item.")
    else:
        reasons.append("Tidak ada riwayat pasien yang melewati threshold similaritas.")
    if retrieved_guidelines:
        reasons.append(f"Panduan klinis relevan ditemukan: {len(retrieved_guidelines)} item.")
    else:
        reasons.append("Tidak ada guideline yang melewati threshold similaritas.")

    rag_active = bool(retrieved_history or retrieved_guidelines)
    return {
        "rag_active": rag_active,
        "knowledge_source": "RAG" if rag_active else "General",
        "history_count": len(retrieved_history),
        "guideline_count": len(retrieved_guidelines),
        "query_length": len((query or "").strip()),
        "patient_selected": bool(patient_data is not None),
        "reasons": reasons,
    }


def process_narrative(narrative, api_key, patient_data=None):
    if not api_key:
        logger.error("API Key missing for narrative processing.")
        return None
    
    start_time = time.time()
    model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")
    
    try:
        logger.info(f"--- Unified App: Processing Request ---")
        logger.info(f"Using Model: {model_name}")
        logger.info(f"Input Narrative: {narrative[:200]}...")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # RAG 1: Retrieve Patient History
        retrieved_history = []
        if patient_data is not None and "User_ID" in patient_data:
            try:
                retrieved_history = retrieve_patient_history(narrative, patient_data["User_ID"], api_key)
                logger.info(f"RAG Patient History retrieved: {len(retrieved_history)} records")
            except Exception as e:
                logger.error(f"Error in patient history RAG: {e}")
                
        # RAG 2: Retrieve Clinical Guidelines
        retrieved_guidelines = []
        try:
            retrieved_guidelines = retrieve_guidelines(narrative, api_key)
            logger.info(f"RAG Clinical Guidelines retrieved: {len(retrieved_guidelines)} chunks")
        except Exception as e:
            logger.error(f"Error in clinical guidelines RAG: {e}")

        patient_context = ""
        if patient_data is not None:
            patient_context = f"""
            DATA PROFIL MEDIS PASIEN:
            - Nama: {patient_data.get('Name')}
            - Tanggal Lahir: {patient_data.get('Birth_Date')}
            - Jenis Kelamin: {patient_data.get('Gender')}
            - Golongan Darah: {patient_data.get('Blood_Type')}
            - Berat/Tinggi: {patient_data.get('Weight_kg')}kg / {patient_data.get('Height_cm')}cm
            - Riwayat Penyakit Kronis: {patient_data.get('Chronic_Diseases')}
            - Konsumsi Obat: {patient_data.get('Current_Medication')}
            - Alergi: {patient_data.get('Allergies')}
            - Tekanan Darah Terakhir: {patient_data.get('Blood_Pressure')}
            - Saturasi Oksigen Terakhir: {patient_data.get('Oxygen_Saturation')}%
            - Riwayat Rawat Inap: {patient_data.get('Hospitalization_History')}
            - Dokter PJ: {patient_data.get('Responsible_Doctor')}
            - Riwayat Medis: {patient_data.get('Medical_History')}
            """

        history_context = ""
        if retrieved_history:
            history_context = "\nRIWAYAT REKAM MEDIS RELEVAN DARI RAG:\n" + "\n---\n".join([h["full_text"] for h in retrieved_history])
        
        guidelines_context = ""
        if retrieved_guidelines:
            guidelines_context = "\nPANDUAN KLINIS RELEVAN DARI RAG:\n" + "\n---\n".join([f"Judul: {g['title']}\nKonten:\n{g['content']}" for g in retrieved_guidelines])

        prompt = f"""
        Anda adalah asisten medis profesional. Analisis keluhan atau kondisi yang disampaikan oleh PASIEN berikut.
        Gunakan data profil pasien, riwayat rekam medis lama yang relevan, serta panduan klinis pendukung untuk membuat rekam medis SOAP berkualitas tinggi dan berbasis panduan medis.
        
        1. Konversi menjadi format SOAP (Subjective, Objective, Assessment, Plan) dari sudut pandang medis.
           - Subjective: Keluhan utama dan riwayat penyakit dari pasien.
           - Objective: Jika ada data fisik yang disebutkan (tanda vital: tensi, nadi, saturasi, nafas, kesadaran).
           - Assessment: Kemungkinan diagnosis atau ringkasan kondisi (hubungkan dengan riwayat lama jika ada kesamaan gejala).
           - Plan: Saran tindakan, obat, atau pemeriksaan lanjutan (sesuaikan dengan panduan klinis jika relevan).
        2. Deteksi apakah ada informasi baru yang dapat memperbarui profil medis pasien (seperti alergi baru, riwayat penyakit kronis baru, obat yang sedang dikonsumsi, tekanan darah, atau kadar oksigen).
        3. EKSTRAKSI DATA VITAL (NEWS2): Jika pasien menyebutkan angka-angka berikut, ambil nilainya:
           - Respirasi (RR): bpm
           - SpO2: %
           - Tekanan Darah Sistolik: angka pertama (misal 120 dari 120/80)
           - Denyut Jantung (HR): bpm
           - Kesadaran (AVPU): Alert, Voice, Pain, atau Unresponsive
        
        DATA PROFIL MEDIS SAAT INI:
        {patient_context}
        {history_context}
        {guidelines_context}
        
        KELUHAN/NARASI PASIEN: 
        "{narrative}"
        
        WAJIB OUTPUT DALAM FORMAT JSON BERIKUT:
        {{
            "soap": {{
                "S": "...",
                "O": "...",
                "A": "...",
                "P": "..."
            }},
            "profile_updates": {{
                "Chronic_Diseases": "isi jika ada info baru",
                "Current_Medication": "isi jika ada info baru",
                "Allergies": "isi jika ada info baru",
                "Blood_Pressure": "isi jika ada info baru (format 120/80)",
                "Oxygen_Saturation": "isi jika ada info baru (angka)",
                "Hospitalization_History": "isi jika ada info baru",
                "Responsible_Doctor": "isi jika ada info baru",
                "Weight_kg": "isi jika ada info baru",
                "Height_cm": "isi jika ada info baru",
                "Medical_History": "tambahkan riwayat baru jika ada"
            }},
            "triage_vitals": {{
                "rr": "angka (int) or null",
                "spo2": "angka (int) or null",
                "bps": "angka (int) or null",
                "hr": "angka (int) or null",
                "avpu": "Alert/Voice/Pain/Unresponsive atau null"
            }}
        }}
        
        Hanya berikan JSON saja.
        """
        
        response = model.generate_content(prompt)
        logger.info(f"AI Raw Response: {response.text}")
        
        text = response.text.strip().replace('```json', '').replace('```', '')
        result = json.loads(text)
        
        # Normalization
        rag_debug = build_rag_debug_summary(narrative, retrieved_history, retrieved_guidelines, patient_data)
        final_result = {
            "soap": {"S": "-", "O": "-", "A": "-", "P": "-"}, 
            "profile_updates": {},
            "triage_vitals": {},
            "retrieved_history": retrieved_history,
            "retrieved_guidelines": retrieved_guidelines,
            "rag_debug": rag_debug
        }
        
        if "soap" in result:
            final_result["soap"] = result["soap"]
        elif "S" in result: # Handle flat structure if AI fails
            final_result["soap"] = {k: result.get(k, "-") for k in ["S", "O", "A", "P"]}
            
        if "profile_updates" in result:
            final_result["profile_updates"] = result["profile_updates"]
            
        if "triage_vitals" in result:
            final_result["triage_vitals"] = result["triage_vitals"]
            
        logger.info("Successfully parsed and normalized JSON response.")
        
        # LOGGING FOR COMPARISON (PAPER DATA)
        latency = time.time() - start_time
        rag_history_used = bool(retrieved_history)
        rag_guidelines_used = bool(retrieved_guidelines)
        rag_used = rag_history_used or rag_guidelines_used
        knowledge_source = "RAG" if rag_used else "General"
        try:
            patient_id = ""
            if patient_data is not None:
                patient_id = str(patient_data.get("User_ID", "")).strip()

            log_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "User_ID": patient_id,
                "Model_Name": model_name,
                "Input_Narrative": narrative,
                "S_Result": final_result["soap"].get("S", ""),
                "O_Result": final_result["soap"].get("O", ""),
                "A_Result": final_result["soap"].get("A", ""),
                "P_Result": final_result["soap"].get("P", ""),
                "RR": final_result["triage_vitals"].get("rr", ""),
                "SpO2": final_result["triage_vitals"].get("spo2", ""),
                "BPS": final_result["triage_vitals"].get("bps", ""),
                "HR": final_result["triage_vitals"].get("hr", ""),
                "AVPU": final_result["triage_vitals"].get("avpu", ""),
                "Latency_sec": round(latency, 2),
                "Knowledge_Source": knowledge_source,
                "RAG_History_Used": int(rag_history_used),
                "RAG_Guidelines_Used": int(rag_guidelines_used),
                "RAG_Used": int(rag_used)
            }
            pd.DataFrame([log_data]).to_csv(COMPARISON_LOG_FILE, mode='a', header=False, index=False)
            logger.info(f"Comparison log entry added for model {model_name} with source={knowledge_source}")
        except Exception as log_err:
            logger.error(f"Failed to write comparison log: {log_err}")
 
        return final_result
    except Exception as e:
        error_msg = f"Error in AI Processing (Unified): {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None

# ==========================================
# 3. SIDEBAR: GLOBAL CONTROL
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=80)
    st.title("MedisAI Pro")
    
    st.divider()
    
    # API status check
    api_key_env = os.getenv("GOOGLE_API_KEY", "")
    model_name_env = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")
    
    st.markdown("### 🤖 AI Configuration")
    selected_model = st.selectbox(
        "Select Model (for Paper/Test)",
        ["gemini-1.5-flash", "gemini-3.1-flash-lite", "gemini-1.0-pro"],
        index=0 if model_name_env == "gemini-1.5-flash" else 1
    )
    # Update environment variable for current session
    os.environ["GOOGLE_MODEL_NAME"] = selected_model

    if api_key_env:
        st.success(f"✅ AI Connected ({selected_model})")
    else:
        st.error("❌ AI Key Missing (.env)")
    
    st.divider()
    
    # Global Patient Selector
    df_p = pd.read_csv(DATA_FILE)
    scanned_uid = st.query_params.get("uid")
    
    active_patient = None
    
    if scanned_uid:
        try:
            # Automatic selection from QR code
            matches = df_p[df_p["User_ID"].astype(str) == str(scanned_uid)]
            if not matches.empty:
                active_patient = matches.iloc[0]
                st.success(f"🔓 Active Session: {active_patient['Name']}")
                st.info(f"ID: {active_patient['User_ID']}")
            else:
                st.error(f"❌ Patient with ID {scanned_uid} was not found.")
        except Exception as e:
            st.error(f"Error accessing QR data: {e}")
    
    if active_patient is None or (isinstance(active_patient, pd.Series) and active_patient.empty):
        st.warning("📥 Waiting for QR Code scan...")
        st.info("This system is designed for fast access via QR Code. Scan the patient's QR Code to begin.")
    
    st.divider()
    
    # Hide manual selector and show current status
    st.markdown("### 🛠️ Device Status")
    st.caption("Manual Selection: **DISABLED** (QR-Only Mode)")

# ==========================================
# 4. MAIN INTERFACE (TABS)
# ==========================================
# Streamlit standard tabs don't support programmatic switching yet.
# To achieve the requirement, we reorder the tabs if UID is present.

tab_list = [
    "📝 Registrasi", 
    "💬 AI Chat SOAP", 
    "🕒 Medical History",
    "🚑 Emergency (CDSS)", 
    "🔐 Admin"
]

if st.query_params.get("uid"):
    # Move Emergency to the first position
    tab_list.insert(0, tab_list.pop(3))

t_emergency_or_reg, t_chat_or_reg, t_hist, t_emergency_or_else, t_admin = st.tabs(tab_list)

# Assign tabs back to meaningful names based on their content
if st.query_params.get("uid"):
    t_emergency, t_reg, t_chat, t_hist, t_admin = t_emergency_or_reg, t_chat_or_reg, t_hist, t_emergency_or_else, t_admin
else:
    t_reg, t_chat, t_hist, t_emergency, t_admin = t_emergency_or_reg, t_chat_or_reg, t_hist, t_emergency_or_else, t_admin

# --- TAB 1: REGISTRATION ---
with t_reg:
    st.subheader("New Patient Registration")
    
    # Auto-generate User ID
    next_id = 1
    if not df_p.empty:
        try:
            # Mengambil angka terakhir dari User_ID (Format MAI-2026-0001)
            # Kita pecah berdasarkan '-' dan ambil bagian terakhir
            last_ids = df_p["User_ID"].astype(str).str.split("-").str[-1]
            valid_ids = pd.to_numeric(last_ids, errors='coerce').dropna()
            if not valid_ids.empty:
                next_id = int(valid_ids.max()) + 1
        except Exception as e:
            logger.error(f"Error generating next ID: {e}")
            next_id = len(df_p) + 1
    
    auto_id = f"MAI-{datetime.now().year}-{next_id:04d}"

    with st.form("reg_form", clear_on_submit=True):
        d = {}
        st.markdown("#### Patient Identity")
        identity_left, identity_right = st.columns(2)
        with identity_left:
            d["User_ID"] = st.text_input("User ID", value=auto_id, help="This ID is generated automatically", disabled=True)
            d["BPJS_ID"] = st.text_input("BPJS_ID")
            d["Name"] = st.text_input("Full Name")
        with identity_right:
            d["Birth_Date"] = st.date_input(
                "Date of Birth",
                value=datetime(1990, 1, 1).date(),
                min_value=datetime(1900, 1, 1).date(),
                max_value=datetime.now().date()
            )
            d["Gender"] = st.selectbox("Gender", ["Male", "Female"])

        st.markdown("#### Physical Condition")
        physical_left, physical_right = st.columns(2)
        with physical_left:
            d["Blood_Type"] = st.selectbox("Blood Type", ["A", "B", "AB", "O"])
            d["Weight_kg"] = st.number_input("Weight (kg)", 0, 200)
            d["Height_cm"] = st.number_input("Height (cm)", 0, 250)
        with physical_right:
            d["Blood_Pressure"] = st.text_input("Blood Pressure (mmHg)", placeholder="120/80")
            d["Oxygen_Saturation"] = st.number_input("Oxygen Saturation (%)", 0, 100, 98)

        st.markdown("#### Medical Information")
        medical_left, medical_right = st.columns(2)
        with medical_left:
            d["Chronic_Diseases"] = st.text_area("Chronic Diseases")
            d["Current_Medication"] = st.text_area("Current Medication")
            d["Allergies"] = st.text_area("Allergies")
        with medical_right:
            d["Hospitalization_History"] = st.text_area("Hospitalization History")
            d["Medical_History"] = st.text_area("Medical History")
            d["Responsible_Doctor"] = st.text_input("Attending Doctor")

        st.markdown("#### Emergency Contact")
        contact_left, contact_right = st.columns(2)
        with contact_left:
            d["Emergency_Contact_Name"] = st.text_input("Emergency Contact Name")
        with contact_right:
            d["Emergency_Contact_Phone"] = st.text_input("Emergency Contact Phone")

        if st.form_submit_button("Save & Generate QR"):
            # Use the auto-generated ID directly as the field is disabled in form
            d["User_ID"] = auto_id
            if d["User_ID"] and d["Name"]:
                d["Last_Update"] = datetime.now().strftime("%Y-%m-%d")
                df_p = pd.concat([df_p, pd.DataFrame([d])], ignore_index=True)
                df_p.to_csv(DATA_FILE, index=False)
                st.success("Patient registered successfully!")
                
                # Mendeteksi domain secara otomatis untuk QR Code
                # Jika di Streamlit Cloud, gunakan query parameter 'host' atau default ke apps domain
                # Cara terbaik di Streamlit Cloud adalah menggunakan headers (memerlukan host rahasia)
                # atau cara termudah: deteksi dari URL browser via st.query_params
                
                # Coba dapatkan host dari sistem
                current_host = "localhost:8501"
                
                # Jika kita ingin benar-benar dinamis di Streamlit Cloud tanpa hardcode:
                # Kita bisa menggunakan trik JavaScript minimal untuk mendapatkan origin, 
                # tapi Streamlit tidak mendukung ini secara native di backend.
                # Jadi kita gunakan solusi paling robust: 
                # Izinkan user mendefinisikan BASE_URL di Secrets untuk Production
                
                base_url = os.getenv("BASE_URL")
                
                if base_url:
                    qr_url = f"{base_url}/?uid={d['User_ID']}"
                else:
                    # Fallback jika BASE_URL tidak diset (lokal) tetap localhost
                    qr_url = f"http://localhost:8501/?uid={d['User_ID']}"
                
                img = qrcode.make(qr_url)
                buf = BytesIO(); img.save(buf, format="PNG")
                qr_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
                components.html(
                    f"""
                    <style>
                        body {{ font-family: Arial, sans-serif; text-align: center; margin: 0; font-size: 10px; }}
                        .qr-versions {{ display: flex; justify-content: center; gap: 18px; align-items: flex-start; }}
                        .qr-version {{ text-align: center; font-size: 9px; }}
                        .qr-frame {{ display: flex; align-items: center; justify-content: center; width: 70px; height: 70px; margin: 0 auto 4px; border: 1px solid #d32f2f; border-radius: 50%; background: white; box-sizing: border-box; }}
                        .square-frame {{ display: flex; align-items: center; justify-content: center; width: 70px; height: 70px; margin: 0 auto 4px; border: 1px solid #555; background: white; box-sizing: border-box; }}
                        img {{ display: block; width: 60px; height: 60px; }}
                        .square-frame img {{ width: 70px; height: 70px; }}
                        p {{ margin: 0 0 12px; font-weight: bold; }}
                        button {{ padding: 8px 14px; border: 1px solid #888; border-radius: 6px; background: white; cursor: pointer; }}
                        @media print {{ button {{ display: none; }} body.printing-round .square-print {{ display: none; }} body.printing-square .round-print {{ display: none; }} .square-frame {{ width: 1.5cm; height: 1.5cm; }} .square-frame img {{ width: 1.4cm; height: 1.4cm; }} }}
                        .print-button {{ margin: 4px; }}
                    </style>
                    <div class="qr-versions">
                        <div class="qr-version round-print"><div class="qr-frame"><img src="data:image/png;base64,{qr_base64}" alt="Round QR Code {d['User_ID']}"></div><div></div></div>
                        <div class="qr-version square-print"><div class="square-frame"><img src="data:image/png;base64,{qr_base64}" alt="Square QR Code {d['User_ID']}"></div><div></div></div>
                    </div>
                    <p>UID: {d['User_ID']}</p>
                    <button class="print-button" type="button" onclick="document.body.classList.add('printing-round'); window.print();">🖨️ Print Round QR</button>
                    <button class="print-button" type="button" onclick="document.body.classList.add('printing-square'); window.print();">🖨️ Print Square QR</button>
                    <script>window.addEventListener('afterprint', function() {{ document.body.classList.remove('printing-round', 'printing-square'); }});</script>
                    """,
                    height=350,
                )
            else: st.error("ID and name are required!")

# --- TAB 2: AI CHAT SOAP ---
with t_chat:
    if active_patient is None or (isinstance(active_patient, pd.Series) and active_patient.empty):
        st.warning("Scan the patient's QR Code to access AI Chat.")
    else:
        st.info("💡 **For the patient:** Describe your symptoms naturally (for example: 'I have had a headache and nausea since yesterday'). AI will summarize them for the doctor.")
        c_l, c_r = st.columns([1, 1])
        with c_l:
            st.subheader(f"💬 Patient: {active_patient['Name']}")
            # Clear button
            if st.button("Clear Chat"): st.session_state.messages = []; st.rerun()
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
            if prompt := st.chat_input("Hello, describe your symptoms here..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                
                with st.spinner("AI is analyzing your symptoms..."):
                    ai_response = process_narrative(prompt, api_key_env, active_patient)
                    if ai_response:
                        st.session_state.soap_record = ai_response.get("soap", {})
                        st.session_state.profile_updates = ai_response.get("profile_updates", {})
                        st.session_state.retrieved_history = ai_response.get("retrieved_history", [])
                        st.session_state.retrieved_guidelines = ai_response.get("retrieved_guidelines", [])
                        rag_debug = ai_response.get("rag_debug", {})
                        if rag_debug:
                            st.session_state.rag_debug_history.insert(0, {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                **rag_debug
                            })
                            st.session_state.rag_debug_history = st.session_state.rag_debug_history[:10]
                        # Update extracted vitals for Triage
                        new_vitals = ai_response.get("triage_vitals", {})
                        if new_vitals:
                            # Only update if not null
                            for k, v in new_vitals.items():
                                if v is not None:
                                    st.session_state.extracted_vitals[k] = v
                        
                        st.session_state.messages.append({"role": "assistant", "content": "Thank you. Your symptoms have been summarized in medical format. Review the SOAP preview and Triage section."})
                        st.rerun()
        
        with c_r:
            st.subheader("📋 SOAP Preview")
            st.markdown(f"""
            <div class="soap-container">
                <div class="soap-header">FORMAT SOAP <br><small>{datetime.now().strftime('%d/%m/%Y')}</small></div>
                <div class="soap-label">Subjective</div><div class="soap-content">{st.session_state.soap_record.get('S', '-')}</div>
                <div class="soap-label">Objective</div><div class="soap-content">{st.session_state.soap_record.get('O', '-')}</div>
                <div class="soap-label">Assessment</div><div class="soap-content">{st.session_state.soap_record.get('A', '-')}</div>
                <div class="soap-label">Plan</div><div class="soap-content">{st.session_state.soap_record.get('P', '-')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display RAG References
            if st.session_state.get("retrieved_history") or st.session_state.get("retrieved_guidelines"):
                with st.expander("🔍 Referensi RAG Terkait (Context)", expanded=True):
                    if st.session_state.get("retrieved_history"):
                        st.markdown("**Related Patient Medical History (CSV RAG):**")
                        for idx, hist in enumerate(st.session_state.retrieved_history):
                            st.caption(f"**{idx+1}. Date: {hist['timestamp']}** (Assessment: *{hist['A']}*)")
                            st.write(f"- **S**: {hist['S']}")
                            st.write(f"- **P**: {hist['P']}")
                    
                    if st.session_state.get("retrieved_guidelines"):
                        st.markdown("**Related Clinical Guidelines (External Doc RAG):**")
                        for idx, guide in enumerate(st.session_state.retrieved_guidelines):
                            st.markdown(f"📖 **{guide['title']}**")
                            st.text(guide['content'])
            
            if st.button("💾 Save Medical Record (SOAP)", use_container_width=True, type="primary"):
                new_enc = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "User_ID": str(active_patient["User_ID"]),
                    **st.session_state.soap_record
                }
                df_e = pd.read_csv(ENCOUNTER_FILE)
                df_e = pd.concat([df_e, pd.DataFrame([new_enc])], ignore_index=True)
                df_e.to_csv(ENCOUNTER_FILE, index=False)
                st.success("Medical record saved!")

            st.divider()
            st.subheader("🧬 Medical Profile Preview")
            
            # Show current vs updated data
            updates = st.session_state.get("profile_updates", {})
            
            def get_display_val(key, default_val):
                new_val = updates.get(key)
                if new_val and str(new_val).lower() != str(default_val).lower():
                    return f"<span class='new-data'>{new_val} (Baru)</span>"
                return default_val if pd.notna(default_val) else "-"

            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-row">
                    <div class="profile-label">Chronic Diseases</div>
                    <div class="profile-value">{get_display_val('Chronic_Diseases', active_patient['Chronic_Diseases'])}</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Current Medication</div>
                    <div class="profile-value">{get_display_val('Current_Medication', active_patient['Current_Medication'])}</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Allergies</div>
                    <div class="profile-value">{get_display_val('Allergies', active_patient['Allergies'])}</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Tekanan Darah</div>
                    <div class="profile-value">{get_display_val('Blood_Pressure', active_patient['Blood_Pressure'])}</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Saturasi Oksigen</div>
                    <div class="profile-value">{get_display_val('Oxygen_Saturation', active_patient['Oxygen_Saturation'])}%</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Hospitalization History</div>
                    <div class="profile-value">{get_display_val('Hospitalization_History', active_patient['Hospitalization_History'])}</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Attending Doctor</div>
                    <div class="profile-value">{get_display_val('Responsible_Doctor', active_patient['Responsible_Doctor'])}</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Berat Badan</div>
                    <div class="profile-value">{get_display_val('Weight_kg', active_patient['Weight_kg'])} kg</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Tinggi Badan</div>
                    <div class="profile-value">{get_display_val('Height_cm', active_patient['Height_cm'])} cm</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if updates:
                if st.button("🆙 Update Patient Profile (Smart Merge)", use_container_width=True):
                    try:
                        df_all = pd.read_csv(DATA_FILE)
                        # Cast text columns to object dtype to prevent dtype float64 errors when columns are empty
                        for col in ["Chronic_Diseases", "Current_Medication", "Allergies", "Medical_History", "Hospitalization_History", "Responsible_Doctor", "Blood_Pressure"]:
                            if col in df_all.columns:
                                df_all[col] = df_all[col].astype(object)
                        
                        # Find the row to update
                        idx = df_all[df_all["User_ID"].astype(str) == str(active_patient["User_ID"])].index[0]
                        
                        # Smart Merge Logic
                        append_fields = ["Chronic_Diseases", "Current_Medication", "Allergies", "Medical_History", "Hospitalization_History"]
                        replace_fields = ["Weight_kg", "Height_cm", "Blood_Pressure", "Oxygen_Saturation", "Responsible_Doctor"]
                        
                        for k, v in updates.items():
                            if k in df_all.columns and v:
                                # Data cleaning: convert "nan" string or empty values from AI to None
                                clean_val = str(v).strip()
                                if clean_val.lower() in ["nan", "none", "null", ""]:
                                    continue

                                if k in append_fields:
                                    current_val = str(df_all.at[idx, k]) if pd.notna(df_all.at[idx, k]) else ""
                                    
                                    # Merge and remove duplicates
                                    existing_items = [i.strip().lower() for i in current_val.split(",") if i.strip()]
                                    new_items = [i.strip() for i in clean_val.split(",") if i.strip()]
                                    
                                    merged_list = current_val.split(",") if current_val else []
                                    for item in new_items:
                                        if item.lower().strip() not in existing_items:
                                            merged_list.append(item.strip())
                                    
                                    df_all.at[idx, k] = ", ".join([i.strip() for i in merged_list if i.strip()])
                                elif k in replace_fields:
                                    # Handle numeric fields safely
                                    if k in ["Weight_kg", "Height_cm", "Oxygen_Saturation"]:
                                        try:
                                            # Strip any non-numeric characters except decimal point
                                            num_val = "".join(filter(lambda x: x.isdigit() or x == '.', clean_val))
                                            df_all.at[idx, k] = float(num_val) if num_val else df_all.at[idx, k]
                                        except:
                                            pass # Keep old value if conversion fails
                                    else:
                                        df_all.at[idx, k] = clean_val
                        
                        df_all.at[idx, "Last_Update"] = datetime.now().strftime("%Y-%m-%d")
                        df_all.to_csv(DATA_FILE, index=False)
                        st.success("Patient profile updated with Smart Merge!")
                        st.session_state.profile_updates = {}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update profile: {e}")

# --- TAB 3: MEDICAL HISTORY ---
with t_hist:
    if active_patient is not None and not isinstance(active_patient, pd.Series) or (isinstance(active_patient, pd.Series) and not active_patient.empty):
        st.subheader(f"🕒 Medical History: {active_patient['Name']}")
        df_e = pd.read_csv(ENCOUNTER_FILE)
        p_hist = df_e[df_e["User_ID"].astype(str) == str(active_patient["User_ID"])]
        
        if p_hist.empty:
            st.info("No examination history yet.")
        else:
            for _, row in p_hist.sort_values("Timestamp", ascending=False).iterrows():
                with st.expander(f"📅 {row['Timestamp']} - {row['A'][:30]}..."):
                    st.write(f"**S:** {row['S']}")
                    st.write(f"**O:** {row['O']}")
                    st.write(f"**A:** {row['A']}")
                    st.write(f"**P:** {row['P']}")
    else:
        st.info("Scan the patient's QR Code to access medical history.")

# --- TAB 4: EMERGENCY MODE ---
with t_emergency:
    if active_patient is not None and not isinstance(active_patient, pd.Series) or (isinstance(active_patient, pd.Series) and not active_patient.empty):
        st.error(f"⚠️ EMERGENCY MODE: {active_patient['Name']} ({active_patient['User_ID']})")

        birth_date = pd.to_datetime(active_patient.get("Birth_Date"), errors="coerce")
        if pd.notna(birth_date):
            today = datetime.now().date()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            age_display = f"{age} years"
        else:
            age_display = "-"

        def emergency_value(field):
            value = active_patient.get(field, "-")
            return str(value).strip() if pd.notna(value) and str(value).strip() else "-"

        st.subheader("Personal Information")
        st.markdown(
            "<div style='font-size: 16px; font-weight: 600; color: #333; margin: 0.25rem 0 0.75rem 0;'>GelangRMQ</div>",
            unsafe_allow_html=True,
        )
        info_left, info_right = st.columns(2)
        info_fields = {
            "Full Name": emergency_value("Name"),
            "Age": age_display,
            "Gender": emergency_value("Gender"),
            "Blood Type": emergency_value("Blood_Type"),
            "Allergy": emergency_value("Allergies"),
            "Diseases": emergency_value("Chronic_Diseases"),
            "Medication": emergency_value("Current_Medication"),
            "Medical History": emergency_value("Medical_History"),
            "Emergency Contact": emergency_value("Emergency_Contact_Name"),
            "Phone Number": emergency_value("Emergency_Contact_Phone"),
            "BPJS Number": emergency_value("BPJS_ID"),
        }

        with info_left:
            for label in ["Full Name", "Age", "Gender", "Blood Type", "Allergy", "Diseases"]:
                st.markdown(
                    f"<div style='background:#fff; border:1px solid #e1e4e8; border-radius:8px; padding:9px 14px; margin-bottom:9px;'><div class='profile-label'>{label}</div><div style='font-size:1.05em; color:#212529;'>{info_fields[label]}</div></div>",
                    unsafe_allow_html=True,
                )
        with info_right:
            for label in ["Medication", "Medical History", "Emergency Contact", "Phone Number", "BPJS Number"]:
                st.markdown(
                    f"<div style='background:#fff; border:1px solid #e1e4e8; border-radius:8px; padding:9px 14px; margin-bottom:9px;'><div class='profile-label'>{label}</div><div style='font-size:1.05em; color:#212529;'>{info_fields[label]}</div></div>",
                    unsafe_allow_html=True,
                )

        st.subheader("QR Code")
        base_url = os.getenv("BASE_URL")
        emergency_qr_url = f"{base_url}/?uid={active_patient['User_ID']}" if base_url else f"http://localhost:8501/?uid={active_patient['User_ID']}"
        emergency_qr = qrcode.make(emergency_qr_url)
        emergency_qr_buf = BytesIO()
        emergency_qr.save(emergency_qr_buf, format="PNG")
        emergency_qr_base64 = base64.b64encode(emergency_qr_buf.getvalue()).decode("ascii")
        components.html(
            f"""
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; margin: 0; font-size: 10px; }}
                .qr-versions {{ display: flex; justify-content: center; gap: 18px; align-items: flex-start; }}
                .qr-version {{ text-align: center; font-size: 9px; }}
                .qr-frame {{ display: flex; align-items: center; justify-content: center; width: 70px; height: 70px; margin: 0 auto 2px; border: 1px solid #d32f2f; border-radius: 50%; background: white; box-sizing: border-box; }}
                .square-frame {{ display: flex; align-items: center; justify-content: center; width: 70px; height: 70px; margin: 0 auto 2px; border: 1px solid #555; background: white; box-sizing: border-box; }}
                img {{ display: block; width: 60px; height: 60px; margin: 1px; }}
                .square-frame img {{ width: 70px; height: 70px; }}
                p {{ margin: 0 0 2px; font-weight: bold; }}
                button {{ padding: 8px 14px; border: 1px solid #888; border-radius: 2px; background: white; cursor: pointer; }}
                @media print {{ button {{ display: none; }} body.printing-round .square-print {{ display: none; }} body.printing-square .round-print {{ display: none; }} .square-frame {{ width: 1.5cm; height: 1.5cm; }} .square-frame img {{ width: 1.4cm; height: 1.4cm; }} }}
                .print-button {{ margin: 2px; }}
            </style>
            <div class="qr-versions">
                <div class="qr-version round-print"><div class="qr-frame"><img src="data:image/png;base64,{emergency_qr_base64}" alt="Round QR Code {active_patient['User_ID']}"></div><div></div></div>
                <div class="qr-version square-print"><div class="square-frame"><img src="data:image/png;base64,{emergency_qr_base64}" alt="Square QR Code {active_patient['User_ID']}"></div><div></div></div>
            </div>
            <p style="font-size: 6px;">UID: {active_patient['User_ID']}</p>
            <button class="print-button" type="button" onclick="document.body.classList.add('printing-round'); window.print();">🖨️ Print Round QR</button>
            <button class="print-button" type="button" onclick="document.body.classList.add('printing-square'); window.print();">🖨️ Print Square QR</button>
            <script>window.addEventListener('afterprint', function() {{ document.body.classList.remove('printing-round', 'printing-square'); }});</script>
            """,
            height=340,
        )
        
        st.divider()

        col_triage, col_action = st.columns([2, 1])
        
        with col_triage:
            st.subheader("🩺 Quick Assessment (NEWS2 Triage)")
            
            # Show detected vitals badge
            if st.session_state.extracted_vitals:
                st.info(f"✨ AI detected vital signs from chat: {', '.join([f'{k.upper()}: {v}' for k,v in st.session_state.extracted_vitals.items()])}")

            with st.expander("Vital Signs Input", expanded=True):
                v_c1, v_c2 = st.columns(2)

                triage_defaults = get_triage_default_vitals(active_patient, st.session_state.extracted_vitals)

                with v_c1:
                    v_rr = st.number_input("Respiratory Rate (bpm)", 5, 50, int(triage_defaults.get('rr', 20)))
                    v_spo2 = st.number_input("SpO2 (%)", 50, 100, int(triage_defaults.get('spo2', 95)))
                    v_bps = st.number_input("Systolic Blood Pressure", 50, 250, int(triage_defaults.get('bps', 120)))
                with v_c2:
                    v_hr = st.number_input("Heart Rate (bpm)", 20, 200, int(triage_defaults.get('hr', 80)))

                    avpu_options = ["Alert", "Voice", "Pain", "Unresponsive"]
                    default_avpu = triage_defaults.get('avpu', "Alert")
                    if default_avpu not in avpu_options: default_avpu = "Alert"
                    v_avpu = st.selectbox("Consciousness (AVPU)", avpu_options, index=avpu_options.index(default_avpu))
                
                if st.button("PROCESS TRIAGE", use_container_width=True, type="primary"):
                    score, level, color = calculate_news2(v_rr, v_spo2, v_bps, v_hr, v_avpu)
                    st.session_state.last_triage = {"score": score, "level": level, "color": color}

            if "last_triage" in st.session_state:
                lt = st.session_state.last_triage
                st.markdown(f"""
                <div style='background:{lt['color']}; color:white; padding:25px; border-radius:15px; text-align:center; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                    <h1 style='margin:0;'>{lt['level']}</h1>
                    <p style='font-size:1.2em; margin:0;'>NEWS2 Score: {lt['score']}</p>
                </div>
                """, unsafe_allow_html=True)

        with col_action:
            st.subheader("🚨 Quick Actions")
            st.info(f"**Attending Doctor:** {active_patient['Responsible_Doctor'] if active_patient['Responsible_Doctor'] else 'Not assigned'}")
            
            # Action Buttons
            st.link_button("☎️ Call Emergency Contact", f"tel:{active_patient['Emergency_Contact_Phone']}", use_container_width=True)
            
            # Nearest Hospital Geofencing
            u_lat, u_lon = -7.7700, 110.3700 # Mock location
            distances = [{"name": h["name"], "dist": geodesic((u_lat, u_lon), (h["lat"], h["lon"])).km} for h in HOSPITALS]
            nearest = sorted(distances, key=lambda x: x["dist"])[0]
            st.success(f"🏥 **Nearest Hospital:** {nearest['name']} ({nearest['dist']:.2f} KM)")
            
            with st.expander("View Hospitalization History"):
                st.write(active_patient["Hospitalization_History"] if active_patient["Hospitalization_History"] else "No history")

    else:
        st.warning("⚠️ Scan the patient's QR Code to access emergency features.")
        st.info("This feature displays critical information such as blood type, allergies, and important medical history within seconds.")

# --- TAB 5: ADMIN ---
with t_admin:
    pwd = st.text_input("Admin Password", type="password")
    if pwd == MASTER_PASSWORD:
        st.subheader("📊 Patient Master Data")
        df_master = pd.read_csv(DATA_FILE)
        st.dataframe(df_master, use_container_width=True)

        st.subheader("✏️ Manage Patient Data")
        if df_master.empty:
            st.info("There is no patient data to manage.")
        else:
            patient_ids = df_master["User_ID"].fillna("").astype(str).tolist()
            selected_admin_id = st.selectbox("Select Patient User ID", patient_ids)
            selected_index = df_master.index[df_master["User_ID"].astype(str) == selected_admin_id][0]
            selected_admin_patient = df_master.loc[selected_index]

            def admin_value(field):
                value = selected_admin_patient.get(field, "")
                return "" if pd.isna(value) else str(value)

            with st.form("admin_patient_edit_form"):
                edit_left, edit_right = st.columns(2)
                with edit_left:
                    edit_name = st.text_input("Full Name", value=admin_value("Name"))
                    edit_bpjs = st.text_input("BPJS_ID", value=admin_value("BPJS_ID"))
                    edit_birth_date = st.text_input("Date of Birth (YYYY-MM-DD)", value=admin_value("Birth_Date"))
                    edit_gender = st.selectbox(
                        "Jenis Kelamin",
                        ["Male", "Female"],
                        index=0 if admin_value("Gender") != "Female" else 1,
                    )
                    edit_blood_type = st.selectbox(
                        "Gol. Darah",
                        ["A", "B", "AB", "O"],
                        index=["A", "B", "AB", "O"].index(admin_value("Blood_Type")) if admin_value("Blood_Type") in ["A", "B", "AB", "O"] else 0,
                    )
                    edit_weight = st.text_input("Weight (kg)", value=admin_value("Weight_kg"))
                    edit_height = st.text_input("Height (cm)", value=admin_value("Height_cm"))
                    edit_chronic = st.text_area("Chronic Diseases", value=admin_value("Chronic_Diseases"))
                    edit_medication = st.text_area("Current Medication", value=admin_value("Current_Medication"))
                with edit_right:
                    edit_allergies = st.text_area("Allergies", value=admin_value("Allergies"))
                    edit_contact_name = st.text_input("Emergency Contact Name", value=admin_value("Emergency_Contact_Name"))
                    edit_contact_phone = st.text_input("Emergency Contact Phone", value=admin_value("Emergency_Contact_Phone"))
                    edit_blood_pressure = st.text_input("Blood Pressure (mmHg)", value=admin_value("Blood_Pressure"))
                    edit_oxygen = st.text_input("Oxygen Saturation (%)", value=admin_value("Oxygen_Saturation"))
                    edit_hospitalization = st.text_area("Hospitalization History", value=admin_value("Hospitalization_History"))
                    edit_doctor = st.text_input("Attending Doctor", value=admin_value("Responsible_Doctor"))
                    edit_medical_history = st.text_area("Medical History", value=admin_value("Medical_History"))

                if st.form_submit_button("💾 Save Changes", type="primary"):
                    updated_values = {
                        "Name": edit_name,
                        "BPJS_ID": edit_bpjs,
                        "Birth_Date": edit_birth_date,
                        "Gender": edit_gender,
                        "Blood_Type": edit_blood_type,
                        "Weight_kg": edit_weight,
                        "Height_cm": edit_height,
                        "Chronic_Diseases": edit_chronic,
                        "Current_Medication": edit_medication,
                        "Allergies": edit_allergies,
                        "Emergency_Contact_Name": edit_contact_name,
                        "Emergency_Contact_Phone": edit_contact_phone,
                        "Blood_Pressure": edit_blood_pressure,
                        "Oxygen_Saturation": edit_oxygen,
                        "Hospitalization_History": edit_hospitalization,
                        "Responsible_Doctor": edit_doctor,
                        "Medical_History": edit_medical_history,
                        "Last_Update": datetime.now().strftime("%Y-%m-%d"),
                    }
                    numeric_fields = {
                        "Weight_kg": edit_weight,
                        "Height_cm": edit_height,
                        "Oxygen_Saturation": edit_oxygen,
                    }
                    numeric_errors = []
                    for field, raw_value in numeric_fields.items():
                        value = raw_value.strip()
                        if not value:
                            updated_values[field] = pd.NA
                            continue
                        try:
                            parsed_value = float(value)
                            if not parsed_value.is_integer():
                                raise ValueError
                            updated_values[field] = int(parsed_value)
                        except ValueError:
                            numeric_errors.append(field)

                    if not edit_name.strip():
                        st.error("Full name is required.")
                    elif numeric_errors:
                        st.error(f"Values must be whole numbers: {', '.join(numeric_errors)}.")
                    else:
                        for field in numeric_fields:
                            df_master[field] = pd.to_numeric(df_master[field], errors="coerce").astype("Int64")
                        text_fields = [field for field in updated_values if field not in numeric_fields]
                        for field in text_fields:
                            df_master[field] = df_master[field].astype("object")
                        for field, value in updated_values.items():
                            df_master.at[selected_index, field] = value
                        df_master.to_csv(DATA_FILE, index=False)
                        st.success(f"Patient data {selected_admin_id} was updated successfully.")
                        st.rerun()

            st.divider()
            confirm_delete = st.checkbox(f"I confirm that I want to delete patient {selected_admin_id}")
            if st.button("🗑️ Delete Patient", type="secondary", disabled=not confirm_delete):
                df_master = df_master[df_master["User_ID"].astype(str) != selected_admin_id]
                df_master.to_csv(DATA_FILE, index=False)
                st.success(f"Patient data {selected_admin_id} was deleted successfully.")
                st.rerun()
        
        st.subheader("📑 SOAP Transaction Data")
        df_soap = pd.read_csv(ENCOUNTER_FILE)
        st.dataframe(df_soap, use_container_width=True)

        st.divider()
        st.subheader("🔍 RAG Debug: Why Is RAG Inactive?")
        debug_history = st.session_state.get("rag_debug_history", [])
        if debug_history:
            debug_df = pd.DataFrame(debug_history)
            st.dataframe(debug_df, use_container_width=True)
            for idx, item in enumerate(debug_history[:3]):
                st.markdown(f"### Request {idx + 1}: {item.get('knowledge_source', 'Unknown')}")
                st.write(f"- Query length: {item.get('query_length', 0)}")
                st.write(f"- History count: {item.get('history_count', 0)}")
                st.write(f"- Guideline count: {item.get('guideline_count', 0)}")
                st.write(f"- Patient selected: {item.get('patient_selected', False)}")
                for reason in item.get("reasons", []):
                    st.write(f"- {reason}")
                st.divider()
        else:
            st.info("No chat request has been processed. RAG debug information will appear here after a new chat.")

        st.divider()
        st.subheader("�🔬 Model Comparison Results")
        if os.path.exists(COMPARISON_LOG_FILE):
            normalize_comparison_log_file(COMPARISON_LOG_FILE)
            df_comp = pd.read_csv(COMPARISON_LOG_FILE)

            st.markdown("#### Filter Analisis")
            analysis_scope = st.radio(
                "Mode Analisis",
                ["Overall", "Per Patient"],
                horizontal=True
            )

            df_comp_view = df_comp.copy()
            if analysis_scope == "Per Patient":
                if "User_ID" not in df_comp.columns:
                    st.warning("The User_ID column is not available in the legacy log. Run a new chat to enable per-patient metrics.")
                    df_comp_view = df_comp.iloc[0:0]
                else:
                    patient_ids = sorted([
                        uid for uid in df_comp["User_ID"].fillna("").astype(str).str.strip().unique().tolist() if uid
                    ])
                    if not patient_ids:
                        st.info("No User_ID is available in the comparison log. Run a new patient chat to populate it.")
                        df_comp_view = df_comp.iloc[0:0]
                    else:
                        selected_patient_id = st.selectbox("Select Patient ID", patient_ids)
                        df_comp_view = df_comp[
                            df_comp["User_ID"].fillna("").astype(str).str.strip() == selected_patient_id
                        ]

            st.dataframe(df_comp_view, use_container_width=True)

            st.markdown("#### Quick Stats")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Total Test Cases", len(df_comp_view))
            with col_s2:
                avg_latency = df_comp_view["Latency_sec"].mean() if not df_comp_view.empty else 0
                st.metric("Avg Latency", f"{avg_latency:.2f}s")
            with col_s3:
                model_counts = df_comp_view["Model_Name"].value_counts().to_dict() if not df_comp_view.empty else {}
                st.write("Model Usage:", model_counts)

            st.markdown("#### Information Usage Percentage")
            if "Knowledge_Source" in df_comp_view.columns and not df_comp_view.empty:
                source_counts = df_comp_view["Knowledge_Source"].value_counts(normalize=True).mul(100)
                general_pct = float(source_counts.get("General", 0.0))
                rag_pct = float(source_counts.get("RAG", 0.0))

                col_g1, col_g2, col_g3 = st.columns(3)
                with col_g1:
                    st.metric("General Information", f"{general_pct:.1f}%")
                with col_g2:
                    st.metric("RAG Information", f"{rag_pct:.1f}%")
                with col_g3:
                    st.metric("Total RAG Events", int((df_comp_view["RAG_Used"] == 1).sum())) if "RAG_Used" in df_comp_view.columns else st.metric("Total RAG Events", 0)

                sub_cols = st.columns(3)
                with sub_cols[0]:
                    history_pct = float((df_comp_view["RAG_History_Used"].fillna(0) == 1).mean() * 100) if "RAG_History_Used" in df_comp_view.columns else 0
                    st.metric("RAG History", f"{history_pct:.1f}%")
                with sub_cols[1]:
                    guidelines_pct = float((df_comp_view["RAG_Guidelines_Used"].fillna(0) == 1).mean() * 100) if "RAG_Guidelines_Used" in df_comp_view.columns else 0
                    st.metric("RAG Guidelines", f"{guidelines_pct:.1f}%")
                with sub_cols[2]:
                    both_pct = float(((df_comp_view["RAG_History_Used"].fillna(0) == 1) & (df_comp_view["RAG_Guidelines_Used"].fillna(0) == 1)).mean() * 100) if "RAG_History_Used" in df_comp_view.columns and "RAG_Guidelines_Used" in df_comp_view.columns else 0
                    st.metric("RAG Both", f"{both_pct:.1f}%")
            else:
                st.info("No matching data is available for this filter, or knowledge-source metadata is missing.")

            st.download_button(
                label="📥 Download Dataset (Filtered)",
                data=df_comp_view.to_csv(index=False).encode('utf-8'),
                file_name="model_comparison_dataset_filtered.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No model comparison data has been saved yet.")

        st.divider()
        st.subheader("📚 Knowledge Base (RAG)")
        if os.path.exists(GUIDELINES_FILE):
            with open(GUIDELINES_FILE, "r", encoding="utf-8") as f:
                guideline_content = f.read()
            
            new_guidelines = st.text_area(
                "Update Clinical Guidelines (clinical_guidelines.txt)",
                value=guideline_content,
                height=400,
                help="Gunakan format '=== PANDUAN KLINIS: Nama Judul ===' untuk setiap bagian baru agar sistem RAG dapat memecah konten dengan benar."
            )
            
            c_kb1, c_kb2 = st.columns(2)
            with c_kb1:
                if st.button("💾 Save Knowledge Base Changes", use_container_width=True):
                    with open(GUIDELINES_FILE, "w", encoding="utf-8") as f:
                        f.write(new_guidelines)
                    # Clear cache so RAG re-embeds the new content
                    if "guidelines_cache" in st.session_state:
                        del st.session_state.guidelines_cache
                    st.success("Knowledge base updated successfully and RAG cache cleared!")
            with c_kb2:
                # Add refresh button to force re-indexing
                if st.button("🔄 Refresh RAG Cache", use_container_width=True):
                    if "guidelines_cache" in st.session_state:
                        del st.session_state.guidelines_cache
                        st.info("RAG cache cleared. AI will reprocess the file during the next chat.")
        else:
            st.error(f"File {GUIDELINES_FILE} was not found!")

    elif pwd: st.error("Incorrect password")

    st.divider()
    st.subheader("🛠️ System Debug Logs")
    if st.button("Refresh Logs"):
        if os.path.exists("app_debug.log"):
            with open("app_debug.log", "r") as f:
                logs = f.readlines()
                # Show last 50 lines
                st.code("".join(logs[-50:]))
        else:
            st.info("No log file is available yet.")

import streamlit as st
import pandas as pd
import google.generativeai as genai
import qrcode
import os
import json
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
    "Timestamp", "Model_Name", "Input_Narrative", 
    "S_Result", "O_Result", "A_Result", "P_Result", 
    "RR", "SpO2", "BPS", "HR", "AVPU", "Latency_sec"
]

if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=DATA_STRUCTURE).to_csv(DATA_FILE, index=False)

if not os.path.exists(ENCOUNTER_FILE):
    pd.DataFrame(columns=ENCOUNTER_STRUCTURE).to_csv(ENCOUNTER_FILE, index=False)

if not os.path.exists(COMPARISON_LOG_FILE):
    pd.DataFrame(columns=COMPARISON_STRUCTURE).to_csv(COMPARISON_LOG_FILE, index=False)

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
        final_result = {
            "soap": {"S": "-", "O": "-", "A": "-", "P": "-"}, 
            "profile_updates": {},
            "triage_vitals": {},
            "retrieved_history": retrieved_history,
            "retrieved_guidelines": retrieved_guidelines
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
        try:
            log_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                "Latency_sec": round(latency, 2)
            }
            pd.DataFrame([log_data]).to_csv(COMPARISON_LOG_FILE, mode='a', header=False, index=False)
            logger.info(f"Comparison log entry added for model {model_name}")
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
    
    st.markdown("### 🤖 Konfigurasi AI")
    selected_model = st.selectbox(
        "Pilih Model (untuk Paper/Test)",
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
                st.success(f"🔓 Sesi Aktif: {active_patient['Name']}")
                st.info(f"ID: {active_patient['User_ID']}")
            else:
                st.error(f"❌ Pasien dengan ID {scanned_uid} tidak ditemukan.")
        except Exception as e:
            st.error(f"Error accessing QR data: {e}")
    
    if active_patient is None or (isinstance(active_patient, pd.Series) and active_patient.empty):
        st.warning("📥 Menunggu Scan QR Code...")
        st.info("Sistem ini didesain untuk akses cepat via QR Code. Silakan scan QR pasien untuk memulai.")
    
    st.divider()
    
    # Hide manual selector and show current status
    st.markdown("### 🛠️ Status Perangkat")
    st.caption("Manual Selection: **DISABLED** (QR-Only Mode)")

# ==========================================
# 4. MAIN INTERFACE (TABS)
# ==========================================
# Streamlit standard tabs don't support programmatic switching yet.
# To achieve the requirement, we reorder the tabs if UID is present.

tab_list = [
    "📝 Registrasi", 
    "💬 AI Chat SOAP", 
    "🕒 Riwayat Medis", 
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
    st.subheader("Pendaftaran Pasien Baru")
    
    # Auto-generate User ID
    next_id = 1
    if not df_p.empty:
        try:
            # Try to get the max numeric part if format is PID-001
            ids = df_p["User_ID"].str.extract('(\d+)').astype(float).dropna()
            if not ids.empty:
                next_id = int(ids.max().iloc[0]) + 1
        except:
            next_id = len(df_p) + 1
    
    auto_id = f"MAI-{datetime.now().year}-{next_id:04d}"

    with st.form("reg_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        d = {}
        with c1:
            d["User_ID"] = st.text_input("User ID", value=auto_id, help="ID ini dibuat otomatis oleh sistem", disabled=True)
            d["BPJS_ID"] = st.text_input("BPJS_ID")
            d["Name"] = st.text_input("Nama Lengkap")
            d["Birth_Date"] = st.date_input(
                "Tanggal Lahir",
                value=datetime(1990, 1, 1).date(),
                min_value=datetime(1900, 1, 1).date(),
                max_value=datetime.now().date()
            )
            d["Gender"] = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        with c2:
            d["Blood_Type"] = st.selectbox("Gol. Darah", ["A", "B", "AB", "O"])
            d["Weight_kg"] = st.number_input("Berat (kg)", 0, 200)
            d["Height_cm"] = st.number_input("Tinggi (cm)", 0, 250)
            d["Blood_Pressure"] = st.text_input("Tekanan Darah (mmHg)", placeholder="120/80")
            d["Oxygen_Saturation"] = st.number_input("Kadar Oksigen (%)", 0, 100, 98)
            d["Hospitalization_History"] = st.text_area("Riwayat Rawat Inap")
            d["Responsible_Doctor"] = st.text_input("Dokter Penanggung Jawab")
            d["Allergies"] = st.text_area("Alergi")
            d["Emergency_Contact_Phone"] = st.text_input("Kontak Darurat (Telp)")
        
        if st.form_submit_button("Simpan & Generate QR"):
            # Use the auto-generated ID directly as the field is disabled in form
            d["User_ID"] = auto_id
            if d["User_ID"] and d["Name"]:
                d["Last_Update"] = datetime.now().strftime("%Y-%m-%d")
                df_p = pd.concat([df_p, pd.DataFrame([d])], ignore_index=True)
                df_p.to_csv(DATA_FILE, index=False)
                st.success("Pasien terdaftar!")
                
                # Link QR Code dinamis (Otomatis deteksi URL jika di Streamlit Cloud)
                # Jika dijalankan lokal tetap localhost
                qr_url = f"http://localhost:8501/?uid={d['User_ID']}"
                
                img = qrcode.make(qr_url)
                buf = BytesIO(); img.save(buf, format="PNG")
                st.image(buf.getvalue(), caption=f"UID: {d['User_ID']}")
            else: st.error("ID dan Nama wajib diisi!")

# --- TAB 2: AI CHAT SOAP ---
with t_chat:
    if active_patient is None or (isinstance(active_patient, pd.Series) and active_patient.empty):
        st.warning("Silakan pilih pasien di sidebar terlebih dahulu.")
    else:
        st.info("💡 **Untuk Pasien:** Sampaikan keluhan Anda secara naratif (misal: 'Saya pusing sejak kemarin dan rasa mual'). AI akan membantu merangkumnya untuk dokter.")
        c_l, c_r = st.columns([1, 1])
        with c_l:
            st.subheader(f"💬 Pasien: {active_patient['Name']}")
            # Clear button
            if st.button("Hapus Chat"): st.session_state.messages = []; st.rerun()
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
            if prompt := st.chat_input("Halo, ceritakan keluhan Anda di sini..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                
                with st.spinner("AI sedang menganalisis keluhan Anda..."):
                    ai_response = process_narrative(prompt, api_key_env, active_patient)
                    if ai_response:
                        st.session_state.soap_record = ai_response.get("soap", {})
                        st.session_state.profile_updates = ai_response.get("profile_updates", {})
                        st.session_state.retrieved_history = ai_response.get("retrieved_history", [])
                        st.session_state.retrieved_guidelines = ai_response.get("retrieved_guidelines", [])
                        # Update extracted vitals for Triage
                        new_vitals = ai_response.get("triage_vitals", {})
                        if new_vitals:
                            # Only update if not null
                            for k, v in new_vitals.items():
                                if v is not None:
                                    st.session_state.extracted_vitals[k] = v
                        
                        st.session_state.messages.append({"role": "assistant", "content": "Terima kasih. Saya telah merangkum keluhan Anda dalam format medis. Silakan cek preview SOAP dan bagian Triage."})
                        st.rerun()
        
        with c_r:
            st.subheader("📋 Preview SOAP")
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
                        st.markdown("**Riwayat Medis Pasien Terkait (CSV RAG):**")
                        for idx, hist in enumerate(st.session_state.retrieved_history):
                            st.caption(f"**{idx+1}. Tanggal: {hist['timestamp']}** (Diagnosis: *{hist['A']}*)")
                            st.write(f"- **S**: {hist['S']}")
                            st.write(f"- **P**: {hist['P']}")
                    
                    if st.session_state.get("retrieved_guidelines"):
                        st.markdown("**Panduan Klinis Medis Terkait (External Doc RAG):**")
                        for idx, guide in enumerate(st.session_state.retrieved_guidelines):
                            st.markdown(f"📖 **{guide['title']}**")
                            st.text(guide['content'])
            
            if st.button("💾 Simpan Rekam Medis (SOAP)", use_container_width=True, type="primary"):
                new_enc = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "User_ID": str(active_patient["User_ID"]),
                    **st.session_state.soap_record
                }
                df_e = pd.read_csv(ENCOUNTER_FILE)
                df_e = pd.concat([df_e, pd.DataFrame([new_enc])], ignore_index=True)
                df_e.to_csv(ENCOUNTER_FILE, index=False)
                st.success("Rekam medis tersimpan!")

            st.divider()
            st.subheader("🧬 Preview Data Medis (Profile)")
            
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
                    <div class="profile-label">Riwayat Penyakit</div>
                    <div class="profile-value">{get_display_val('Chronic_Diseases', active_patient['Chronic_Diseases'])}</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Obat Rutin</div>
                    <div class="profile-value">{get_display_val('Current_Medication', active_patient['Current_Medication'])}</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Alergi</div>
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
                    <div class="profile-label">Riwayat Rawat Inap</div>
                    <div class="profile-value">{get_display_val('Hospitalization_History', active_patient['Hospitalization_History'])}</div>
                </div>
                <div class="profile-row">
                    <div class="profile-label">Dokter PJ</div>
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
                if st.button("🆙 Perbarui Profil Pasien (Smart Merge)", use_container_width=True):
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
                        st.success("Profil pasien berhasil diperbarui dengan Smart Merge!")
                        st.session_state.profile_updates = {}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal memperbarui profil: {e}")

# --- TAB 3: MEDICAL HISTORY ---
with t_hist:
    if active_patient is not None and not isinstance(active_patient, pd.Series) or (isinstance(active_patient, pd.Series) and not active_patient.empty):
        st.subheader(f"🕒 Riwayat Medis: {active_patient['Name']}")
        df_e = pd.read_csv(ENCOUNTER_FILE)
        p_hist = df_e[df_e["User_ID"].astype(str) == str(active_patient["User_ID"])]
        
        if p_hist.empty:
            st.info("Belum ada riwayat pemeriksaan.")
        else:
            for _, row in p_hist.sort_values("Timestamp", ascending=False).iterrows():
                with st.expander(f"📅 {row['Timestamp']} - {row['A'][:30]}..."):
                    st.write(f"**S:** {row['S']}")
                    st.write(f"**O:** {row['O']}")
                    st.write(f"**A:** {row['A']}")
                    st.write(f"**P:** {row['P']}")
    else:
        st.info("Pilih pasien untuk melihat riwayat.")

# --- TAB 4: EMERGENCY MODE ---
with t_emergency:
    if active_patient is not None and not isinstance(active_patient, pd.Series) or (isinstance(active_patient, pd.Series) and not active_patient.empty):
        st.error(f"⚠️ MODE DARURAT: {active_patient['Name']} ({active_patient['User_ID']})")
        
        # Critical Info Top Bar
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card' style='border-top: 5px solid #d32f2f;'><div class='profile-label'>Gol. Darah</div><h2 style='margin:0;'>{active_patient['Blood_Type']}</h2></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card' style='border-top: 5px solid #d32f2f;'><div class='profile-label'>Alergi</div><div style='font-size:0.9em;'>{active_patient['Allergies'] if active_patient['Allergies'] else '-'}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card' style='border-top: 5px solid #d32f2f;'><div class='profile-label'>Penyakit Kronis</div><div style='font-size:0.9em;'>{active_patient['Chronic_Diseases'] if active_patient['Chronic_Diseases'] else '-'}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-card' style='border-top: 5px solid #ffc107;'><div class='profile-label'>Kontak Darurat</div><h3 style='margin:0;'>{active_patient['Emergency_Contact_Phone']}</h3></div>", unsafe_allow_html=True)

        st.divider()

        col_triage, col_action = st.columns([2, 1])
        
        with col_triage:
            st.subheader("🩺 Penilaian Cepat (NEWS2 Triage)")
            
            # Show detected vitals badge
            if st.session_state.extracted_vitals:
                st.info(f"✨ AI mendeteksi tanda vital dari chat: {', '.join([f'{k.upper()}: {v}' for k,v in st.session_state.extracted_vitals.items()])}")

            with st.expander("Input Tanda-Tanda Viral", expanded=True):
                v_c1, v_c2 = st.columns(2)
                
                # Fetch values from session state if extracted by AI
                ev = st.session_state.extracted_vitals
                
                with v_c1:
                    v_rr = st.number_input("Respirasi (bpm)", 5, 50, int(ev.get('rr', 20)))
                    v_spo2 = st.number_input("SpO2 (%)", 50, 100, int(ev.get('spo2', 95)))
                    v_bps = st.number_input("Tekanan Darah Sistolik", 50, 250, int(ev.get('bps', 120)))
                with v_c2:
                    v_hr = st.number_input("Denyut Jantung (bpm)", 20, 200, int(ev.get('hr', 80)))
                    
                    avpu_options = ["Alert", "Voice", "Pain", "Unresponsive"]
                    default_avpu = ev.get('avpu', "Alert")
                    if default_avpu not in avpu_options: default_avpu = "Alert"
                    v_avpu = st.selectbox("Kesadaran (AVPU)", avpu_options, index=avpu_options.index(default_avpu))
                
                if st.button("PROSES TRIAGE", use_container_width=True, type="primary"):
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
            st.subheader("🚨 Tindakan Cepat")
            st.info(f"**Dokter PJ:** {active_patient['Responsible_Doctor'] if active_patient['Responsible_Doctor'] else 'Belum ditentukan'}")
            
            # Action Buttons
            st.link_button("☎️ Hubungi Kontak Darurat", f"tel:{active_patient['Emergency_Contact_Phone']}", use_container_width=True)
            
            # Nearest Hospital Geofencing
            u_lat, u_lon = -7.7700, 110.3700 # Mock location
            distances = [{"name": h["name"], "dist": geodesic((u_lat, u_lon), (h["lat"], h["lon"])).km} for h in HOSPITALS]
            nearest = sorted(distances, key=lambda x: x["dist"])[0]
            st.success(f"🏥 **RS Terdekat:** {nearest['name']} ({nearest['dist']:.2f} KM)")
            
            with st.expander("Lihat Riwayat Inap"):
                st.write(active_patient["Hospitalization_History"] if active_patient["Hospitalization_History"] else "Tidak ada riwayat")

    else:
        st.warning("⚠️ Silakan scan QR Code pasien atau pilih pasien di sidebar untuk mengakses fitur darurat.")
        st.info("Fitur ini akan menampilkan informasi kritis seperti Golongan Darah, Alergi, dan riwayat medis penting dalam hitungan detik.")

# --- TAB 5: ADMIN ---
with t_admin:
    pwd = st.text_input("Password Admin", type="password")
    if pwd == MASTER_PASSWORD:
        st.subheader("📊 Data Master Pasien")
        df_master = pd.read_csv(DATA_FILE)
        st.dataframe(df_master, use_container_width=True)
        
        st.subheader("📑 Data Transaksi SOAP")
        df_soap = pd.read_csv(ENCOUNTER_FILE)
        st.dataframe(df_soap, use_container_width=True)

        st.divider()
        st.subheader("🔬 Model Comparison Results (for Paper)")
        if os.path.exists(COMPARISON_LOG_FILE):
            df_comp = pd.read_csv(COMPARISON_LOG_FILE)
            st.dataframe(df_comp, use_container_width=True)
            
            # Simple stats for the paper
            st.markdown("#### Quick Stats")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Total Test Cases", len(df_comp))
            with col_s2:
                avg_latency = df_comp["Latency_sec"].mean() if not df_comp.empty else 0
                st.metric("Avg Latency", f"{avg_latency:.2f}s")
            with col_s3:
                model_counts = df_comp["Model_Name"].value_counts().to_dict()
                st.write("Model Usage:", model_counts)

            st.download_button(
                label="📥 Download Dataset for Analysis",
                data=df_comp.to_csv(index=False).encode('utf-8'),
                file_name="model_comparison_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Belum ada data perbandingan yang tersimpan.")

    elif pwd: st.error("Salah password")

    st.divider()
    st.subheader("🛠️ System Debug Logs")
    if st.button("Refresh Logs"):
        if os.path.exists("app_debug.log"):
            with open("app_debug.log", "r") as f:
                logs = f.readlines()
                # Show last 50 lines
                st.code("".join(logs[-50:]))
        else:
            st.info("Belum ada file log.")

import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import logging
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# 0. LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    filename='app_debug.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="MedisAI Chat: Smart Medical Recorder",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .soap-container {
        background-color: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #007bff;
    }
    .soap-header {
        color: #007bff;
        font-weight: bold;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .soap-section {
        margin-bottom: 15px;
    }
    .soap-label {
        font-weight: bold;
        color: #495057;
        font-size: 0.9em;
        text-transform: uppercase;
    }
    .soap-content {
        color: #212529;
        font-size: 1.1em;
        line-height: 1.5;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7d32,#1b5e20);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "soap_record" not in st.session_state:
    st.session_state.soap_record = {
        "S": "Belum ada data",
        "O": "Belum ada data",
        "A": "Belum ada data",
        "P": "Belum ada data"
    }

# ==========================================
# 3. SIDEBAR: Settings & Patient Info
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=100)
    st.title("MedisAI Pro")
    st.subheader("Konversi Narasi ke SOAP")
    
    st.divider()
    
    # API status check
    api_key_env = os.getenv("GOOGLE_API_KEY", "")
    model_name_env = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")
    if api_key_env:
        st.success(f"✅ AI Connected ({model_name_env})")
    else:
        st.error("❌ AI Key Missing (.env)")
    
    st.divider()
    
    # Load Patient Data
    patient_data_obj = None
    try:
        df_patients = pd.read_csv("medical_registry_v3.csv")
        patient_list = df_patients["Name"].tolist()
        selected_patient = st.selectbox("Pilih Pasien", ["Umum / Pasien Baru"] + patient_list)
        if selected_patient != "Umum / Pasien Baru":
            patient_data_obj = df_patients[df_patients["Name"] == selected_patient].iloc[0].to_dict()
    except:
        selected_patient = "Umum / Pasien Baru"
        st.warning("Database pasien tidak ditemukan.")

    st.divider()
    st.info("💡 **Tips:** Ceritakan kondisi pasien secara naratif, MedisAI akan menyusun format SOAP secara otomatis.")

# ==========================================
# 4. LLM LOGIC
# ==========================================
def process_narrative(narrative, patient_data=None):
    if not api_key_env:
        st.error("Silakan masukkan API Key Gemini di file .env.")
        logger.error("API Key missing in environment variables.")
        return None

    try:
        logger.info(f"--- New Processing Request ---")
        logger.info(f"Using Model: {model_name_env}")
        logger.info(f"Input Narrative: {narrative[:200]}...") # Log partial narrative for privacy/brevity
        
        genai.configure(api_key=api_key_env)
        model = genai.GenerativeModel(model_name_env)
        
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
            - Riwayat Medis: {patient_data.get('Medical_History')}
            """

        prompt = f"""
        Anda adalah asisten medis profesional. Konversi narasi medis berikut menjadi format JSON SOAP (Subjective, Objective, Assessment, Plan).
        Gunakan data profil medis pasien jika relevan untuk membantu analisis.
        
        {patient_context}
        
        NARASI DOKTER SAAT INI: 
        "{narrative}"
        
        WAJIB OUTPUT DALAM FORMAT JSON BERIKUT:
        {{
            "S": "Isi bagian Subjective",
            "O": "Isi bagian Objective",
            "A": "Isi bagian Assessment",
            "P": "Isi bagian Plan"
        }}
        
        Hanya berikan JSON saja. Jangan ada teks lain atau struktur lain.
        """
        
        response = model.generate_content(prompt)
        
        # Log raw response for debugging
        logger.info(f"AI Raw Response: {response.text}")
        
        # Clean response text from markdown code blocks if present
        text = response.text.strip().replace('```json', '').replace('```', '')
        
        import json
        result = json.loads(text)
        
        # Normalization: handle nested "SOAP" key
        if "SOAP" in result and isinstance(result["SOAP"], dict):
            result = result["SOAP"]
            
        # Normalization: map long names to shorthand keys
        mapping = {"Subjective": "S", "Objective": "O", "Assessment": "A", "Plan": "P"}
        for long, short in mapping.items():
            if long in result and short not in result:
                result[short] = str(result[long])

        logger.info("Successfully parsed and normalized JSON response.")
        return result
        
    except Exception as e:
        error_msg = f"Error AI Processing: {str(e)}"
        st.error(error_msg)
        
        # Log the full traceback for debugging
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
col_chat, col_soap = st.columns([1, 1])

with col_chat:
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader("💬 Chat Narasi Dokter")
    with header_col2:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Contoh: Pasien mengeluh pusing sejak 2 hari lalu, TD 120/80, diagnosis Migrain..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with AI
        with st.spinner("Menganalisis narasi..."):
            soap_data = process_narrative(prompt, patient_data_obj)
            if soap_data:
                st.session_state.soap_record = soap_data
                response_text = "Data SOAP telah diperbarui berdasarkan narasi Anda."
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.rerun()

with col_soap:
    st.subheader("📋 Rekam Medis (SOAP)")
    
    st.markdown(f"""
    <div class="soap-container">
        <div class="soap-header">
            PASIEN: {selected_patient}<br>
            <small>Tanggal: {datetime.now().strftime('%d %B %Y')}</small>
        </div>
        
        <div class="soap-section">
            <div class="soap-label">Subjective (S)</div>
            <div class="soap-content">{st.session_state.soap_record.get('S', '-')}</div>
        </div>
        
        <div class="soap-section">
            <div class="soap-label">Objective (O)</div>
            <div class="soap-content">{st.session_state.soap_record.get('O', '-')}</div>
        </div>
        
        <div class="soap-section">
            <div class="soap-label">Assessment (A)</div>
            <div class="soap-content">{st.session_state.soap_record.get('A', '-')}</div>
        </div>
        
        <div class="soap-section">
            <div class="soap-label">Plan (P)</div>
            <div class="soap-content">{st.session_state.soap_record.get('P', '-')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    if st.button("💾 Simpan ke Database", use_container_width=True):
        try:
            record_file = "medical_encounters.csv"
            new_record = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Patient_Name": selected_patient,
                "S": st.session_state.soap_record.get('S', ''),
                "O": st.session_state.soap_record.get('O', ''),
                "A": st.session_state.soap_record.get('A', ''),
                "P": st.session_state.soap_record.get('P', '')
            }
            
            if os.path.exists(record_file):
                df_rec = pd.read_csv(record_file)
                df_rec = pd.concat([df_rec, pd.DataFrame([new_record])], ignore_index=True)
            else:
                df_rec = pd.DataFrame([new_record])
            
            df_rec.to_csv(record_file, index=False)
            st.success(f"Rekam medis untuk {selected_patient} berhasil disimpan!")
        except Exception as e:
            st.error(f"Gagal menyimpan: {str(e)}")
    
    if st.button("🗑️ Reset SOAP", use_container_width=True, type="secondary"):
        st.session_state.soap_record = {"S": "-", "O": "-", "A": "-", "P": "-"}
        st.rerun()

    if st.button("🖨️ Cetak PDF", use_container_width=True):
        st.info("Fitur cetak sedang dikembangkan.")

# ==========================================
# 6. FOOTER
# ==========================================
st.markdown("---")
st.markdown("<center><small>MedisAI Pro v1.0 | NLP-driven Medical Assistant</small></center>", unsafe_allow_html=True)

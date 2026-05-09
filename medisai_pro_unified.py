import streamlit as st
import pandas as pd
import google.generativeai as genai
import qrcode
import os
import json
from io import BytesIO
from datetime import datetime
from geopy.distance import geodesic
import logging
import traceback
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
MASTER_PASSWORD = "access123"

# Ensure Data Structures
DATA_STRUCTURE = [
    "User_ID", "BPJS_ID", "Name", "Birth_Date", "Gender",
    "Blood_Type", "Weight_kg", "Height_cm", "Chronic_Diseases",
    "Current_Medication", "Allergies", "Emergency_Contact_Name",
    "Emergency_Contact_Phone", "Medical_History", "Last_Update"
]

ENCOUNTER_STRUCTURE = ["Timestamp", "User_ID", "S", "O", "A", "P"]

if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=DATA_STRUCTURE).to_csv(DATA_FILE, index=False)

if not os.path.exists(ENCOUNTER_FILE):
    pd.DataFrame(columns=ENCOUNTER_STRUCTURE).to_csv(ENCOUNTER_FILE, index=False)

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

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

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
    try:
        logger.info(f"--- Unified App: Processing Request ---")
        logger.info(f"Using Model: {os.getenv('GOOGLE_MODEL_NAME', 'gemini-1.5-flash')}")
        logger.info(f"Input Narrative: {narrative[:200]}...")
        
        genai.configure(api_key=api_key)
        model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        
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
        Anda adalah asisten medis profesional. Analisis narasi medis berikut.
        1. Konversi menjadi format SOAP (Subjective, Objective, Assessment, Plan).
        2. Deteksi apakah ada informasi baru yang dapat memperbarui profil medis pasien (seperti alergi baru, riwayat penyakit kronis baru, atau obat yang sedang dikonsumsi).
        
        DATA PROFIL MEDIS SAAT INI:
        {patient_context}
        
        NARASI DOKTER SAAT INI: 
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
                "Chronic_Diseases": "isi jika ada info baru, jika tidak ada biarkan sama dengan data saat ini",
                "Current_Medication": "isi jika ada info baru",
                "Allergies": "isi jika ada info baru",
                "Weight_kg": "isi jika ada info baru (angka)",
                "Height_cm": "isi jika ada info baru (angka)",
                "Medical_History": "tambahkan riwayat baru jika ada"
            }}
        }}
        
        Hanya berikan JSON saja.
        """
        
        response = model.generate_content(prompt)
        logger.info(f"AI Raw Response: {response.text}")
        
        text = response.text.strip().replace('```json', '').replace('```', '')
        result = json.loads(text)
        
        # Normalization
        final_result = {"soap": {"S": "-", "O": "-", "A": "-", "P": "-"}, "profile_updates": {}}
        
        if "soap" in result:
            final_result["soap"] = result["soap"]
        elif "S" in result: # Handle flat structure if AI fails
            final_result["soap"] = {k: result.get(k, "-") for k in ["S", "O", "A", "P"]}
            
        if "profile_updates" in result:
            final_result["profile_updates"] = result["profile_updates"]
            
        logger.info("Successfully parsed and normalized JSON response.")
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
    if api_key_env:
        st.success(f"✅ AI Connected ({model_name_env})")
    else:
        st.error("❌ AI Key Missing (.env)")
    
    st.divider()
    
    # Global Patient Selector
    df_p = pd.read_csv(DATA_FILE)
    scanned_uid = st.query_params.get("uid")
    
    default_idx = 0
    if scanned_uid:
        try:
            matched_idx = df_p[df_p["User_ID"].astype(str) == str(scanned_uid)].index[0]
            default_idx = int(matched_idx) + 1
        except: pass

    patient_options = ["Pilih Pasien..."] + df_p["Name"].tolist()
    selected_name = st.selectbox("Pasien Aktif", patient_options, index=default_idx)
    
    active_patient = None
    if selected_name != "Pilih Pasien...":
        active_patient = df_p[df_p["Name"] == selected_name].iloc[0]
        st.success(f"Aktif: {active_patient['Name']} ({active_patient['User_ID']})")

# ==========================================
# 4. MAIN INTERFACE (TABS)
# ==========================================
t_reg, t_chat, t_hist, t_emergency, t_admin = st.tabs([
    "📝 Registrasi", 
    "💬 AI Chat SOAP", 
    "🕒 Riwayat Medis", 
    "🚑 Emergency (CDSS)", 
    "🔐 Admin"
])

# --- TAB 1: REGISTRATION ---
with t_reg:
    st.subheader("Pendaftaran Pasien Baru")
    with st.form("reg_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        d = {}
        with c1:
            d["User_ID"] = st.text_input("User ID")
            d["BPJS_ID"] = st.text_input("BPJS_ID")
            d["Name"] = st.text_input("Nama Lengkap")
            d["Birth_Date"] = st.date_input("Tanggal Lahir")
            d["Gender"] = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        with c2:
            d["Blood_Type"] = st.selectbox("Gol. Darah", ["A", "B", "AB", "O"])
            d["Weight_kg"] = st.number_input("Berat (kg)", 0, 200)
            d["Height_cm"] = st.number_input("Tinggi (cm)", 0, 250)
            d["Allergies"] = st.text_area("Alergi")
            d["Emergency_Contact_Phone"] = st.text_input("Kontak Darurat (Telp)")
        
        if st.form_submit_button("Simpan & Generate QR"):
            if d["User_ID"] and d["Name"]:
                d["Last_Update"] = datetime.now().strftime("%Y-%m-%d")
                df_p = pd.concat([df_p, pd.DataFrame([d])], ignore_index=True)
                df_p.to_csv(DATA_FILE, index=False)
                st.success("Pasien terdaftar!")
                qr_url = f"http://localhost:8501/?uid={d['User_ID']}"
                img = qrcode.make(qr_url)
                buf = BytesIO(); img.save(buf, format="PNG")
                st.image(buf.getvalue(), caption=f"UID: {d['User_ID']}")
            else: st.error("ID dan Nama wajib diisi!")

# --- TAB 2: AI CHAT SOAP ---
with t_chat:
    if not active_patient is not None:
        st.warning("Silakan pilih pasien di sidebar terlebih dahulu.")
    else:
        c_l, c_r = st.columns([1, 1])
        with c_l:
            st.subheader(f"💬 Nama Pasien: {active_patient['Name']}")
            # Clear button
            if st.button("Hapus Chat"): st.session_state.messages = []; st.rerun()
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
            if prompt := st.chat_input("Tulis narasi pemeriksaan..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                
                with st.spinner("AI sedang merangkum..."):
                    ai_response = process_narrative(prompt, api_key_env, active_patient)
                    if ai_response:
                        st.session_state.soap_record = ai_response.get("soap", {})
                        st.session_state.profile_updates = ai_response.get("profile_updates", {})
                        st.session_state.messages.append({"role": "assistant", "content": "Analisis selesai. SOAP dan Profil Medis telah diperbarui."})
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
                        # Find the row to update
                        idx = df_all[df_all["User_ID"].astype(str) == str(active_patient["User_ID"])].index[0]
                        
                        # Smart Merge Logic
                        append_fields = ["Chronic_Diseases", "Current_Medication", "Allergies", "Medical_History"]
                        replace_fields = ["Weight_kg", "Height_cm"]
                        
                        for k, v in updates.items():
                            if k in df_all.columns and v:
                                if k in append_fields:
                                    current_val = str(df_all.at[idx, k]) if pd.notna(df_all.at[idx, k]) else ""
                                    new_val = str(v)
                                    
                                    # Merge and remove duplicates
                                    existing_items = [i.strip().lower() for i in current_val.split(",") if i.strip()]
                                    new_items = [i.strip() for i in new_val.split(",") if i.strip()]
                                    
                                    merged_list = current_val.split(",") if current_val else []
                                    for item in new_items:
                                        if item.lower().strip() not in existing_items:
                                            merged_list.append(item.strip())
                                    
                                    df_all.at[idx, k] = ", ".join([i.strip() for i in merged_list if i.strip()])
                                elif k in replace_fields:
                                    df_all.at[idx, k] = v
                        
                        df_all.at[idx, "Last_Update"] = datetime.now().strftime("%Y-%m-%d")
                        df_all.to_csv(DATA_FILE, index=False)
                        st.success("Profil pasien berhasil diperbarui dengan Smart Merge!")
                        st.session_state.profile_updates = {}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal memperbarui profil: {e}")

# --- TAB 3: MEDICAL HISTORY ---
with t_hist:
    if active_patient is not None:
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
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("🔍 Info Darurat")
        if active_patient is not None:
            st.error(f"**PASIEN: {active_patient['Name']}**")
            st.markdown(f"""
            - **Gol. Darah:** {active_patient['Blood_Type']}
            - **Alergi:** {active_patient['Allergies']}
            - **Kontak:** {active_patient['Emergency_Contact_Phone']}
            """)
        else: st.info("Scan QR atau pilih pasien.")
        
        st.divider()
        v_rr = st.number_input("RR (bpm)", 5, 50, 20)
        v_spo2 = st.number_input("SpO2 (%)", 50, 100, 95)
        v_bps = st.number_input("Sistolik", 50, 250, 120)
        v_hr = st.number_input("HR (bpm)", 20, 200, 80)
        v_avpu = st.selectbox("AVPU", ["Alert", "Voice", "Pain", "Unresponsive"])

    with c2:
        if st.button("HITUNG TRIAGE NEWS2", use_container_width=True, type="primary"):
            score, level, color = calculate_news2(v_rr, v_spo2, v_bps, v_hr, v_avpu)
            st.markdown(f"<div style='background:{color}; color:white; padding:30px; border-radius:15px; text-align:center;'><h1>{level}</h1><h3>Score: {score}</h3></div>", unsafe_allow_html=True)
            
            # Geofencing
            u_lat, u_lon = -7.7700, 110.3700
            distances = [{"name": h["name"], "dist": geodesic((u_lat, u_lon), (h["lat"], h["lon"])).km} for h in HOSPITALS]
            nearest = sorted(distances, key=lambda x: x["dist"])[0]
            st.metric("Rumah Sakit Terdekat", nearest["name"], f"{nearest['dist']:.2f} KM")

# --- TAB 5: ADMIN ---
with t_admin:
    pwd = st.text_input("Password Admin", type="password")
    if pwd == MASTER_PASSWORD:
        st.subheader("Data Master Pasien")
        st.dataframe(pd.read_csv(DATA_FILE), use_container_width=True)
        st.subheader("Data Transaksi SOAP")
        st.dataframe(pd.read_csv(ENCOUNTER_FILE), use_container_width=True)
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

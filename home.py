import streamlit as st
import leafmap.foliumap as leafmap
from epistemx import auto_initialize
from modules.nav import Navbar
import os
import tempfile
from ui_helper import show_footer, show_header

st.set_page_config(layout="wide")

# Load custom CSS
def load_css():
    """Load custom CSS for EpistemX theme"""
    try:
        with open('.streamlit/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file not found, continue without custom styling

# Apply custom theme
load_css()
# Show header
show_header()

# Set up temporary directory for leafmap
if not os.path.exists('/tmp'):
    os.makedirs('/tmp', exist_ok=True)
os.environ['TMPDIR'] = '/tmp'

# Initialize Earth Engine once when the app starts
if 'ee_initialized' not in st.session_state:
    try:
        result = auto_initialize()
        st.session_state.ee_initialized = result
        if result:
            st.success("Earth Engine berhasil diinisialisasi!")
        else:
            st.error("Gagal menginisialisasi Earth Engine. Mohon periksa autentikasi Anda.")
    except Exception as e:
        st.error(f"Earth Engine initialization error: {e}")
        st.session_state.ee_initialized = False

# Add navigation sidebar
Navbar()

st.markdown("""
<div class="main-header">
    <h2>Selamat Datang di</h2>
    <h1>Platform Pemetaan Tutupan Lahan<br>EpistemX</h1>
    <p>Pemrosesan & Analisis Data Observasi Bumi Lanjutan.<br>Platform multi-halaman ini mendemonstrasikan kemampuan pemetaan tutupan lahan EpistemX yang canggih, 
    dengan fitur pemrosesan citra Landsat otomatis untuk area minat Anda.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="module-header">📋 Instruksi</div>', unsafe_allow_html=True)

st.markdown("""
<div class="epistemx-card">
    <div class="steps">
        <div class="step">
            <strong>1. Tentukan Area Minat:</strong>
            <p>Gambar persegi panjang di peta atau unggah shapefile (zip)</p>
        </div>
        <div class="step">
            <strong>2. Tentukan Tanggal Akuisisi:</strong>
            <p>Tentukan tahun - citra akan difilter dari 1 Januari hingga 31 Desember</p>
        </div>
        <div class="step">
            <strong>3. Konfigurasi Parameter:</strong>
            <p>Tentukan persentase tutupan awan dan tipe sensor (Landsat 5 TM - Landsat 9 OLI2)</p>
        </div>
        <div class="step">
            <strong>4. Buat Mozaik:</strong>
            <p>Klik jalankan untuk membuat mozaik citra satelit Anda</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Status indicator
if st.session_state.get('ee_initialized', False):
    st.markdown('<div class="success-message">✅ Earth Engine Siap - Anda dapat melanjutkan analisis</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="warning-message">⚠️ Earth Engine Belum Diinisialisasi - Mohon periksa autentikasi</div>', unsafe_allow_html=True)

m = leafmap.Map(center = [-5.003394, 113.598633], zoom = 5)
m.add_basemap("OpenTopoMap")
m.to_streamlit(height=500)

show_footer()

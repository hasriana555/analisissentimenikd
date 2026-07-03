import streamlit as st
import joblib
import re
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
import nltk

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="Analisis Sentimen IKD",
    page_icon="📱",
    layout="centered"
)

# =============================================
# LOAD MODEL DAN VECTORIZER
# =============================================
@st.cache_resource
def load_model():
    model   = joblib.load('model_svm.pkl')
    tfidf   = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

@st.cache_resource
def load_preprocessing_tools():
    nltk.download('stopwords', quiet=True)

    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()

    factory_stop = StopWordRemoverFactory()
    sastrawi_sw  = set(factory_stop.get_stop_words())
    nltk_sw      = set(stopwords.words('indonesian'))
    all_stopwords = sastrawi_sw.union(nltk_sw)

    custom_stopwords = {
        'app', 'apps', 'aplikasi', 'ikd', 'ktp', 'dukcapil',
        'mobile', 'hp', 'android', 'update', 'versi', 'ya',
        'yg', 'aja', 'sy', 'aku', 'ku', 'wae', 'aya', 'tee',
        'mah', 'atuh', 'euy', 'ae', 'rek', 'in', 'log',
        'astagfirullan', 'astaghfirullah', 'astagfirullah',
    }
    all_stopwords = all_stopwords.union(custom_stopwords)

    slang_dict = {
        'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak',
        'nggak': 'tidak', 'gk': 'tidak', 'tdk': 'tidak',
        'kga': 'tidak', 'kagak': 'tidak',
        'knp': 'kenapa', 'klo': 'kalau', 'klw': 'kalau',
        'apk': 'aplikasi', 'dlm': 'dalam',
        'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
        'blm': 'belum', 'bgt': 'sangat', 'banget': 'sangat',
        'yg': 'yang', 'krn': 'karena', 'karna': 'karena',
        'dgn': 'dengan', 'utk': 'untuk',
        'tp': 'tapi', 'tpi': 'tapi',
        'mau': 'ingin', 'pengen': 'ingin',
        'bs': 'dapat', 'lg': 'lagi',
        'lemot': 'lambat', 'susah': 'sulit', 'ribet': 'rumit',
        'eror': 'error', 'err': 'error',
        'loading': 'muat', 'load': 'muat',
        'download': 'unduh', 'dowload': 'unduh',
        'login': 'masuk', 'bet': 'sangat',
        'gjls': 'tidak jelas', 'gajelas': 'tidak jelas',
    }

    return stemmer, all_stopwords, slang_dict

# =============================================
# FUNGSI PREPROCESSING
# =============================================
def preprocessing(teks, stemmer, all_stopwords, slang_dict):
    if not teks or teks.strip() == '':
        return ''

    # 1. Case folding
    teks = teks.lower()

    # 2. Cleaning
    teks = re.sub(r'http\S+|www\S+', '', teks)
    teks = re.sub(r'@\w+|#\w+', '', teks)
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = re.sub(r'\d+', '', teks)
    teks = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()

    # 3. Normalisasi slang
    kata_kata = teks.split()
    teks = ' '.join([slang_dict.get(kata, kata) for kata in kata_kata])

    # 4. Tokenizing
    tokens = teks.split()

    # 5. Stopword removal
    tokens = [k for k in tokens if k not in all_stopwords]

    # 6. Stemming
    tokens = [stemmer.stem(k) for k in tokens]

    return ' '.join(tokens)

# =============================================
# LOAD SEMUA TOOLS
# =============================================
model, tfidf = load_model()
stemmer, all_stopwords, slang_dict = load_preprocessing_tools()

# =============================================
# TAMPILAN APLIKASI
# =============================================
st.title("📱 Analisis Sentimen Ulasan Aplikasi IKD")
st.markdown(
    "Aplikasi ini memprediksi sentimen ulasan pengguna aplikasi "
    "**Identitas Kependudukan Digital (IKD)** menggunakan algoritma "
    "**Support Vector Machine (SVM)**."
)

st.divider()

# ===== INPUT ULASAN =====
st.subheader("📝 Input Ulasan")
ulasan = st.text_area(
    label       = "Masukkan ulasan pengguna aplikasi IKD:",
    placeholder = "Contoh: Aplikasi sangat membantu, tidak perlu bawa KTP fisik kemana-mana...",
    height      = 150
)

col1, col2 = st.columns([1, 4])
with col1:
    prediksi_btn = st.button("🔍 Prediksi", type="primary", use_container_width=True)
with col2:
    reset_btn = st.button("🔄 Reset", use_container_width=True)

if reset_btn:
    st.rerun()

# ===== HASIL PREDIKSI =====
if prediksi_btn:
    if ulasan.strip() == '':
        st.warning("⚠️ Silakan masukkan ulasan terlebih dahulu.")
    else:
        with st.spinner('Memproses ulasan...'):
            # Preprocessing
            teks_bersih = preprocessing(ulasan, stemmer, all_stopwords, slang_dict)

            if teks_bersih.strip() == '':
                st.error("❌ Teks tidak dapat diproses. Coba masukkan ulasan yang lebih lengkap.")
            else:
                # Prediksi
                teks_tfidf  = tfidf.transform([teks_bersih])
                hasil       = model.predict(teks_tfidf)[0]
                confidence  = model.decision_function(teks_tfidf)[0]

                st.divider()
                st.subheader("📊 Hasil Prediksi")

                if hasil == 'Positif':
                    st.success(f"✅ **Sentimen: POSITIF**")
                    st.markdown(
                        "Ulasan ini mengandung sentimen **positif** — "
                        "pengguna cenderung merasa puas dengan aplikasi IKD."
                    )
                else:
                    st.error(f"❌ **Sentimen: NEGATIF**")
                    st.markdown(
                        "Ulasan ini mengandung sentimen **negatif** — "
                        "pengguna cenderung tidak puas atau mengalami masalah dengan aplikasi IKD."
                    )

                # Detail preprocessing
                with st.expander("🔍 Lihat detail preprocessing"):
                    st.markdown(f"**Teks asli:**")
                    st.info(ulasan)
                    st.markdown(f"**Teks setelah preprocessing:**")
                    st.info(teks_bersih if teks_bersih else "(kosong setelah preprocessing)")

st.divider()

# ===== INFO MODEL =====
st.subheader("📈 Informasi Model")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Akurasi",  "88.38%")
col2.metric("Presisi",  "86.21%")
col3.metric("Recall",   "83.68%")
col4.metric("F1-Score", "84.93%")

st.divider()

# ===== INFO DATASET =====
st.subheader("📂 Informasi Dataset")

col1, col2, col3 = st.columns(3)
col1.metric("Total Data",   "3.031")
col2.metric("Data Positif", "1.212")
col3.metric("Data Negatif", "1.867")

st.caption(
    "Data ulasan diambil dari Google Play Store aplikasi IKD "
    "(gov.dukcapil.mobile_id) menggunakan teknik web scraping."
)

st.divider()

# ===== FOOTER =====
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px'>"
    "Dibuat untuk keperluan penelitian skripsi | "
    "Algoritma: Support Vector Machine (SVM) | "
    "Fitur: TF-IDF"
    "</div>",
    unsafe_allow_html=True
)

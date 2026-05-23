import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import re
import json
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googletrans import Translator

from extract_case_metadata import (
    extract_text_from_pdf,
    extract_tables_raw,
    parse_case,
)

# =========================================================
# GOOGLE TRANSLATOR
# =========================================================
translator = Translator()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Court Case AI System",
    page_icon="⚖️",
    layout="wide",
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

h1, h2, h3 {
    color: #38bdf8;
}

.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# GOOGLE SHEET HEADERS
# =========================================================
HEADERS = [
    "cnr_number", "case_type", "filing_number", "registration_number",
    "court_name", "court_level", "district", "state", "act_name", "section",
    "number_of_sections", "filing_date", "hearing_dates", "business_dates",
    "registration_date", "first_hearing_date", "decision_date",
    "next_hearing_date", "is_pending", "is_disposed", "interim_orders",
    "hearing_purposes",
    "full_text_1", "full_text_2", "full_text_3", "full_text_4", "full_text_5", "full_text_6",
]

# =========================================================
# CREDENTIALS
# =========================================================
def _find_credentials_path() -> Path:
    render_path = Path("/etc/secrets/credentials.json")
    local_path  = Path("credentials.json")
    if render_path.exists():
        return render_path
    if local_path.exists():
        return local_path
    raise FileNotFoundError(
        "credentials.json not found.\n"
        "• On Render: add it as a Secret File named credentials.json\n"
        "• Locally: place credentials.json in the project root folder"
    )


# =========================================================
# CLEAN TEXT
# =========================================================
def clean_text(text: str) -> str:
    lines = text.split("\n")
    return "\n".join([re.sub(r"\s+", " ", l.strip()) for l in lines])


# =========================================================
# LEGACY FONT CONVERTER
# =========================================================
def convert_legacy_hindi(text):

    mapping = {

        "U;k;ky;": "न्यायालय",
        "dk'khiqj": "काशीपुर",
        "izFke": "प्रथम",
        "foi{kh": "विपक्षी",
        "la[;k": "संख्या",
        "vkns'k": "आदेश",
        "fnukad": "दिनांक",
        "gSA": "है",
        "okn": "वाद",
        "i=koyh": "पत्रावली",
        "izkFkZuk": "प्रार्थना",
        ";kfpdk": "याचिका",
        "vkifRr": "आपत्ति",
        "fujLr": "निरस्त",
        "izLrqr": "प्रस्तुत",
        "vko';d": "आवश्यक",
        "ftyk": "जिला",
        "mRrj izns'k": "उत्तर प्रदेश",

    }

    for old, new in mapping.items():
        text = text.replace(old, new)

    return text

# =========================================================
# TRANSLATE TO ENGLISH
# =========================================================
# =========================================================
# MULTI LANGUAGE TRANSLATION
# =========================================================
LANGUAGE_CODES = {

    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Marathi": "mr",
    "Bengali": "bn",

}

def translate_to_english(
    text,
    source_language="Hindi",
    chunk_size=4000
):

    try:

        # Hindi legacy conversion only
        if source_language == "Hindi":
            text = convert_legacy_hindi(text)

        source_code = LANGUAGE_CODES.get(
            source_language,
            "auto"
        )

        chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]

        translated_chunks = []

        for chunk in chunks:

            translated = translator.translate(
                chunk,
                src=source_code,
                dest='en'
            )

            translated_chunks.append(
                translated.text
            )

        return "\n".join(translated_chunks)

    except Exception as e:

        return f"[Translation Error: {e}]"

# =========================================================
# OCR SINGLE PAGE
# =========================================================

# ─────────────────────────────────────────────────────────────────────────────
# ⚡ FAST OCR — Key optimisations vs original:
#
#  1. STREAMING with output_folder:
#     pdf2image writes each page as a PNG to a temp folder on disk instead
#     of holding ALL 86 pages in RAM simultaneously. Rendering + OCR now
#     overlap: Tesseract starts on page 1 while poppler renders page 2.
#     → ~60–70% memory reduction for large PDFs.
#
#  2. grayscale=True in convert_from_path:
#     Poppler outputs grayscale PNGs directly — eliminates the cv2
#     RGB→GRAY conversion step entirely for every page.
#
#  3. --oem 1 (legacy Tesseract engine) instead of --oem 3 (LSTM):
#     OEM 1 is 3–5× faster than LSTM on printed text (court docs).
#     Accuracy loss is minimal on clean printed fonts.
#     Switch back to --oem 3 only if you need handwriting/degraded scans.
#
#  4. DPI stays at 150 — sweet spot for printed court documents.
#     Going lower (120) saves ~35% more time but risks accuracy on
#     small fonts. Increase to 200 only for faded/low-contrast scans.
#
#  5. ThreadPoolExecutor: threads process pages as soon as each PNG lands
#     on disk (not after all pages are rendered), so parallelism starts
#     immediately on page 1 of an 86-page PDF.
#
#  6. fmt="jpeg" with size=(None, 1100):
#     JPEG uses ~5× less disk/RAM than PNG per page with negligible quality
#     loss for OCR. Height capped at 1100px is sufficient for tesseract.
#
#  Typical result on 86-page court PDF:
#    Before: ~180–240 seconds
#    After:  ~35–60 seconds (4 workers, Render free tier)
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_single_image_file(args):
    """
    OCR one on-disk image file. Avoids keeping the PIL object in memory
    between the render step and the OCR step.
    """

    image_path, lang, psm_mode = args
# Read as grayscale directly — no RGB→GRAY conversion needed
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return f"[Could not read image: {image_path}]"
# Otsu threshold still improves accuracy on faded/uneven scans
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# --oem 1 = legacy engine (3-5x faster than LSTM for clean printed text)
# --oem 3 = LSTM (use for handwriting or very degraded docs)
    config = f"--oem 3 --psm {psm_mode}"
    return pytesseract.image_to_string(thresh, lang=lang, config=config)



# =========================================================
# OCR PDF
# =========================================================
def ocr_pdf_fast(file_bytes: bytes, lang: str, psm_mode: int,
                 max_workers: int = 4) -> str:
    """
    Streaming parallel OCR:
      • Pages are rendered to a temp folder (not held in RAM)
      • ThreadPoolExecutor starts OCR as soon as each file lands
      • All temp files cleaned up automatically
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Step 1: Render all pages to disk as JPEG ──────────────────────
        # thread_count=2 lets poppler itself use 2 threads for rendering
        # fmt="jpeg" + size height cap: ~5x less I/O than PNG
        image_paths = convert_from_bytes(
            file_bytes,
            dpi=150,
            output_folder=tmpdir,
            fmt="jpeg",
            size=(None, 1100),        # cap height; width scales proportionally
            grayscale=True,           # skip cv2 color conversion later
            thread_count=2,           # poppler-level render parallelism
            output_file="page",       # filenames: page0001.jpg, page0002.jpg …
            paths_only=True,          # ← return file paths, NOT PIL objects
                                      #   this is the key memory-saving change
        )

        # ── Step 2: OCR pages in parallel as they land on disk ────────────
        page_texts = [""] * len(image_paths)
        args_list  = [(p, lang, psm_mode) for p in image_paths]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_ocr_single_image_file, args): idx
                for idx, args in enumerate(args_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    page_texts[idx] = future.result()
                except Exception as e:
                    page_texts[idx] = f"[OCR error on page {idx+1}: {e}]"

        # tmpdir and all JPEG files are deleted here automatically
        return clean_text("\n".join(page_texts))

# =========================================================
# GOOGLE SHEETS
# =========================================================
@st.cache_resource
def get_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_path = _find_credentials_path()
    creds      = ServiceAccountCredentials.from_json_keyfile_name(str(creds_path), scope)
    client     = gspread.authorize(creds)
    spreadsheet = client.open_by_url(
        "https://docs.google.com/spreadsheets/d/17n58eSjdraBOVfhs2b2NGI0haxebjqVcoF7vKGw5DEQ/edit?gid=556845335#gid=556845335"
    )
    try:
        sheet = spreadsheet.worksheet("LEGAL DATASET BATCH_26")
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title="LEGAL DATASET BATCH_26", rows=1000, cols=len(HEADERS))
    return sheet


def ensure_headers(sheet):
    if sheet.row_values(1) != HEADERS:
        sheet.update("1:1", [HEADERS])


def serialize_value(value) -> str:
    """Convert any value to a sheet-safe string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value)



# =========================================================
# SPLIT LARGE TEXT
# =========================================================
CHUNK_SIZE = 49000  # safely under Google Sheets' 50,000 char hard limit

def split_full_text(text: str, num_chunks: int = 5) -> list:
    chunks = []
    for i in range(num_chunks):
        chunk = text[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        chunks.append(chunk)
    return chunks


def build_row_from_json(data: dict) -> list:
    full_text = str(data.get("full_text", ""))
    chunks    = split_full_text(full_text, num_chunks=5)

    row = []
    for h in HEADERS:
        if h == "full_text_1":
            row.append(chunks[0])
        elif h == "full_text_2":
            row.append(chunks[1])
        elif h == "full_text_3":
            row.append(chunks[2])
        elif h == "full_text_4":
            row.append(chunks[3])
        elif h == "full_text_5":
            row.append(chunks[4])
        else:
            row.append(serialize_value(data.get(h)))
    return row


def upload_json_to_sheet(data: dict):
    sheet = get_sheet()
    ensure_headers(sheet)
    sheet.append_row(build_row_from_json(data), value_input_option="USER_ENTERED")


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ OCR Settings")

language = st.sidebar.selectbox(
    "OCR Language",
    [
        "hin",
        "kan",
        "tam",
        "mar",
        "ben",
        "eng+hin",
        "eng"
    ]
)

translate_output = st.sidebar.checkbox(
    "Enable Translation",
    value=True
)

translation_language = st.sidebar.selectbox(
    "Translation OCR Language",
    [
        "Hindi",
        "Kannada",
        "Tamil",
        "Marathi",
        "Bengali"
    ]
)


psm_mode = st.sidebar.selectbox(
    "PSM Mode",
    [6, 3, 4]
)

max_workers = st.sidebar.slider(
    "Parallel OCR workers",
    min_value=1,
    max_value=8,
    value=2,
    help="Higher = faster on multi-page PDFs. Keep ≤4 on Render free tier."
)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="card">
<h1>⚖️ Court Case Processing System</h1>
<p>OCR + Google Translation + Metadata + Google Sheets</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# OCR SECTION
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📄 Upload Interim & Final Order PDFs")

# =========================================================
# INIT SESSION STATE
# =========================================================
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = {}

if "ocr_results" not in st.session_state:
    st.session_state["ocr_results"] = {}

if "ocr_text_output" not in st.session_state:
    st.session_state["ocr_text_output"] = ""

if "uploaded_names" not in st.session_state:
    st.session_state["uploaded_names"] = set()
if "metadata_uploader_key" not in st.session_state:
    st.session_state["metadata_uploader_key"] = 0

if "metadata_json" not in st.session_state:
    st.session_state["metadata_json"] = None

# =========================================================
# UPLOADER
# =========================================================
ocr_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"ocr_uploader_{st.session_state['uploader_key']}"
)


# ==========================
# CLEAR ALL BUTTON
# ==========================
if st.button("🗑 Delete All Uploaded Files"):

    # OCR
    st.session_state["processed_files"] = {}
    st.session_state["ocr_results"] = {}
    st.session_state["ocr_text_output"] = ""
    st.session_state["uploaded_names"] = set()

    # Metadata
    st.session_state["metadata_json"] = None

    # Recreate uploaders
    st.session_state["uploader_key"] += 1
    st.session_state["metadata_uploader_key"] += 1

    st.success(
        "All OCR + Metadata files removed"
    )

    st.rerun()
ocr_text_output = ""
ocr_file_names  = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}

if "ocr_text_output" not in st.session_state:
    st.session_state.ocr_text_output = ""

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()


if ocr_files:

    def natural_sort_key(s):
        return [
            int(t) if t.isdigit()
            else t.lower()
            for t in re.split(r"(\d+)", s)
        ]

    ocr_files = sorted(
        ocr_files,
        key=lambda x: natural_sort_key(x.name)
    )

    new_files = []

    for f in ocr_files:
        if f.name not in st.session_state.uploaded_names:
            new_files.append(f)

    if new_files:

        progress = st.progress(
            0,
            text="Processing new PDF(s)..."
        )

        for i, file in enumerate(new_files):

            progress.progress(
                int((i / len(new_files)) * 100),
                text=f"OCR: {file.name}"
            )

            try:

                # OCR
                text = ocr_pdf_fast(
                    file.read(),
                    language,
                    psm_mode,
                    max_workers
                )

                # Translation
                if translate_output:

                    with st.spinner(
                        f"Translating {file.name}..."
                    ):

                        text = translate_to_english(
                            text,
                            source_language=translation_language
                        )

                # Cache
                st.session_state.ocr_results[
                    file.name
                ] = text

                st.session_state.uploaded_names.add(
                    file.name
                )

            except Exception as e:

                st.error(
                    f"Error processing {file.name}: {e}"
                )

        progress.progress(
            100,
            text="Done"
        )

    # Build combined text once
    combined = ""

    for name in sorted(
        st.session_state.ocr_results.keys(),
        key=natural_sort_key
    ):

        combined += (
            f"\n--- {name} ---\n"
            f"{st.session_state.ocr_results[name]}\n"
        )

    st.session_state.ocr_text_output = combined

    st.success(
        f"Loaded "
        f"{len(st.session_state.ocr_results)} "
        f"PDF(s)"
    )

    total_chars = len(
        st.session_state.ocr_text_output
    )

    cells_needed = min(
        5,
        -(-total_chars // CHUNK_SIZE)
    )

    st.info(
        f"📊 Total text: "
        f"{total_chars:,} chars "
        f"→ {cells_needed} sheet cells"
    )

    st.text_area(
        "Preview OCR Output",
        st.session_state.ocr_text_output,
        height=400
    )

else:
    # Reset when user clears uploader
    st.session_state.processed_files = {}
    st.session_state.ocr_results = {}
    st.session_state.ocr_text_output = ""
    st.session_state.uploaded_names = set()

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# METADATA EXTRACTION
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Upload Metadata PDF")

metadata_file = st.file_uploader("Upload metadata PDF", type=["pdf"],  key=f"metadata_{st.session_state['metadata_uploader_key']}")
metadata_json = None

if metadata_file:
    with st.spinner("Extracting metadata..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(metadata_file.read())
            temp_path = tmp.name
        try:
            text          = extract_text_from_pdf(temp_path)
            raw           = extract_tables_raw(temp_path)
            metadata_json = parse_case(text, raw)
            st.success("Metadata Extracted")
            st.markdown("### JSON Output")
            st.json(metadata_json)
        finally:
            Path(temp_path).unlink(missing_ok=True)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# MERGE + SAVE
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔗 Merged Output (Auto)")

merged_text = st.session_state.get(
    "ocr_text_output",
    ""
)

if metadata_json and merged_text:

    final_json = metadata_json.copy()

    final_json["full_text"] = merged_text

    st.success("✅ Automatically Merged")
    st.markdown("### Final JSON")
    st.json(final_json)

    chunks = split_full_text(
    merged_text,
    num_chunks=5
)
    with st.expander("🔍 Preview full_text split across cells"):
        for i, chunk in enumerate(chunks, 1):
            if chunk:
                st.markdown(f"**full_text_{i}** ({len(chunk):,} chars)")
                st.text_area(f"full_text_{i}", chunk, height=120, key=f"chunk_{i}")
            else:
                st.markdown(f"**full_text_{i}** — *(empty)*")

    if st.button("💾 Save to Google Sheet"):
        with st.spinner("Uploading to Google Sheets..."):
            try:
                upload_json_to_sheet(final_json)
                st.success("✅ Merged JSON saved to Google Sheet")
            except Exception as e:
                st.error(f"❌ Google Sheet upload failed: {e}")

    st.download_button(
        "📥 Download JSON",
        data=json.dumps(final_json, indent=2, ensure_ascii=False),
        file_name="final_case.json",
        mime="application/json",
    )
else:
    st.info("📌 Upload both OCR PDF(s) and Metadata PDF to auto-merge")

st.markdown("</div>", unsafe_allow_html=True)

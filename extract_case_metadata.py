"""
Court Case Metadata Extractor
==============================
Extracts structured metadata from Indian eCourts PDF case detail pages.

Requirements:
    pip install pdfplumber

Usage (Jupyter Notebook):
    process_folder("./pdfs", "cases_metadata.json")

Usage (Terminal):
    python extract_case_metadata.py --input ./pdfs --output cases.json
"""

import re
import sys
import json
import argparse
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    raise ImportError("Please install pdfplumber: pip install pdfplumber")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def clean(text) -> str:
    if text is None:
        return ""
    # Replace newlines with space, collapse whitespace
    return re.sub(r'\s+', ' ', str(text)).strip()


def cell(text) -> str:
    """Clean a table cell — strips newlines inside dates too."""
    if text is None:
        return ""
    return re.sub(r'\s+', '', str(text)).strip()   # no spaces at all (good for dates)


def find_field(pattern: str, text: str, group: int = 1, default=None):
    m = re.search(pattern, text, re.IGNORECASE)
    return clean(m.group(group)) if m else default


# ─────────────────────────────────────────────
# Date normaliser → DD-MM-YYYY
# ─────────────────────────────────────────────

MONTH_MAP = {
    "january":"01","february":"02","march":"03","april":"04",
    "may":"05","june":"06","july":"07","august":"08",
    "september":"09","october":"10","november":"11","december":"12",
    "jan":"01","feb":"02","mar":"03","apr":"04",
    "jun":"06","jul":"07","aug":"08",
    "sep":"09","oct":"10","nov":"11","dec":"12",
}

DATE_PATTERN = re.compile(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}')


def normalise_date(raw) -> str | None:
    """
    Accepts:
      '06-03-2026', '06-03-\n2026', '06-03- 2026'  → '06-03-2026'
      '6th March 2026', '28th April 2025'           → '06-03-2026'
      '↑\n05th May 2026'  (arrow icon prefix)       → '05-05-2026'
    """
    if not raw:
        return None

    raw_str = str(raw)

    # ── Strip leading/trailing non-date junk (arrows ↑↓, icons, symbols) ──
    # Remove any characters before the first digit or letter
    raw_str = re.sub(r'^[^a-zA-Z0-9]+', '', raw_str.strip())
    raw_str = re.sub(r'[^a-zA-Z0-9\s\-/]+$', '', raw_str).strip()

    if not raw_str:
        return None

    # Strip ALL internal whitespace (fixes '06-03-\n2026', '06-03- 2026')
    s = re.sub(r'\s+', '', raw_str)

    # DD-MM-YYYY or DD/MM/YYYY
    m = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$', s)
    if m:
        return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}-{m.group(3)}"

    # "6th March 2026" / "05th May 2026" — needs spaces preserved
    m = re.match(r'^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})$', raw_str, re.I)
    if m:
        month = MONTH_MAP.get(m.group(2).lower())
        if month:
            return f"{int(m.group(1)):02d}-{month}-{m.group(3)}"

    # "March 6, 2026"
    m = re.match(r'^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$', raw_str, re.I)
    if m:
        month = MONTH_MAP.get(m.group(1).lower())
        if month:
            return f"{int(m.group(2)):02d}-{month}-{m.group(3)}"

    return None


# ─────────────────────────────────────────────
# PDF extraction
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Flatten all table rows + plain text into a single string."""
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    parts = [clean(c) for c in row if c and clean(c)]
                    if parts:
                        lines.append(" ".join(parts))
            text = page.extract_text()
            if text:
                lines.append(text)
    return "\n".join(lines)


def extract_tables_raw(pdf_path: str) -> list:
    """Return every raw table row from all pages."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                rows.extend(table)
    return rows


# ─────────────────────────────────────────────
# Court level
# ─────────────────────────────────────────────

def detect_court_level(court_name: str) -> str:
    name = court_name.lower()
    if "high court" in name:
        return "High Court"
    if any(k in name for k in ["sessions", "district"]):
        return "Sessions"
    if any(k in name for k in ["magistrate", "judicial", "jmfc", "acjm", "cjm"]):
        return "Magistrate"
    if any(k in name for k in ["civil judge", "civil court", "cj sr", "cj jr",
                            "sr. divn", "jr. divn", "sr divn", "jr divn", "civil"]):
        return "Civil"
    return "Unknown"


# ─────────────────────────────────────────────
# Location from CNR + court name
# ─────────────────────────────────────────────

CNR_DISTRICT_MAP = {
    # Tripura
    "TRWT": ("West Tripura", "Tripura"),
    "TRSE": ("Sepahijala", "Tripura"),
    "TRDL": ("Dhalai", "Tripura"),
    "TRNT": ("North Tripura", "Tripura"),
    "TRST": ("South Tripura", "Tripura"),
    "TRKT": ("Khowai", "Tripura"),
    "TRGT": ("Gomati", "Tripura"),
    "TRUT": ("Unakoti", "Tripura"),

    # Sikkim
    "SKNM": ("Namchi (South Sikkim)", "Sikkim"),
    "SKGT": ("Gangtok (East Sikkim)", "Sikkim"),
    "SKMN": ("Mangan (North Sikkim)", "Sikkim"),
    "SKGZ": ("Gyalshing (West Sikkim)", "Sikkim"),
    "SKSR": ("Soreng", "Sikkim"),
    "SKPK": ("Pakyong", "Sikkim"),

    # Assam
    "ASKM": ("Kamrup Metropolitan", "Assam"),
    "ASKR": ("Kamrup", "Assam"),
    "ASDB": ("Dibrugarh", "Assam"),
    "ASCC": ("Cachar", "Assam"),
    "ASNG": ("Nagaon", "Assam"),
    "ASJR": ("Jorhat", "Assam"),
    "ASGA": ("Goalpara", "Assam"),
    "ASTN": ("Tinsukia", "Assam"),
    "ASKJ": ("Karimganj", "Assam"),
    "ASNL": ("Nalbari", "Assam"),
    "ASDU": ("Dhubri", "Assam"),
    "ASGL": ("Golaghat", "Assam"),
    "ASBR": ("Barpeta/Bajali", "Assam"),
    "ASHI": ("Hailakandi", "Assam"),
    "ASKK": ("Kokrajhar", "Assam"),
    "ASUA": ("Udalguri", "Assam"),
    "ASCH": ("Chirang", "Assam"),
    "ASKA": ("Karbi Anglong", "Assam"),
    "ASDM": ("Dima Hasao", "Assam"),
    "ASCA": ("Charaideo", "Assam"),
    "ASHJ": ("Hojai", "Assam"),
    "ASSS": ("South Salmara-Mankachar", "Assam"),
    "ASSV": ("Sivasagar", "Assam"),
    "ASSN": ("Sonitpur/Biswanath", "Assam"),
    "ASMJ": ("Majuli", "Assam"),
    "ASWP": ("West Karbi Anglong", "Assam"),
    "ASMR": ("Morigaon", "Assam"),
    "ASLK": ("Lakhimpur", "Assam"),
    "ASDR": ("Darrang", "Assam"),
    "ASBN": ("Bongaigaon", "Assam"),
    "ASDM": ("Dhemaji", "Assam"),
    "ASBK": ("Baksa", "Assam"),

    # Manipur
    "MNIW": ("Imphal West", "Manipur"),
    "MNIE": ("Imphal East", "Manipur"),
    "MNBP": ("Bishnupur", "Manipur"),
    "MNNT": ("Thoubal", "Manipur"),
    "MNSP": ("Senapati", "Manipur"),
    "MNUK": ("Ukhrul", "Manipur"),
    "MNCP": ("Churachandpur", "Manipur"),
    "MNCD": ("Chandel", "Manipur"),
    "MNTL": ("Tamenglong", "Manipur"),
    
    
    # Meghalaya
    "MLSH": ("East Khasi Hills (Shillong)", "Meghalaya"),
    "MLWK": ("West Khasi Hills", "Meghalaya"),
    "MLSW": ("South West Khasi Hills", "Meghalaya"),
    "MLEW": ("Eastern West Khasi Hills", "Meghalaya"),
    "MLRB": ("Ri Bhoi", "Meghalaya"),
    "MLWJ": ("West Jaintia Hills", "Meghalaya"),
    "MLEJ": ("East Jaintia Hills", "Meghalaya"),
    "MLWG": ("West Garo Hills", "Meghalaya"),
    "MLEG": ("East Garo Hills", "Meghalaya"),
    "MLSG": ("South Garo Hills", "Meghalaya"),
    "MLNG": ("North Garo Hills", "Meghalaya"),
    "MLSW": ("South West Garo Hills", "Meghalaya"),
    "MLKL": ("East Jaintia Hills (Khliehriat)", "Meghalaya"),
    "MLMR": ("Eastern West Khasi Hills (Mairang)", "Meghalaya"),
    "MLTU": ("West Garo Hills (Tura)", "Meghalaya"),
    "MLJW": ("West Jaintia Hills (Jowai)", "Meghalaya"),
    "ML07": ("East Garro Hills", "Meghalaya"),  # old code for East Garo Hills, still appears in some CNRs
    "ML12": ("South West Khasi Hills", "Meghalaya"),  # old code for South West Khasi Hills, still appears in some CNRs
    "MLNS": ("West Khasi Hills (Nongstoin)", "Meghalaya"),
    "MLAP": ("South West Garo Hills (Ampati)", "Meghalaya"),
    "MLKA": ("Khasi Hills Autonomous District", "Meghalaya"),
    "MLJA": ("Jaintia Hills Autonomous District", "Meghalaya"),


    # Nagaland
    "NLDM": ("Dimapur", "Nagaland"),
    "NLKO": ("Kohima", "Nagaland"),
    "NLMG": ("Mokokchung", "Nagaland"),
    "NLMO": ("Mon", "Nagaland"),
    "NLWK": ("Wokha", "Nagaland"),
    "NLZB": ("Zunheboto", "Nagaland"),
    "NLPK": ("Phek", "Nagaland"),
    "NLKR": ("Kiphire", "Nagaland"),
    "NLLN": ("Longleng", "Nagaland"),

    # Arunachal Pradesh
    "ARPP": ("Papum Pare", "Arunachal Pradesh"),
    "ARTA": ("Tawang", "Arunachal Pradesh"),
    "ARTI": ("Tirap", "Arunachal Pradesh"),
    "ARWS": ("West Siang", "Arunachal Pradesh"),
    "ARES": ("East Siang", "Arunachal Pradesh"),
    "ARTE": ("Lohit(Tezu)", "Arunachal Pradesh"),
    "ARLE": ("Lepa Rada", "Arunachal Pradesh"),
    "AREK": ("East Kameng", "Arunachal Pradesh"),
    "ARLD": ("Lower Dibang Valley", "Arunachal Pradesh"),
    "ARNA": ("Namsai", "Arunachal Pradesh"),
    "ARWK": ("West Kameng", "Arunachal Pradesh"),
    "ARUP": ("Upper Siang", "Arunachal Pradesh"),
    "ARAN": ("Anjaw", "Arunachal Pradesh"),
    "ARLS": ("Lower Subansiri", "Arunachal Pradesh"),
    "ARSI": ("Siang", "Arunachal Pradesh"),
    "ARKD": ("Kra Dadi", "Arunachal Pradesh"),
    "ARKK": ("Kurung Kumey", "Arunachal Pradesh"),

    # Mizoram
    "MZAZ": ("Aizawl", "Mizoram"),
    "MZKL": ("Kolasib", "Mizoram"),
    "MZSC": ("Serchhip", "Mizoram"),
    "MZLL": ("Lunglei", "Mizoram"),
    "MZLT": ("Lawngtlai", "Mizoram"),
    "MZCH": ("Champhai", "Mizoram"),

    # Delhi
    "DLCT": ("Central", "Delhi"),
    "DLND": ("New Delhi", "Delhi"),
    "DLNE": ("North-East", "Delhi"),
    "DLNW": ("North-West", "Delhi"),
    "DLET": ("East", "Delhi"),
    "DLWT": ("West", "Delhi"),
    "DLSE": ("South-East", "Delhi"),
    "DLSW": ("South-West", "Delhi"),
    "DLST": ("South", "Delhi"),
    "DLSH": ("Shahdara", "Delhi"),
    "DLNT": ("North", "Delhi"),

    # Maharashtra
    "MHAH": ("Ahmednagar", "Maharashtra"),
    "MHAK": ("Akola", "Maharashtra"),
    "MHAM": ("Amravati", "Maharashtra"),
    "MHAU": ("Aurangabad", "Maharashtra"),
    "MHBI": ("Beed", "Maharashtra"),
    "MHBH": ("Bhandara", "Maharashtra"),
    "MHBU": ("Buldhana", "Maharashtra"),
    "MHCH": ("Chandrapur", "Maharashtra"),
    "MHDH": ("Dhule", "Maharashtra"),
    "MHGA": ("Gadchiroli", "Maharashtra"),
    "MHGO": ("Gondia", "Maharashtra"),
    "MHHI": ("Hingoli", "Maharashtra"),
    "MHJG": ("Jalgaon", "Maharashtra"),
    "MHJN": ("Jalna", "Maharashtra"),
    "MHMC": ("Mumbai City", "Maharashtra"),
    "MHMU": ("Mumbai-Suburban", "Maharashtra"),
    "MHKO": ("Kolhapur", "Maharashtra"),
    "MHLA": ("Latur", "Maharashtra"),
    "MHNB": ("Nandurbar", "Maharashtra"),
    "MHND": ("Nanded", "Maharashtra"),
    "MHNG": ("Nagpur", "Maharashtra"),
    "MHNS": ("Nashik", "Maharashtra"),
    "MHOS": ("Osmanabad", "Maharashtra"),
    "MHPL": ("Palghar", "Maharashtra"),
    "MHPA": ("Parbhani", "Maharashtra"),
    "MHPU": ("Pune", "Maharashtra"),
    "MHRG": ("Raigad", "Maharashtra"),
    "MHRT": ("Ratnagiri", "Maharashtra"),
    "MHSI": ("Sindhudurg", "Maharashtra"),
    "MHSN": ("Sangli", "Maharashtra"),
    "MHST": ("Satara", "Maharashtra"),
    "MHSO": ("Solapur", "Maharashtra"),
    "MHTH": ("Thane", "Maharashtra"),
    "MHWR": ("Wardha", "Maharashtra"),
    "MHWS": ("Washim", "Maharashtra"),
    "MHYA": ("Yavatmal", "Maharashtra"),
    
    # Tamil Nadu
    "TNAL": ("Ariyalur", "Tamil Nadu"),
    "TNCG": ("Chengalpattu", "Tamil Nadu"),
    "TNCH": ("Chennai", "Tamil Nadu"),
    "TNCB": ("Coimbatore", "Tamil Nadu"),
    "TNCD": ("Cuddalore", "Tamil Nadu"),
    "TNDP": ("Dharmapuri", "Tamil Nadu"),
    "TNDG": ("Dindigul", "Tamil Nadu"),
    "TNED": ("Erode", "Tamil Nadu"),
    "TNKA": ("Kallakurichi", "Tamil Nadu"),
    "TNKP": ("Kancheepuram", "Tamil Nadu"),
    "TNKK": ("Kanyakumari", "Tamil Nadu"),
    "TNKR": ("Karur", "Tamil Nadu"),
    "TNKI": ("Krishnagiri", "Tamil Nadu"),
    "TNMD": ("Madurai", "Tamil Nadu"),
    "TNMY": ("Mayiladuthurai", "Tamil Nadu"),
    "TNNG": ("Nagapattinam", "Tamil Nadu"),
    "TNNM": ("Namakkal", "Tamil Nadu"),
    "TNNS": ("Nilgiris", "Tamil Nadu"),
    "TNPB": ("Perambalur", "Tamil Nadu"),
    "TNPD": ("Pudukkottai", "Tamil Nadu"),
    "TNRM": ("Ramanathapuram", "Tamil Nadu"),
    "TNRP": ("Ranipet", "Tamil Nadu"),
    "TNSA": ("Salem", "Tamil Nadu"),
    "TNSV": ("Sivaganga", "Tamil Nadu"),
    "TNTS": ("Tenkasi", "Tamil Nadu"),
    "TNTJ": ("Thanjavur", "Tamil Nadu"),
    "TNTH": ("Theni", "Tamil Nadu"),
    "TNTT": ("Thoothukudi", "Tamil Nadu"),
    "TNTP": ("Tiruchirappalli", "Tamil Nadu"),
    "TNTL": ("Tirunelveli", "Tamil Nadu"),
    "TNTU": ("Tirupathur", "Tamil Nadu"),
    "TNTI": ("Tiruppur", "Tamil Nadu"),
    "TNTR": ("Tiruvallur", "Tamil Nadu"),
    "TNTM": ("Tiruvannamalai", "Tamil Nadu"),
    "TNTV": ("Tiruvarur", "Tamil Nadu"),
    "TNVL": ("Vellore", "Tamil Nadu"),
    "TNVP": ("Viluppuram", "Tamil Nadu"),
    "TNVR": ("Virudhunagar", "Tamil Nadu"),

    # Telangana
    "TSAD": ("Adilabad", "Telangana"),
    "TSBK": ("Bhadradri Kothagudem", "Telangana"),
    "TSWR": ("Hanumakonda", "Telangana"),
    "TSJI": ("Jagtial", "Telangana"),
    "TSJN": ("Jangaon", "Telangana"),
    "TSJB": ("Jayashankar Bhupalapally", "Telangana"),
    "TSJG": ("Jogulamba Gadwal", "Telangana"),
    "TSKM": ("Kamareddy", "Telangana"),
    "TSKA": ("Karimnagar", "Telangana"),
    "TSKH": ("Khammam", "Telangana"),
    "TSKB": ("Kumuram Bheem Asifabad", "Telangana"),
    "TSMH": ("Mahabubabad", "Telangana"),
    "TSMA": ("Mahabubnagar", "Telangana"),
    "TSMN": ("Mancherial", "Telangana"),
    "TSME": ("Medak", "Telangana"),
    "TSMM": ("Medchal Malkajgiri", "Telangana"),
    "TSML": ("Mulugu", "Telangana"),
    "TSNG": ("Nagarkurnool", "Telangana"),
    "TSNA": ("Nalgonda", "Telangana"),
    "TSNP": ("Narayanpet", "Telangana"),
    "TSNR": ("Nirmal", "Telangana"),
    "TSNI": ("Nizamabad", "Telangana"),
    "TSPD": ("Peddapalli", "Telangana"),
    "TSRS": ("Rajanna Sircilla", "Telangana"),
    "TSRA": ("Rangareddy", "Telangana"),
    "TSSN": ("Sangareddy", "Telangana"),
    "TSSD": ("Siddipet", "Telangana"),
    "TSSR": ("Suryapet", "Telangana"),
    "TSVK": ("Vikarabad", "Telangana"),
    "TSWN": ("Wanaparthy", "Telangana"),
    "TSWL": ("Warangal", "Telangana"),
    "TSYB": ("Yadadri Bhuvanagiri", "Telangana"),
    # Hyderabad — Multiple Separate Court Units
    "TSHB": ("Hyderabad - City Civil Court", "Telangana"),
    "TSSC": ("Hyderabad - City Small Cause Court", "Telangana"),
    "TSMS": ("Hyderabad - Metropolitan Sessions Court", "Telangana"),
    "TSCB": ("Hyderabad - Prl. CBI Court", "Telangana"),
    "TSIF": ("Hyderabad - Integrated Family Courts", "Telangana"),
    "TSFC": ("Hyderabad - Family Court", "Telangana"),

    # Karnataka
    "KABK": ("Bagalkote", "Karnataka"),
    "KABI": ("Ballari", "Karnataka"),
    "KABG": ("Belagavi", "Karnataka"),
    "KABC": ("Bengaluru Urban", "Karnataka"),
    "KABR": ("Bengaluru Rural", "Karnataka"),
    "KABD": ("Bidar", "Karnataka"),
    "KACN": ("Chamarajanagara", "Karnataka"),
    "KACB": ("Chikkaballapura ", "Karnataka"),
    "KACM": ("Chikkamagaluru", "Karnataka"),
    "KACD": ("Chitradurga", "Karnataka"),
    "KADK": ("Dakshina Kannada", "Karnataka"),
    "KADG": ("Davanagere", "Karnataka"),
    "KADW": ("Dharwad", "Karnataka"),
    "KAGD": ("Gadag", "Karnataka"),
    "KAHS": ("Hassan", "Karnataka"),
    "KAHV": ("Haveri", "Karnataka"),
    "KA32": ("Kalaburagi", "Karnataka"),
    "KAKD": ("Kodagu", "Karnataka"),
    "KAKL": ("Kolar", "Karnataka"),
    "KAKP": ("Koppal", "Karnataka"),
    "KAMD": ("Mandya", "Karnataka"),
    "KAMS": ("Mysuru", "Karnataka"),
    "KARC": ("Raichur", "Karnataka"),
    "KARN": ("Ramanagara", "Karnataka"),
    "KA14": ("Shivamogga", "Karnataka"),
    "KATK": ("Tumakuru", "Karnataka"),
    "KAUP": ("Udupi", "Karnataka"),
    "KAUK": ("Uttara Kannada", "Karnataka"),
    "KAVP": ("Vijayapura", "Karnataka"),
    "KAVN": ("Vijayanagara", "Karnataka"),
    "KAYG": ("Yadgir", "Karnataka"),

    # Andaman and Nicobar
    "ANPB": ("Port Blair", "Andaman & Nicobar Islands"),

    #The Dadra And Nagar Haveli And Daman And Diu
   "UTDD": ("Daman/Diu", "Dadra & Nagar Haveli & Daman & Diu"),
   "UTDN": ("Dadra & Nagar Haveli(Silvassa)", "Dadra & Nagar Haveli & Daman & Diu"),

   # Puducherry
   "PYPY": ("Puducherry", "Puducherry"),
   "PYKL": ("Karaikal", "Puducherry"),
   "PYME": ("Mahe", "Puducherry"),
   "PYYM": ("Yanam", "Puducherry"),

   # Lakshadweep
   "KLLD": ("Lakshadweep (Kavaratti)", "Lakhsadweep"),

   #Goa
   "GANG": ("North Goa (Panaji)", "Goa"),
   "GASG": ("South Goa (Margao)", "Goa"),

   #Ladakh
   "LDKR": ("Kargil", "Ladakh"),
   "LDLH": ("Leh", "Ladakh"),

   
    # West Bengal
    "WBJP": ("Jalpaiguri/Alipurduar(ambiguous)", "West Bengal"),
    "WBBK": ("Bankura", "West Bengal"),
    "WBBB": ("Birbhum", "West Bengal"),
    "WBCC": ("Kolkata (Calcutta)", "West Bengal"),
    "WBCB": ("Cooch Behar", "West Bengal"),
    "WBDJ": ("Darjeeling/kalimpong(ambiguous)", "West Bengal"),
    "WBHG": ("Hooghly", "West Bengal"),
    "WBHW": ("Howrah", "West Bengal"),
    "WBWM": ("Jhargram/Paschim Medinipur(ambiguous)", "West Bengal"),
    "WBML": ("Malda", "West Bengal"),
    "WBMD": ("MUrshidabad", "West Bengal"),
    "WBND": ("Nadia", "West Bengal"),
    "WBNP": ("North 24 Parganas", "West Bengal"),
    "WBUD": ("North Dinajpur", "West Bengal"),
    "WBBD": ("Paschim Bardhaman/Purba Bardhaman(ambiguous)", "West Bengal"),
    "WBEM": ("Purba Medinipur", "West Bengal"),
    "WBPU": ("Purulia", "West Bengal"),
    "WBSP": ("South 24 Parganas", "West Bengal"),
    "WBDD": ("South Dinajpur", "West Bengal"),

    # Himachal Pradesh
    "HPBI": ("Bilaspur", "Himachal Pradesh"),
    "HPCH": ("Chamba", "Himachal Pradesh"),
    "HPHA": ("Hamirpur", "Himachal Pradesh"),
    "HPKA": ("Kangra", "Himachal Pradesh"),
    "HPKI": ("Kinnaur", "Himachal Pradesh"),
    "HPKU": ("Kullu", "Himachal Pradesh"),
    "HPMA": ("Mandi", "Himachal Pradesh"),
    "HPSH": ("Shimla", "Himachal Pradesh"),
    "HPSR": ("Sirmaur", "Himachal Pradesh"),
    "HPSO": ("Solan", "Himachal Pradesh"),
    "HPUN": ("Una", "Himachal Pradesh"),

     # Uttarakhand
    "UKAL": ("Almora", "Uttarakhand"),
    "UKBA": ("Bageshwar", "Uttarakhand"),
    "UKCL": ("Chamoli", "Uttarakhand"),
    "UKCP": ("Champawat", "Uttarakhand"),
    "UKDD": ("Dehradun", "Uttarakhand"),
    "UKHA": ("Haridwar", "Uttarakhand"),
    "UKNA": ("Nainital", "Uttarakhand"),
    "UKPG": ("Pauri Garhwal", "Uttarakhand"),
    "UKPI": ("Pithoragarh", "Uttarakhand"),
    "UKUS": ("Udham Singh Nagar", "Uttarakhand"),

    # Bihar
    "BRPT": ("Patna", "Bihar"),
    # Uttar Pradesh
    "UPLK": ("Lucknow", "Uttar Pradesh"),
    "UPLB": ("Prayagraj", "Uttar Pradesh"),
    # Rajasthan
    "RJJP": ("Jaipur", "Rajasthan"),
    # Gujarat
    "GJAH": ("Ahmedabad", "Gujarat"),
    "GJSR": ("Surat", "Gujarat"),
    "GJGN": ("Gandhinagar", "Gujarat"),
    # Kerala
    "KLKT": ("Kottayam", "Kerala"),
    "KLTV": ("Thiruvananthapuram", "Kerala"),
    # Madhya Pradesh
    "MPBP": ("Bhopal", "Madhya Pradesh"),
    "MPIN": ("Indore", "Madhya Pradesh"),
    # Chhattisgarh
    "CGRP": ("Raipur", "Chhattisgarh"),
    # Jharkhand
    "JHRN": ("Ranchi", "Jharkhand"),
    # Odisha
    "ODBB": ("Bhubaneswar", "Odisha"),
    "ODCT": ("Cuttack", "Odisha"),
    # Punjab
    "PBLD": ("Ludhiana", "Punjab"),
    "PBAS": ("Amritsar", "Punjab"),
    "PBHO": ("Hoshiarpur", "Punjab"),
    "PBPO": ("Pathankot", "Punjab"),
    "PBBT": ("Bathinda", "Punjab"),
    "PBRO": ("Rupnagar", "Punjab"),

    # Haryana
    "HRGR": ("Gurugram", "Haryana"),
    
   
    # Jammu & Kashmir
    "JKJM": ("Jammu", "Jammu & Kashmir"),
    "JKSR": ("Srinagar", "Jammu & Kashmir"),
    
}

CNR_STATE_MAP = {
    "TR":"Tripura","SK":"Sikkim","AS":"Assam","MN":"Manipur",
    "ML":"Meghalaya","MZ":"Mizoram","NL":"Nagaland","AR":"Arunachal Pradesh",
    "DL":"Delhi","MH":"Maharashtra","TN":"Tamil Nadu","TS":"Telangana",
    "KA":"Karnataka","WB":"West Bengal","BR":"Bihar","UP":"Uttar Pradesh",
    "RJ":"Rajasthan","PB":"Punjab","HR":"Haryana","MP":"Madhya Pradesh",
    "CG":"Chhattisgarh","JH":"Jharkhand","OD":"Odisha","GJ":"Gujarat",
    "KL":"Kerala","UK":"Uttarakhand","HP":"Himachal Pradesh","KL":"Lakshadweep",
    "JK":"Jammu & Kashmir","GA":"Goa","AN":"Andaman & Nicobar Islands",
    "UT":"Dadra & Nagar Haveli & Daman & Diu","LD":"Ladakh",
    "PY":"Puducherry","CH":"Chandigarh",
}

COURT_NAME_MAP = {
    "bishalgarh":("Sepahijala (Bishalgarh)","Tripura"),
    "agartala":("West Tripura","Tripura"),
    "namchi":("Namchi (South Sikkim)","Sikkim"),
    "gangtok":("East Sikkim","Sikkim"),
    "jorethang":("South Sikkim","Sikkim"),
    "kohima":("Kohima","Nagaland"),
    "imphal":("Imphal","Manipur"),
    "shillong":("East Khasi Hills","Meghalaya"),
    "aizawl":("Aizawl","Mizoram"),
    "itanagar":("Papum Pare","Arunachal Pradesh"),
    "guwahati":("Kamrup Metro","Assam"),
    "dibrugarh":("Dibrugarh","Assam"),
    "silchar":("Cachar","Assam"),
    "delhi":("New Delhi","Delhi"),
    "mumbai":("Mumbai","Maharashtra"),
    "pune":("Pune","Maharashtra"),
    "chennai":("Chennai","Tamil Nadu"),
    "hyderabad":("Hyderabad","Telangana"),
    "bengaluru":("Bengaluru","Karnataka"),
    "bangalore":("Bengaluru","Karnataka"),
    "kolkata":("Kolkata","West Bengal"),
    "patna":("Patna","Bihar"),
    "lucknow":("Lucknow","Uttar Pradesh"),
    "allahabad":("Prayagraj","Uttar Pradesh"),
    "jaipur":("Jaipur","Rajasthan"),
    "chandigarh":("Chandigarh","Punjab & Haryana"),
    "bhopal":("Bhopal","Madhya Pradesh"),
    "indore":("Indore","Madhya Pradesh"),
    "raipur":("Raipur","Chhattisgarh"),
    "ranchi":("Ranchi","Jharkhand"),
    "bhubaneswar":("Bhubaneswar","Odisha"),
    "cuttack":("Cuttack","Odisha"),
    "ahmedabad":("Ahmedabad","Gujarat"),
    "surat":("Surat","Gujarat"),
    "kochi":("Ernakulam","Kerala"),
    "thiruvananthapuram":("Thiruvananthapuram","Kerala"),
    "dehradun":("Dehradun","Uttarakhand"),
    "shimla":("Shimla","Himachal Pradesh"),
    "jammu":("Jammu","Jammu & Kashmir"),
    "srinagar":("Srinagar","Jammu & Kashmir"),
    "panaji":("North Goa","Goa"),
    "puducherry":("Puducherry","Puducherry"),
    "pondicherry":("Puducherry","Puducherry"),
}


def detect_location(court_name: str, cnr: str):
    # 1. CNR 4-char prefix → district + state
    if cnr and len(cnr) >= 4:
        key = cnr[:4].upper()
        if key in CNR_DISTRICT_MAP:
            return CNR_DISTRICT_MAP[key]
    # 2. Court name keyword
    lower = court_name.lower()
    for keyword, (district, state) in COURT_NAME_MAP.items():
        if keyword in lower:
            return district, state
    # 3. CNR 2-char state only
    if cnr and len(cnr) >= 2:
        state = CNR_STATE_MAP.get(cnr[:2].upper(), "Unknown")
        return "Unknown", state
    return "Unknown", "Unknown"


# ─────────────────────────────────────────────
# Acts & Sections extractor  ← FIXED
# ─────────────────────────────────────────────

def extract_acts_sections(raw_rows: list):
    """
    Scans raw table rows for the Acts table which looks like:
        Row: ['Under Act(s)', 'Under Section(s)']   ← header  (2-col layout)
        Row: ['Limitation Act', '5']                 ← data

        OR (6-col layout with blank columns):
        Row: ['Under Act(s)', None, 'Under Section(s)', '', '', None]
        Row: ['Bombay Regulation Act,1827', None, 'VIII', '', '', None]

    Detects act_col and sec_col dynamically from the header row.
    """
    act_names = []
    sections  = []
    in_acts   = False
    act_col   = 0   # default: col 0 = act
    sec_col   = 1   # default: col 1 = section

    for row in raw_rows:
        # Normalise row cells (strip newlines)
        r = [clean(c) for c in row]

        # Detect the header row
        joined = " ".join(r).lower()
        if "under act" in joined and "under section" in joined:
            in_acts = True
            # Find which column index holds "under act(s)" and "under section(s)"
            for i, c in enumerate(r):
                if c and "under act" in c.lower():
                    act_col = i
                if c and "under section" in c.lower():
                    sec_col = i
            continue

        if not in_acts:
            continue

        # Stop when we hit another section
        if any(k in joined for k in [
            "fir details", "case history", "petitioner",
            "processes", "process id", "process title",
            "interim order", "police station",
            "field", "details", "order number", "judge"
        ]):
            break

        # Skip empty rows
        non_empty = [c for c in r if c]
        if not non_empty:
            continue

        act_val = r[act_col] if len(r) > act_col else ""
        sec_val = r[sec_col] if len(r) > sec_col else ""

        if act_val and act_val.lower() not in ("under act(s)", ""):
            act_names.append(act_val)
        if sec_val and sec_val.lower() not in ("under section(s)", ""):
            sections.append(sec_val)

    act_name     = ", ".join(act_names) if act_names else None
    section_str  = ", ".join(sections)  if sections  else None
    # Count both numeric (166) and Roman-numeral (VIII) sections
    num_sections = len(sections) if sections else 0

    return act_name, section_str, num_sections


# ─────────────────────────────────────────────
# Case History date extractor  ← FIXED
# ─────────────────────────────────────────────

def extract_history_dates(raw_rows: list):
    """
    Scans raw table rows for Case History tables.

    Standard 4-col layout:  ['Judge', 'Business on Date', 'Hearing Date', 'Purpose']
    Extended 6-col layout:  ['', 'Judge', 'Business on Date', 'Hearing Date', 'Purpose', '']
    (Some courts add empty border columns — seen in Manipur PDFs)

    Fix: detect col positions dynamically from the header row so both layouts work.
    """
    business_dates = []
    hearing_dates  = []
    in_history     = False
    biz_col        = 1    # default column positions
    hear_col       = 2

    for row in raw_rows:
        r = [clean(c) for c in row]
        joined_lower = " ".join(r).lower()

        # ── Detect history header row ─────────────────────────────────
        if ("business on" in joined_lower or "business" in joined_lower)                 and "hearing" in joined_lower                 and "judge" in joined_lower:
            in_history = True

            # Dynamically find which column contains "business" and "hearing"
            # so we handle both 4-col and 6-col table layouts
            biz_col  = next((i for i, c in enumerate(r) if "business" in c.lower()), 1)
            hear_col = next((i for i, c in enumerate(r) if "hearing" in c.lower()
                             and "purpose" not in c.lower()), biz_col + 1)
            continue

        if not in_history:
            continue

        # ── Stop at orders tables ─────────────────────────────────────
        if any(k in joined_lower for k in [
            "order number", "interim order", "final order",
            "about us", "order date", "order details"
        ]):
            in_history = False   # reset — may be another history table on next page
            continue

        # ── Skip empty rows ───────────────────────────────────────────
        if not any(c for c in r if c):
            continue

        # ── Extract dates using detected column positions ─────────────
        biz_raw  = cell(row[biz_col])  if len(row) > biz_col  else ""
        hear_raw = cell(row[hear_col]) if len(row) > hear_col else ""

        biz_date  = normalise_date(biz_raw)
        hear_date = normalise_date(hear_raw)

        if biz_date:
            business_dates.append(biz_date)
        if hear_date:
            hearing_dates.append(hear_date)

    return sorted(set(business_dates)), sorted(set(hearing_dates))


# ─────────────────────────────────────────────
# Orders extractor (Interim + Final)
# ─────────────────────────────────────────────

def extract_orders(raw_rows: list, full_text: str = ""):
    """
    Extracts interim order dates as a flat list.

    Handles three layouts produced by pdfplumber on eCourts PDFs:

    Layout A — standard 3-col table (most pages):
        ['Order Number', 'Order Date', 'Order Details']
        ['1', '08-03-2024', 'ORDER SHEET']

    Layout B — extended 5-col table with empty border columns (e.g. page 9 of
                metadata7.pdf — Sikkim PDFs pad with empty first/last columns):
        ['', 'Order Number', 'Order Date', 'Order Details', '']
        ['', '14', '04-12-2024', 'ORDER SHEET', '']
        Previously broken: code read row[1] for the date, which is the order
        NUMBER in this layout, so all rows were silently skipped.

    Layout C — page-break blob (pdfplumber merges the last page's tables into
                one giant single-cell row containing ALL order text including
                "Final Orders / Judgements" heading and final order data):
        ['Order Number ... 32 13-02-2026 ORDER SHEET Final Orders / Judgements ...']
        These rows are skipped; actual data comes from the separate Table 2/3.

    Final-orders boundary:
        Plain text (page.extract_text) always preserves the
        "Final Orders / Judgements" heading even when the table extractor
        drops it (Layout C, metadata6/7).  We pre-scan full_text once to
        find the FIRST order number that follows the heading — that number
        becomes `final_start_num`.  In the main loop, as soon as we see a
        data row whose order_num == final_start_num we stop — regardless of
        which layout the row comes from, and without being fooled by subsequent
        header rows that would otherwise re-enable in_interim.
    """
    interim_orders  = []
    in_interim      = False

    # Column positions — updated dynamically from each header row so that
    # both Layout A (num_col=0, date_col=1) and Layout B (num_col=1, date_col=2)
    # are handled correctly.
    num_col  = 0
    date_col = 1

    # ── Pre-scan: find first final-order number from plain text ──────────
    # Works for all layouts because extract_text() always has the heading line.
    final_start_num: str | None = None
    if full_text:
        m = re.search(
            r'final\s+orders?\s*/?\s*judgements?'   # section heading
            r'(?:[^\d]{0,150}?)'                     # skip column header + icons
            r'(?<!\d)(\d{1,3})(?!\d)',               # first standalone 1-3 digit number
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            final_start_num = m.group(1)

    for row in raw_rows:
        r            = [clean(c) for c in row]
        joined_lower = " ".join(r).lower()
        non_empty    = [c for c in r if c]

        # ── Skip blob rows ───────────────────────────────────────────────
        # A blob is a single-cell row that contains embedded dates — it is an
        # artefact of pdfplumber collapsing multiple tables into one cell.
        # Real data for those orders comes from the proper Table 2 / Table 3
        # on the same page, so we simply skip the blob.
        if (
            len(non_empty) == 1
            and re.search(r'\d{2}-\d{2}-\d{4}', joined_lower)
        ):
            continue

        # ── Footer rows — skip without stopping ─────────────────────────
        # The footer blob (About Us / Newsletter / Disclaimer …) appears as a
        # merged single-cell row inside Table 1 on the LAST page, BEFORE
        # Tables 2 and 3 which contain the remaining real order data.
        # Using `break` here would stop us from ever seeing those tables.
        # `continue` safely skips the noise while letting the loop reach them.
        if any(k in joined_lower for k in [
            "about us", "newsletter", "disclaimer",
            "site map", "contact us", "help videos",
            "hyperlinking policy", "screen reader",
        ]):
            continue

         # ── Stop at Case Transfer Details section ────────────────────────
        # The "Case Transfer Details within Establishment" table appears right
        # after Interim Orders and contains a "Transfer Date" column whose
        # dates would otherwise be misread as interim order dates.
        if any(k in joined_lower for k in [
            "transfer date", "case transfer", "within establishment",
        ]):
            in_interim = False
            continue

        # ── Standalone "Interim Orders" title row ────────────────────────
        if (
            "interim order" in joined_lower
            and "order number" not in joined_lower
            and "order date" not in joined_lower
        ):
            in_interim = True
            continue

        # ── Column header row — update column positions dynamically ───────
        # This fires for EVERY header row, including the second one that
        # belongs to the Final Orders table; but `final_start_num` prevents
        # final order rows from being added to interim_orders.
        if "order number" in joined_lower and "order date" in joined_lower:
            in_interim = True
            num_col  = next(
                (i for i, c in enumerate(r) if "order number" in c.lower()), 0)
            date_col = next(
                (i for i, c in enumerate(r) if "order date"   in c.lower()), num_col + 1)
            continue

        if not in_interim:
            continue

        # Skip blank rows
        if not any(c for c in r if c):
            continue

        # ── Parse the data row using detected column positions ────────────
        order_num  = r[num_col]                          if len(r)   > num_col  else ""
        order_date = normalise_date(cell(row[date_col])) if len(row) > date_col else None

        if not order_num or not order_date:
            continue

        # ── Final-orders boundary: stop before first final order ──────────
        if final_start_num and order_num == final_start_num:
            break

        interim_orders.append(order_date)

    return interim_orders if interim_orders else None

# ─────────────────────────────────────────────
# Hearing purpose extractor
# ─────────────────────────────────────────────

def extract_hearing_purposes(raw_rows: list) -> list:
    """
    Scans Case History rows and builds a list of
    "DD-MM-YYYY: <Purpose of Hearing>" strings, keyed on Hearing Date.

    Case History table columns (standard 4-col layout):
        Judge (0) | Business on Date (1) | Hearing Date (2) | Purpose of Hearing (3)

    The key is Hearing Date (col 2) — the date the hearing is actually
    scheduled to occur — NOT Business on Date (col 1), which is only
    the date the case was last called to schedule the next hearing.

    Example output:
        ["05-08-2026: Appereance", "01-04-2026: Appereance", "06-02-2026: SR", ...]

    Handles both the standard 4-col layout and extended 6-col layout.
    Also captures single-cell "Disposed" rows that appear in some PDFs.
    """
    purposes:   list = []
    in_history: bool = False
    biz_col:    int  = 1   # Business on Date — used only as fallback
    hear_col:   int  = 2   # Hearing Date     — primary key for output
    purp_col:   int  = 3   # Purpose of Hearing

    for row in raw_rows:
        r            = [clean(c) for c in row]
        joined_lower = " ".join(r).lower()

        # ── Detect history header ─────────────────────────────────────
        if (
            ("business on" in joined_lower or "business" in joined_lower)
            and "hearing" in joined_lower
            and "judge" in joined_lower
        ):
            in_history = True
            biz_col  = next((i for i, c in enumerate(r)
                             if "business" in c.lower()), 1)
            # "Hearing Date" column — skip cells that also contain "purpose"
            hear_col = next((i for i, c in enumerate(r)
                             if "hearing" in c.lower()
                             and "purpose" not in c.lower()), biz_col + 1)
            purp_col = next((i for i, c in enumerate(r)
                             if "purpose" in c.lower()), hear_col + 1)
            continue

        if not in_history:
            continue

        # ── Stop at orders section ────────────────────────────────────
        if any(k in joined_lower for k in [
            "order number", "interim order", "final order",
            "about us", "order date", "order details",
        ]):
            in_history = False
            continue

        # Skip blank rows
        if not any(c for c in r if c):
            continue

        # ── Extract hearing date + purpose ────────────────────────────
        # Key on Hearing Date (when the hearing actually takes place).
        # Fall back to Business on Date only if Hearing Date is absent.
        hear_raw = cell(row[hear_col]) if len(row) > hear_col else ""
        purp_raw = r[purp_col]         if len(r)   > purp_col else ""
        hear_date = normalise_date(hear_raw)

        if not hear_date:
            # Fallback: try Business on Date
            biz_raw   = cell(row[biz_col]) if len(row) > biz_col else ""
            hear_date = normalise_date(biz_raw)

        if hear_date:
            if purp_raw:
                purposes.append(f"{hear_date}: {purp_raw}")
            else:
                # Some PDFs show a lone "Disposed" cell in a merged row
                for cell_val in r:
                    if cell_val and cell_val.lower() in (
                            "disposed", "disposal", "decided"):
                        purposes.append(f"{hear_date}: {cell_val}")
                        break

    return purposes


# ─────────────────────────────────────────────
# Core parser
# ─────────────────────────────────────────────

def parse_case(text: str, raw_rows: list) -> dict:

    # ── Basic identifiers — scan table rows directly ──────────────────
    cnr        = None
    case_type  = None
    filing_num = None
    reg_num    = None
    filing_date_raw   = None
    reg_date_raw      = None

    # ── Helpers for flexible column-position-independent row parsing ──────
    def _label_idx(r, *keywords):
        """Return index of first cell containing ALL keywords (case-insensitive)."""
        for i, c in enumerate(r):
            if c and all(kw in c.lower() for kw in keywords):
                return i
        return -1

    def _next_val(r, label_i, raw_row=None):
        """First non-empty cell to the RIGHT of label_i.
        Uses raw_row (un-cleaned) when provided so cell() can strip internal \n."""
        src = raw_row if raw_row is not None else r
        for j in range(label_i + 1, len(src)):
            v = cell(src[j]) if raw_row is not None else r[j]
            if v:
                return v
        return ""
    # ─────────────────────────────────────────────────────────────────────

    for row in raw_rows:
        r = [clean(c) for c in row]
        joined = " ".join(r)

        # Case Type — label may be at any position
        if not case_type:
            li = _label_idx(r, "case type")
            if li >= 0:
                v = _next_val(r, li)
                if v and v.lower() != "case type":
                    case_type = v

        # Filing Number + Filing Date — both can be anywhere in the row
        if "filing" in joined.lower() and "number" in joined.lower():
            if not filing_num:
                li = _label_idx(r, "filing", "number")
                if li >= 0:
                    v = _next_val(r, li)
                    if v and re.match(r'\d', v):
                        filing_num = v
            if not filing_date_raw:
                li = _label_idx(r, "filing", "date")
                if li < 0:                       # some PDFs just say "Filing Date"
                    li = _label_idx(r, "filing date")
                if li >= 0:
                    filing_date_raw = _next_val(r, li, row)

        # Registration Number + Registration Date
        if "registration" in joined.lower():
            if not reg_num:
                li = _label_idx(r, "registration", "number")
                if li >= 0:
                    v = _next_val(r, li)
                    if v and re.match(r'\d', v):
                        reg_num = v
            if not reg_date_raw:
                li = _label_idx(r, "registration", "date")
                if li >= 0:
                    reg_date_raw = _next_val(r, li, row)

        # CNR Number — label may be at any column; value is the next non-empty cell
        if not cnr and "cnr" in joined.lower():
            li = _label_idx(r, "cnr")
            if li >= 0:
                # Value cell may itself contain extra text "(Note the CNR…)"
                raw_val = _next_val(r, li)
                # CNR numbers are strictly uppercase: 2-4 uppercase letters + 8-14 digits/uppercase
                m = re.search(r'\b([A-Z]{2,4}[A-Z0-9]{8,16})\b', raw_val)
                if m:
                    cnr = m.group(1)

    # Fallback to regex on text if still missing
    if not cnr:
        # Match uppercase-only CNR format to avoid catching mixed-case words like "Navigation"
        m = re.search(r'\b([A-Z]{2,4}[0-9]{2}[A-Z0-9]{6,14})\b', text)
        cnr = m.group(1) if m else None
    if not case_type:
        case_type = find_field(r'Case\s*Type\s*[:\-]?\s*(.+?)(?:\n|Filing)', text)
    if not filing_num:
        filing_num = find_field(r'Filing\s*Number\s*[:\-]?\s*([\d/]+)', text)
    if not reg_num:
        reg_num = find_field(r'Registration\s*Number\s*[:\-]?\s*([\d/]+)', text)

    filing_date = normalise_date(filing_date_raw)
    reg_date    = normalise_date(reg_date_raw)

    # ── Court name  (3-tier extraction) ──────────────────────────────
    #
    # Tier 1 — Structural anchor (most reliable):
    #   eCourts always renders:  "Back\n<Court Name>\nCase Details"
    #   Find the line immediately BEFORE "Case Details" in the raw text.
    #
    # Tier 2 — Keyword regex (catches courts whose name contains a
    #   known type-word even when the structural anchor is absent).
    #
    # Tier 3 — Filtered line-scan fallback (last resort).
    # ─────────────────────────────────────────────────────────────────

    # Shared helper: strip section-header suffixes that sometimes appear
    # on the same line as the court name in flat/image PDFs.
    def _strip_section_suffix(name: str) -> str:
        return re.split(
            r'\s+(?:Case\s+Details|Case\s+Type|Filing(?:\s+Number|\s+Date)?|'
            r'CNR(?:\s+Number)?|Registration(?:\s+Number)?|'
            r'Process\s+Details|Court\s+Number)\b',
            name, maxsplit=1, flags=re.IGNORECASE
        )[0].rstrip(" ,")

    court_name = None

    # ── Tier 1: line before "Case Details" ───────────────────────────
    # Works on both the plain-text block and the table-flattened block
    # because extract_text_from_pdf() appends page.extract_text() which
    # preserves the original line order from the portal.
    _cd_match = re.search(
        r'([^\n]{3,120})\s*\n\s*Case\s+Details\b',
        text, re.IGNORECASE
    )
    if _cd_match:
        candidate = clean(_cd_match.group(1))
        # Reject obvious UI-chrome lines (navigation bar, Back button, etc.)
        _bad_starts = (
            "back", "e-committee", "skip to", "search menu", "download",
            "site map", "location", "language", "a-", "a+",
        )
        _bad_re = re.compile(r'^(?:[Aa]\s*[-+]|[©®]|S\d+\s+©|\d{1,2}[-/]\d{1,2}[-/]\d{4})')
        if (len(candidate) > 3
                and not candidate.lower().startswith(_bad_starts)
                and not _bad_re.match(candidate)):
            court_name = _strip_section_suffix(candidate)

    # ── Tier 2: keyword regex ─────────────────────────────────────────
    if not court_name:
        _court_pat = re.compile(
            r'(?:Establishment\s+of\s+)?'
            r'((?:Additional\s+|Addl\.?\s+|Principal\s+)?'
            r'(?:District\s+and\s+Sessions?\s+(?:Court|Judge)|'
            r'District\s+(?:Court|Judge)|'
            r'Sessions?\s+(?:Court|Judge)|'
            r'Chief\s+Judicial\s+Magistrate|'
            r'Judicial\s+Magistrate|'
            r'High\s+Court|'
            r'Metropolitan\s+Magistrate|'
            r'Motor\s+Accident\s+Claims?\s+Tribunal|'
            r'Civil\s+Judge|'
            r'Family\s+Court|'
            r'Labour\s+Court|'
            r'Consumer\s+(?:Disputes?\s+)?(?:Redressal\s+)?(?:Forum|Commission))'
            r'[^\n]{0,80}'
            r'(?:\n[^\n]{1,80})?)',
            re.IGNORECASE,
        )
        _m = _court_pat.search(text)
        if _m:
            court_name = _strip_section_suffix(clean(_m.group(1)))

    # ── Tier 3: filtered line-scan ────────────────────────────────────
    if not court_name:
        _ui_noise = (
            "back", "download", "case", "cnr", "filing", "about",
            "registration", "under", "court number",
            "e-committee", "skip to", "search menu", "language",
            "process details", "location", "cause list", "caveat",
            "site map", "newsletter", "forms for", "help videos",
            "copyright", "hyperlinking", "accessibility", "disclaimer",
            "this site", "national informatics", "content reviewed",
            "terms and", "privacy policy", "all rights",
            # eCourts accessibility / font-size controls
            "a- a", "a-a", "a+ a", "a +", "a-", "a+",
            # Case-detail field labels that must never be mistaken for a court
            "e-filing", "first hearing", "decision date", "case status",
            "nature of disposal", "petitioner", "respondent", "advocate",
            "interim order", "final order", "order number",
        )
        _ui_noise_re = re.compile(
            r'^(?:'
            r'[Aa][-+]\s*[Aa]'              # "A- A …" font controls
            r'|[Aa]\s*[-+]'                  # bare "A-" / "A+"
            r'|\d{1,2}[-/]\d{1,2}[-/]\d{4}' # stray date
            r'|[©®]'                          # copyright symbol
            r'|S\d+\s+©'                      # "S6 © 2022…"
            r'|\d+\)'                          # "1) …" party list
            r')'
        )
        for line in text.splitlines():
            line = clean(line)
            if (len(line) > 15
                    and not line.lower().startswith(_ui_noise)
                    and not _ui_noise_re.match(line)):
                court_name = _strip_section_suffix(line)
                break

    court_name  = court_name.strip(" ,") if court_name else "Unknown"
    court_level = detect_court_level(court_name)
    district, state = detect_location(court_name, cnr or "")

    # ── Acts & Sections ───────────────────────────────────────────────
    act_name, section_str, num_sections = extract_acts_sections(raw_rows)

    # ── Status dates — scan Case Status table ─────────────────────────
    first_hearing  = None
    decision_date  = None
    next_hearing   = None

    for row in raw_rows:
        r = [clean(c) for c in row]
        if not r:
            continue

        # Check both (col0→col1) and (col2→col3) label/value pairs
        pairs = []
        if len(r) >= 2:
            pairs.append((r[0].lower(), r[1]))
        if len(r) >= 4:
            pairs.append((r[2].lower(), r[3]))

        for label, value in pairs:
            if not value:
                continue
            d = normalise_date(value)
            if not d:
                continue

            if not first_hearing and any(k in label for k in [
                    "first hearing", "first date"]):
                first_hearing = d

            if not decision_date and any(k in label for k in [
                    "decision", "decided", "disposal date", "date of decision"]):
                decision_date = d

            if not next_hearing and any(k in label for k in [
                    "next hearing", "next date", "next date of hearing",
                    "adjourned", "next"]):
                next_hearing = d

    # ── Regex fallback on full text if table scan missed anything ─────
    DATE_FMTS = (
        r'[\d]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}'
        r'|[\d]{1,2}[-/][\d]{1,2}[-/][\d]{4}'
    )
    if not first_hearing:
        first_hearing = normalise_date(
            find_field(rf'First\s*Hearing\s*Date\s*[:\-]?\s*({DATE_FMTS})', text))

    if not decision_date:
        decision_date = normalise_date(
            find_field(rf'Decision\s*Date\s*[:\-]?\s*({DATE_FMTS})', text))

    if not next_hearing:
        next_hearing = normalise_date(
            find_field(
                rf'Next\s*(?:Hearing\s*)?(?:Date\s*(?:of\s*Hearing)?)?\s*[:\-]?\s*({DATE_FMTS})',
                text))

    if not next_hearing:
        next_hearing = normalise_date(
            find_field(
                rf'(?:Next\s+Date|Adjourned\s+(?:to|on))\s*[:\-]?\s*({DATE_FMTS})',
                text))

    # ── Hearing & Business dates from Case History ────────────────────
    business_dates, hearing_dates = extract_history_dates(raw_rows)

    # ── Status ────────────────────────────────────────────────────────
    disposed_keywords = ["disposed", "acquitted", "convicted", "dismissed",
                         "allowed", "withdrawn", "settled", "decided"]
    is_disposed = int(any(k in text.lower() for k in disposed_keywords))
    is_pending  = 1 - is_disposed

    if is_disposed:
        next_hearing  = None
    if is_pending:
        decision_date = None

    # ── Orders (interim always; final only for disposed cases) ────────
    interim_orders = extract_orders(raw_rows, text)
    # ── Purpose of hearing keyed on business date ─────────────────────
    hearing_purposes = extract_hearing_purposes(raw_rows)

    return {
        "cnr_number":          cnr,
        "case_type":           case_type,
        "filing_number":       filing_num,
        "registration_number": reg_num,
        "court_name":          court_name,
        "court_level":         court_level,
        "district":            district,
        "state":               state,
        "act_name":            act_name,
        "section":             section_str,
        "number_of_sections":  num_sections,
        "filing_date":         filing_date,
        "hearing_dates":       hearing_dates,
        "business_dates":      business_dates,
        "registration_date":   reg_date,
        "first_hearing_date":  first_hearing,
        "decision_date":       decision_date,
        "next_hearing_date":   next_hearing,
        "is_pending":          is_pending,
        "is_disposed":         is_disposed,
        "interim_orders":      interim_orders,
        "hearing_purposes":    hearing_purposes,
    }


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

def process_folder(input_dir: str, output_file: str):
    pdf_files = sorted(Path(input_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No PDF files found in: {input_dir}")
        return

    results = []
    for pdf_path in pdf_files:
        print(f"[+] Processing: {pdf_path.name}")
        try:
            text     = extract_text_from_pdf(str(pdf_path))
            raw_rows = extract_tables_raw(str(pdf_path))
            data     = parse_case(text, raw_rows)
            data["source_file"] = pdf_path.name
            results.append(data)
            print(f"    [OK] CNR: {data.get('cnr_number')} | "
                  f"District: {data.get('district')} | "
                  f"Act: {data.get('act_name')} | "
                  f"Status: {'Pending' if data['is_pending'] else 'Disposed'}")
        except Exception as e:
            import traceback
            print(f"    [ERROR] {pdf_path.name}: {e}")
            traceback.print_exc()
            results.append({"source_file": pdf_path.name, "error": str(e)})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. {len(results)} case(s) written to: {output_file}")


# ─────────────────────────────────────────────
# Entry point — works in Jupyter AND Terminal
# ─────────────────────────────────────────────

if __name__ == "__main__":

    if any("ipykernel" in arg or "jupyter" in arg for arg in sys.argv):
        # ✏️ Change these two paths to match your setup
        input_dir   = "./pdfs"               # ← folder with your PDFs
        output_file = "cases_metadata.json"  # ← output file
        process_folder(input_dir, output_file)

    else:
        parser = argparse.ArgumentParser(
            description="Extract court case metadata from eCourts PDFs"
        )
        parser.add_argument("--input",  "-i", default="./pdfs",
                            help="Folder containing PDF files (default: ./pdfs)")
        parser.add_argument("--output", "-o", default="cases_metadata.json",
                            help="Output JSON file (default: cases_metadata.json)")
        args = parser.parse_args()
        process_folder(args.input, args.output)
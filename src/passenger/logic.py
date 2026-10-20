"""Pure helpers for the passenger view (no Streamlit, no network)."""

STATION_NAMES = {
    "MAS": "Chennai Central", "MS": "Chennai Egmore", "PER": "Perambur", "AJJ": "Arakkonam", "TRL": "Tiruvallur",
    "WJR": "Walajah Road", "KPD": "Katpadi", "AB": "Ambur", "JTJ": "Jolarpettai", "CGL": "Chengalpattu",
    "TBM": "Tambaram", "VM": "Villupuram", "VRI": "Vridhachalam", "TPJ": "Tiruchirappalli", "KRR": "Karur",
    "SLY": "Samalpatti", "MAP": "Morappur", "BQI": "Bommidi", "SA": "Salem", "ED": "Erode", "TUP": "Tiruppur",
    "CBF": "Coimbatore North", "CBE": "Coimbatore", "MTP": "Mettupalayam", "PGT": "Palakkad", "SRR": "Shoranur",
    "TIR": "Tirur", "CLT": "Kozhikode", "TLY": "Thalassery", "CAN": "Kannur", "PAY": "Payyanur",
    "KGQ": "Kasaragod", "MAQ": "Mangaluru Central", "ALLP": "Alappuzha", "ERS": "Ernakulam Jn", "AWY": "Aluva",
    "CKI": "Chalakudi", "TCR": "Thrissur", "OTP": "Ottappalam",
}

# Rain scenarios offered to the passenger, in mm for the day.
WEATHER = {"Normal": None, "Rainy": 15.0, "Heavy rain": 40.0}


def station_label(code):
    return f"{STATION_NAMES.get(code, code)} ({code})"


def classify(delay_min):
    """(label, colour) shown next to the headline number."""
    if delay_min <= 5:
        return "On time", "green"
    if delay_min <= 15:
        return "Slightly late", "orange"
    if delay_min <= 30:
        return "Late", "orange"
    return "Heavily delayed", "red"


def fmt_delay(minutes):
    m = int(round(minutes))
    if m < 60:
        return f"{m} min"
    return f"{m // 60} h {m % 60:02d} min"

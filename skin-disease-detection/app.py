"""
AI Skin Disease Detector - Flask application
Educational / research use only. NOT for medical diagnosis.
"""
import os
import uuid
import sqlite3
from datetime import datetime
from functools import wraps

import numpy as np
import requests
from flask import (Flask, render_template, request, jsonify, session,
                   send_file, redirect, url_for)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from fpdf import FPDF

# --------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_model.h5")
DB_PATH = os.path.join(BASE_DIR, "database.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB uploads

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# --------------------------------------------------------------------------
# HAM10000 class information
# --------------------------------------------------------------------------
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # alphabetical (Keras default)

DISEASE_INFO = {
    "akiec": ("Actinic Keratoses",
              "Rough, scaly patches caused by long-term sun exposure. Can be pre-cancerous.",
              "Consult a dermatologist. Use daily SPF 50 and avoid midday sun."),
    "bcc": ("Basal Cell Carcinoma",
            "The most common skin cancer. Grows slowly and rarely spreads.",
            "See a dermatologist promptly for examination and possible biopsy."),
    "bkl": ("Benign Keratosis-like Lesions",
            "Non-cancerous growths such as seborrheic keratoses or solar lentigines.",
            "Usually harmless. Monitor for changes and consult a doctor if unsure."),
    "df": ("Dermatofibroma",
           "A common benign fibrous skin nodule, often on the legs.",
           "Generally harmless. See a doctor if it changes, bleeds or becomes painful."),
    "mel": ("Melanoma",
            "A serious form of skin cancer that can spread if not caught early.",
            "Seek urgent evaluation by a dermatologist."),
    "nv": ("Melanocytic Nevi",
           "Common benign moles.",
           "Monitor using the ABCDE rule and see a doctor if a mole changes."),
    "vasc": ("Vascular Lesions",
             "Lesions of blood vessels such as angiomas and hemorrhages.",
             "Usually benign. A dermatologist can confirm and discuss treatment."),
}

# --------------------------------------------------------------------------
# Model loading (lazy, so the site still runs if no model file exists yet)
# --------------------------------------------------------------------------
_model = None
_model_error = None


def get_model():
    global _model, _model_error
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        _model_error = "Model file not found. Train it with training/train_cnn_improved.py"
        return None
    try:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
        print("Model loaded")
    except Exception as exc:  # noqa: BLE001
        _model_error = f"Could not load model: {exc}"
        print(_model_error)
    return _model


def preprocess_image(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unreadable image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(path):
    model = get_model()
    if model is None:
        raise RuntimeError(_model_error or "Model unavailable")
    preds = model.predict(preprocess_image(path), verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], round(float(preds[idx]) * 100, 2)


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT DEFAULT 'Completed',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, email TEXT, message TEXT,
            feedback_type TEXT, device_type TEXT,
            rating INTEGER, page TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, email TEXT, date TEXT, time TEXT, clinic TEXT,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


init_db()


class Row(dict):
    """dict that also allows attribute access (for Jinja: report.disease)."""
    __getattr__ = dict.get


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
CLINICS = [
    {"name": "Dermacare Jaipur, C-Scheme", "address": "C-Scheme, Jaipur, Rajasthan",
     "contact": "+91 141 000 0001", "timing": "Mon-Sat, 10:00 AM - 7:00 PM"},
    {"name": "Skin Clinic Jaipur, MI Road", "address": "MI Road, Jaipur, Rajasthan",
     "contact": "+91 141 000 0002", "timing": "Mon-Sat, 9:00 AM - 6:00 PM"},
    {"name": "Apollo Dermatology Center, Tonk Road", "address": "Tonk Road, Jaipur, Rajasthan",
     "contact": "+91 141 000 0003", "timing": "Mon-Sun, 9:00 AM - 8:00 PM"},
    {"name": "Max Healthcare - Dermatology, Jagatpura", "address": "Jagatpura, Jaipur, Rajasthan",
     "contact": "+91 141 000 0004", "timing": "Mon-Sat, 10:00 AM - 5:00 PM"},
    {"name": "Glowderma Skin & Hair Clinic, Bani Park", "address": "Bani Park, Jaipur, Rajasthan",
     "contact": "+91 141 000 0005", "timing": "Mon-Sat, 11:00 AM - 8:00 PM"},
    {"name": "Fortis Hospital - Dermatology Wing, Chanakyapuri", "address": "Chanakyapuri, Jaipur, Rajasthan",
     "contact": "+91 141 000 0006", "timing": "24 x 7 (OPD 9 AM - 5 PM)"},
]
# NOTE: contact numbers are placeholders - replace with real data.


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/appointments")
def appointments():
    return render_template("appointments.html", clinics=CLINICS)


@app.route("/clinics")
def clinics():
    return render_template("clinics.html", clinics=CLINICS)


@app.route("/reports")
def reports():
    conn = get_db()
    if session.get("user_id"):
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 20",
            (session["user_id"],)).fetchall()
    else:
        rows = []
    conn.close()
    items = []
    for r in rows:
        d = dict(r)
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        items.append(Row(d))
    return render_template("reports.html", reports=items)


@app.route("/tips")
def tips():
    return render_template("tips.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/chat-page")
def chat_page():
    return render_template("chat.html")


# --------------------------------------------------------------------------
# Prediction + result + report
# --------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "" or not allowed_file(file.filename):
        return "Please upload a PNG or JPG image.", 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        code, confidence = predict_image(path)
    except Exception as exc:  # noqa: BLE001
        return f"Prediction failed: {exc}", 500

    name, desc, tip = DISEASE_INFO[code]

    conn = get_db()
    conn.execute(
        "INSERT INTO predictions (user_id, image_path, disease, confidence, created_at) VALUES (?,?,?,?,?)",
        (session.get("user_id"), filename, name, confidence, datetime.now().isoformat()))
    conn.commit()
    hist = conn.execute(
        "SELECT * FROM predictions WHERE user_id IS ? ORDER BY id DESC LIMIT 5",
        (session.get("user_id"),)).fetchall()
    conn.close()

    history = [{"image": h["image_path"][33:] or h["image_path"], "disease": h["disease"],
                "confidence": h["confidence"], "time": h["created_at"][:16].replace("T", " ")}
               for h in hist]

    return render_template("result.html", filename=filename, disease=name,
                           confidence=confidence, desc=desc, tip=tip, history=history)


@app.route("/report/<filename>/<disease>/<confidence>")
def report_path(filename, disease, confidence):
    return build_report(filename, disease, confidence)


@app.route("/report")
def report():
    return build_report(request.args.get("filename", ""),
                        request.args.get("disease", "Unknown"),
                        request.args.get("confidence", "0"))


def build_report(filename, disease, confidence):
    filename = secure_filename(filename)
    img_path = os.path.join(UPLOAD_FOLDER, filename)

    def clean(text):  # FPDF core fonts are latin-1 only
        return str(text).encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "AI Skin Analysis Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", ln=True, align="C")
    pdf.ln(6)
    if filename and os.path.exists(img_path):
        pdf.image(img_path, x=65, w=80)
        pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, clean(f"Predicted condition: {disease}"), ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, clean(f"Confidence: {confidence}%"), ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, "Disclaimer: This report is produced by an educational machine-learning "
                         "model and is NOT a medical diagnosis. Please consult a qualified "
                         "dermatologist.")
    out = os.path.join(REPORT_FOLDER, f"report_{uuid.uuid4().hex[:8]}.pdf")
    pdf.output(out)
    return send_file(out, as_attachment=True, download_name="skin_report.pdf")


# --------------------------------------------------------------------------
# Chatbot (OpenRouter) - with offline fallback
# --------------------------------------------------------------------------
FALLBACK_REPLIES = {
    "acne": "For acne: cleanse gently twice a day, avoid picking, use salicylic acid or benzoyl peroxide, and change pillowcases often.",
    "spf": "Use broad-spectrum SPF 30+ daily and reapply every 2-3 hours when outdoors.",
    "dry": "For dry skin use a hydrating cleanser, hyaluronic acid serum and a rich moisturizer.",
    "oily": "For oily skin use a foaming cleanser, niacinamide and an oil-free moisturizer.",
    "mole": "Watch moles using ABCDE (Asymmetry, Border, Color, Diameter, Evolving). See a dermatologist if one changes.",
}


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please type a message."}), 400

    if OPENROUTER_API_KEY:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={"model": OPENROUTER_MODEL, "messages": [
                    {"role": "system", "content":
                        "You are a friendly skincare assistant. Give general skincare "
                        "information only, never diagnose, and recommend seeing a "
                        "dermatologist for medical concerns. Keep answers short."},
                    {"role": "user", "content": message}]},
                timeout=25)
            r.raise_for_status()
            return jsonify({"reply": r.json()["choices"][0]["message"]["content"]})
        except Exception as exc:  # noqa: BLE001
            print("OpenRouter error:", exc)

    lower = message.lower()
    for key, reply in FALLBACK_REPLIES.items():
        if key in lower:
            return jsonify({"reply": reply})
    return jsonify({"reply": "I can help with general skincare questions (acne, SPF, dry/oily skin, moles). "
                             "For medical concerns please see a dermatologist."})


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
@app.route("/signup", methods=["POST"])
def signup():
    d = request.get_json(silent=True) or {}
    username = (d.get("username") or "").strip()
    email = (d.get("email") or "").strip().lower()
    password = d.get("password") or ""
    if len(username) < 3 or "@" not in email or len(password) < 6:
        return jsonify(success=False, message="Check username (3+), email and password (6+).")
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?,?,?,?)",
            (username, email, generate_password_hash(password), datetime.now().isoformat()))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify(success=False, message="Email already registered.")
    uid = cur.lastrowid
    conn.close()
    session["user_id"], session["username"] = uid, username
    return jsonify(success=True, message="Account created!", user={"username": username})


@app.route("/login", methods=["POST"])
def login():
    d = request.get_json(silent=True) or {}
    email = (d.get("email") or "").strip().lower()
    password = d.get("password") or ""
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify(success=False, message="Invalid email or password.")
    session["user_id"], session["username"] = user["id"], user["username"]
    return jsonify(success=True, message="Welcome back!", user={"username": user["username"]})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify(success=True, message="Logged out.")


@app.route("/check-auth")
def check_auth():
    if session.get("user_id"):
        return jsonify(authenticated=True, user={"username": session["username"]})
    return jsonify(authenticated=False)


# --------------------------------------------------------------------------
# Feedback + appointments
# --------------------------------------------------------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    d = request.get_json(silent=True) or {}
    if not d.get("name") or not d.get("email") or not d.get("message") or not d.get("agree"):
        return jsonify(success=False, reply="Please fill all required fields.")
    conn = get_db()
    conn.execute(
        "INSERT INTO feedback (name,email,message,feedback_type,device_type,rating,page,created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (d["name"], d["email"], d["message"], d.get("feedback_type"), d.get("device_type"),
         d.get("rating"), d.get("page"), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return jsonify(success=True)


@app.route("/book-appointment", methods=["POST"])
def book_appointment():
    d = request.get_json(silent=True) or {}
    if not all(d.get(k) for k in ("name", "email", "date", "time", "clinic")):
        return jsonify(success=False, message="Please fill all fields.")
    conn = get_db()
    conn.execute("INSERT INTO appointments (name,email,date,time,clinic,created_at) VALUES (?,?,?,?,?,?)",
                 (d["name"], d["email"], d["date"], d["time"], d["clinic"], datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return jsonify(success=True, message="Appointment booked!")


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, send_file
from routes.mock_test import mock_test_bp
from routes.mock_interview import mock_interview_bp
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import PyPDF2
import docx
import re
import numpy as np
import json
import os
import re


from dotenv import load_dotenv
import os
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.register_blueprint(mock_test_bp)
app.register_blueprint(mock_interview_bp)
CORS(app)

# ── Database config ──────────────────────────────────────────────
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///alignmetric.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── Gemini API config ────────────────────────────────────────────
# Put this in your .env file:
# GEMINI_API_KEY=your_actual_api_key_here

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")



# You can also set this in .env if you want:
# GEMINI_MODEL=gemini-2.5-flash
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ── Models ───────────────────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    name = db.Column(db.String(150))
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('Analysis', backref='user', lazy=True)


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(150), db.ForeignKey('user.email'), nullable=False)
    match_score = db.Column(db.Integer)
    jd_snippet = db.Column(db.Text)
    resume_snippet = db.Column(db.Text)
    met_count = db.Column(db.Integer)
    missed_count = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

# ── Sentence Transformer ─────────────────────────────────────────
print("[INFO] Loading sentence-transformer model …")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[INFO] Model ready.")
SIMILARITY_THRESHOLD = 0.72


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def call_gemini(prompt: str, max_tokens: int = 1500) -> str:

    # Use the correct API for Gemini (google.generativeai)
    model = getattr(genai, 'GenerativeModel', None)
    if model is not None:
        model = model(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": 0.3
            }
        )
        return response.text if hasattr(response, 'text') else str(response)
    else:
        # Fallback: try lower-level API if available
        return "Gemini API not available. Please check your google-generativeai package version."




def strip_json_fences(text: str) -> str:
    """Extract clean JSON from AI response."""

    if not text:
        return ""

    # Remove ```json and ```
    text = re.sub(r"```json\s*|\s*```", "", text)

    # Extract JSON part only
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        return text[start:end+1].strip()

    return text.strip()


# ── Text extraction ───────────────────────────────────────────────
def extract_text(file) -> str:
    if file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "".join(p.extract_text() or "" for p in reader.pages)
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join(p.text for p in doc.paragraphs)
    return ""


# ── Segment helpers ───────────────────────────────────────────────
def split_into_segments(text: str) -> list:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.split(r"[\n•\-–—]+", text)
    segs = []
    for part in raw:
        for s in re.split(r"(?<=[.!?])\s+", part.strip()):
            s = s.strip()
            if len(s) > 8:
                segs.append(s)
    return segs


def categorise_segment(seg: str) -> str:
    pat = re.compile(
        r"\b(led|managed|developed|built|designed|implemented|worked|"
        r"achieved|years?|months?|delivered|increased|reduced|improved|"
        r"spearheaded|collaborated|responsible|experience)\b",
        re.IGNORECASE
    )
    return "experience" if pat.search(seg) else "skill"


def compute_similarity_matrix(res_segs: list, jd_segs: list) -> np.ndarray:
    if not res_segs or not jd_segs:
        return np.zeros((max(len(res_segs), 1), max(len(jd_segs), 1)))
    re_emb = MODEL.encode(res_segs, convert_to_tensor=True)
    jd_emb = MODEL.encode(jd_segs, convert_to_tensor=True)
    return util.cos_sim(re_emb, jd_emb).cpu().numpy()


def build_annotated_segments(res_segs, jd_segs, sim, threshold):
    res_out = [
        {
            "text": s,
            "is_match": float(sim[i].max()) >= threshold,
            "score": round(float(sim[i].max()), 3),
            "category": categorise_segment(s)
        }
        for i, s in enumerate(res_segs)
    ]
    jd_out = [
        {
            "text": s,
            "is_match": float(sim[:, j].max()) >= threshold,
            "score": round(float(sim[:, j].max()), 3),
            "category": categorise_segment(s)
        }
        for j, s in enumerate(jd_segs)
    ]
    return res_out, jd_out


def calculate_match_score(jd_out: list) -> int:
    if not jd_out:
        return 0
    return round(sum(1 for s in jd_out if s["is_match"]) / len(jd_out) * 100)


def analyze_text_score(resume_text: str, jd_text: str) -> int:
    """Quick score — used by /simulate."""
    res_segs = split_into_segments(resume_text)
    jd_segs = split_into_segments(jd_text)
    if not res_segs or not jd_segs:
        return 0
    sim = compute_similarity_matrix(res_segs, jd_segs)
    _, jd_out = build_annotated_segments(res_segs, jd_segs, sim, SIMILARITY_THRESHOLD)
    return calculate_match_score(jd_out)


def save_analysis(email, jd_text, resume_text, jd_out, score):
    met = sum(1 for s in jd_out if s["is_match"])
    missed = sum(1 for s in jd_out if not s["is_match"])
    record = Analysis()
    record.user_email = email
    record.match_score = score
    record.jd_snippet = jd_text[:120]
    record.resume_snippet = resume_text[:120]
    record.met_count = met
    record.missed_count = missed
    db.session.add(record)
    db.session.commit()


# ═══════════════════════════════════════════════════════════════
# ATS SCORE ENGINE
# ═══════════════════════════════════════════════════════════════
def compute_ats_score(resume_text: str, jd_text: str) -> dict:
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()

    # 1. Keyword match (30%)
    STOPWORDS = {
        "with", "that", "this", "from", "have", "will", "your", "they", "their",
        "about", "into", "more", "also", "been", "were", "when", "what", "which",
        "would", "could", "should", "there", "these", "those", "other", "some",
        "such", "than", "then", "them", "well", "only", "both", "each", "over",
        "after", "before", "while", "where", "work", "must", "able", "like",
        "make", "need", "team", "good", "using", "being", "help", "used",
    }
    jd_tokens = set(w for w in re.findall(r"\b[a-z][a-z0-9+#.]{2,}\b", jd_lower) if w not in STOPWORDS)
    resume_tokens = set(re.findall(r"\b[a-z][a-z0-9+#.]{2,}\b", resume_lower))
    if jd_tokens:
        kw_hits = jd_tokens & resume_tokens
        kw_score = min(100, round(len(kw_hits) / len(jd_tokens) * 100))
        missing_kw = sorted(jd_tokens - resume_tokens)
    else:
        kw_score, kw_hits, missing_kw = 0, set(), []

    # 2. Semantic match (25%)
    res_segs = split_into_segments(resume_text)
    jd_segs = split_into_segments(jd_text)
    sim_mat = compute_similarity_matrix(res_segs, jd_segs)
    sem_score = round(float(sim_mat.max(axis=0).mean()) * 100) if sim_mat.size > 0 else 0

    # 3. Skills coverage (20%)
    TECH_SKILLS = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue", "node",
        "django", "flask", "fastapi", "spring", "express", "rails", "laravel",
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins",
        "git", "linux", "bash", "rest", "graphql", "grpc", "kafka", "rabbitmq",
        "machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "sklearn",
        "pandas", "numpy", "spark", "hadoop", "airflow", "dbt", "tableau", "powerbi",
        "figma", "sketch", "photoshop", "css", "html", "tailwind", "bootstrap",
        "agile", "scrum", "jira", "confluence", "ci/cd", "devops", "microservices",
        "c++", "c#", "go", "rust", "kotlin", "swift", "r", "scala", "matlab",
    ]
    jd_skills = [s for s in TECH_SKILLS if s in jd_lower]
    matched_skills = [s for s in jd_skills if s in resume_lower]
    missing_skills = [s for s in jd_skills if s not in resume_lower]
    skills_score = round(len(matched_skills) / len(jd_skills) * 100) if jd_skills else 80

    # 4. Experience depth (10%)
    QUANT_PAT = re.compile(
        r"\b\d+[\+%x]?\s*(years?|months?|\%|x|users?|clients?|engineers?|million|billion|k\b|projects?)\b",
        re.I
    )
    ACTION_VERBS = re.compile(
        r"\b(led|managed|built|designed|architected|scaled|reduced|increased|improved|delivered|launched|optimised|optimized|automated|mentored|hired)\b",
        re.I
    )
    quant_hits = len(QUANT_PAT.findall(resume_text))
    action_hits = len(ACTION_VERBS.findall(resume_text))
    exp_score = min(100, quant_hits * 12 + action_hits * 5)

    # 5. Format quality (10%)
    SECTION_HEADERS = re.compile(
        r"\b(experience|education|skills?|summary|objective|projects?|certifications?|achievements?|publications?|awards?|languages?)\b",
        re.I
    )
    BULLET_CUES = re.compile(r"(•|\-|\*|›|▸|\d+\.)\s+\w")
    sections_found = len(set(SECTION_HEADERS.findall(resume_text)))
    bullets_found = len(BULLET_CUES.findall(resume_text))
    word_count = len(resume_text.split())
    fmt_score = min(
        100,
        sections_found * 12 + min(bullets_found, 10) * 4 + (20 if 300 <= word_count <= 1200 else 0)
    )

    # 6. Education match (5%)
    EDU_TERMS = re.compile(
        r"\b(bachelor|master|phd|doctorate|mba|b\.?s\.?c?|m\.?s\.?c?|b\.?e\.?|m\.?e\.?|degree|diploma|certified|certification|coursera|udemy|bootcamp|stanford|mit|iit|nit|university|college)\b",
        re.I
    )
    edu_hits = len(set(EDU_TERMS.findall(resume_text)))
    edu_score = min(100, edu_hits * 20)

    # Weighted total
    weights = dict(keyword=0.30, semantic=0.25, skills=0.20, experience=0.10, format=0.10, education=0.05)
    scores = dict(keyword=kw_score, semantic=sem_score, skills=skills_score, experience=exp_score, format=fmt_score, education=edu_score)
    total = round(sum(scores[k] * weights[k] for k in weights))

    # Improvement tips
    tips = []
    if missing_skills:
        tips.append({
            "priority": "high",
            "category": "Skills",
            "icon": "code",
            "title": "Add missing technical skills",
            "detail": "These skills appear in the JD but not your resume: " + ", ".join(missing_skills[:8]) + ("…" if len(missing_skills) > 8 else ""),
            "impact": f"+{min(20, len(missing_skills) * 3)}pts potential"
        })
    top_missing_kw = [w for w in missing_kw if len(w) > 4 and w not in missing_skills][:10]
    if top_missing_kw:
        tips.append({
            "priority": "high",
            "category": "Keywords",
            "icon": "key",
            "title": "Weave in missing JD keywords",
            "detail": "Consider naturally adding: " + ", ".join(top_missing_kw),
            "impact": f"+{min(15, len(top_missing_kw) * 2)}pts potential"
        })
    if quant_hits < 4:
        tips.append({
            "priority": "medium",
            "category": "Impact",
            "icon": "bar_chart",
            "title": "Quantify your achievements",
            "detail": f"Only {quant_hits} quantified result(s) found. Add metrics like 'Reduced load time by 40%'.",
            "impact": "+8pts potential"
        })
    if sections_found < 4:
        tips.append({
            "priority": "medium",
            "category": "Format",
            "icon": "view_list",
            "title": "Add standard resume sections",
            "detail": f"Only {sections_found} section header(s) detected. Include: Summary, Experience, Skills, Education, Projects.",
            "impact": "+7pts potential"
        })
    if bullets_found < 5:
        tips.append({
            "priority": "medium",
            "category": "Format",
            "icon": "format_list_bulleted",
            "title": "Use bullet points for experience",
            "detail": "Bullet-point lists improve ATS parsing. Replace dense paragraphs with action-verb bullets.",
            "impact": "+5pts potential"
        })
    if word_count < 300:
        tips.append({
            "priority": "medium",
            "category": "Content",
            "icon": "edit_note",
            "title": "Expand your resume content",
            "detail": f"Your resume has ~{word_count} words. Aim for 400–800 words.",
            "impact": "+6pts potential"
        })
    elif word_count > 1200:
        tips.append({
            "priority": "low",
            "category": "Content",
            "icon": "compress",
            "title": "Trim resume length",
            "detail": f"~{word_count} words detected. Most ATS systems prefer under 900 words.",
            "impact": "+3pts potential"
        })
    if edu_score < 40:
        tips.append({
            "priority": "low",
            "category": "Education",
            "icon": "school",
            "title": "Strengthen education section",
            "detail": "No degree, certification, or institution found. Add your highest qualification.",
            "impact": "+4pts potential"
        })
    if action_hits < 5:
        tips.append({
            "priority": "low",
            "category": "Language",
            "icon": "bolt",
            "title": "Use stronger action verbs",
            "detail": f"Only {action_hits} strong action verb(s) detected. Start bullets with: Led, Built, Designed, Reduced.",
            "impact": "+4pts potential"
        })
    tips.sort(key=lambda t: {"high": 0, "medium": 1, "low": 2}[t["priority"]])

    return {
        "ats_score": total,
        "breakdown": {
            "keyword": {"score": kw_score, "weight": 30, "label": "Keyword Match"},
            "semantic": {"score": sem_score, "weight": 25, "label": "Semantic Relevance"},
            "skills": {"score": skills_score, "weight": 20, "label": "Skills Coverage"},
            "experience": {"score": exp_score, "weight": 10, "label": "Experience Depth"},
            "format": {"score": fmt_score, "weight": 10, "label": "Format Quality"},
            "education": {"score": edu_score, "weight": 5, "label": "Education Match"},
        },
        "missing_skills": missing_skills[:12],
        "missing_keywords": top_missing_kw,
        "matched_skills": matched_skills,
        "tips": tips,
        "stats": {
            "word_count": word_count,
            "quant_hits": quant_hits,
            "action_hits": action_hits,
            "sections": sections_found,
            "bullets": bullets_found,
        },
    }


# ═══════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route("/")
def home():
    return send_file("landing.html")

@app.route("/auth")          # ← ADD THIS
def auth():
    return send_file("auth.html")

@app.route("/app")
def dashboard():
    return send_file("index.html")


@app.route("/register", methods=["POST"])
def register():
    data = request.json
    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "User already exists"}), 400
    user = User()
    user.email = data["email"]
    user.name = data.get("name", "")
    user.password_hash = generate_password_hash(data["password"])
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Registered successfully"})


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = User.query.filter_by(email=data["email"]).first()
    if not user or not check_password_hash(user.password_hash, data["password"]):
        return jsonify({"error": "Invalid credentials"}), 401
    return jsonify({"message": "Login successful", "name": user.name})


# ═══════════════════════════════════════════════════════════════
# HISTORY
# ═══════════════════════════════════════════════════════════════
@app.route("/history", methods=["GET"])
def history():
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "email param required"}), 400
    records = (
        Analysis.query
        .filter_by(user_email=email)
        .order_by(Analysis.created_at.desc())
        .limit(50).all()
    )
    return jsonify([{
        "id": r.id,
        "match_score": r.match_score,
        "jd_snippet": r.jd_snippet,
        "resume_snippet": r.resume_snippet,
        "met_count": r.met_count,
        "missed_count": r.missed_count,
        "created_at": r.created_at.strftime("%d %b %Y, %H:%M"),
    } for r in records])


# ═══════════════════════════════════════════════════════════════
# MATCH ANALYSIS ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("resume")
    job = request.form.get("job")
    email = request.form.get("email", "")
    if not file or not job:
        return jsonify({"error": "Missing data"}), 400

    resume_text = extract_text(file)
    if not resume_text.strip():
        return jsonify({"error": "Could not extract text from file."}), 422

    res_segs, jd_segs = split_into_segments(resume_text), split_into_segments(job)
    sim = compute_similarity_matrix(res_segs, jd_segs)
    res_out, jd_out = build_annotated_segments(res_segs, jd_segs, sim, SIMILARITY_THRESHOLD)
    score = calculate_match_score(jd_out)

    if email:
        save_analysis(email, job, resume_text, jd_out, score)

    return jsonify({"match": score, "resume_segments": res_out, "jd_segments": jd_out})


@app.route("/analyze-text", methods=["POST"])
def analyze_text_route():
    data = request.get_json(force=True)
    resume_text = data.get("resume", "").strip()
    jd_text = data.get("jd", "").strip()
    email = data.get("email", "")
    if not resume_text or not jd_text:
        return jsonify({"error": "Both 'resume' and 'jd' fields are required."}), 400

    res_segs, jd_segs = split_into_segments(resume_text), split_into_segments(jd_text)
    sim = compute_similarity_matrix(res_segs, jd_segs)
    res_out, jd_out = build_annotated_segments(res_segs, jd_segs, sim, SIMILARITY_THRESHOLD)
    score = calculate_match_score(jd_out)

    if email:
        save_analysis(email, jd_text, resume_text, jd_out, score)

    return jsonify({"match": score, "resume_segments": res_out, "jd_segments": jd_out})


# ═══════════════════════════════════════════════════════════════
# ATS SCORE ROUTE
# ═══════════════════════════════════════════════════════════════
@app.route("/ats-score", methods=["POST"])
def ats_score_route():
    if request.content_type and "multipart" in request.content_type:
        file = request.files.get("resume")
        jd_text = request.form.get("job", "").strip()
        if not file:
            return jsonify({"error": "No resume file provided."}), 400
        resume_text = extract_text(file)
    else:
        data = request.get_json(force=True)
        resume_text = data.get("resume", "").strip()
        jd_text = data.get("jd", "").strip()

    if not resume_text:
        return jsonify({"error": "Could not read resume text."}), 422
    if not jd_text:
        return jsonify({"error": "Job description is required."}), 400

    return jsonify(compute_ats_score(resume_text, jd_text))


# ═══════════════════════════════════════════════════════════════
# AI COACH  (Gemini — section-by-section rewrite suggestions)
# ═══════════════════════════════════════════════════════════════
@app.route("/ai-coach", methods=["POST"]) # type: ignore
def ai_coach():
    data = request.get_json(force=True)
    resume_text = data.get("resume", "").strip()
    jd_text = data.get("jd", "").strip()
    if not resume_text or not jd_text:
        return jsonify({"error": "Both 'resume' and 'jd' fields are required."}), 400

    prompt = f"""
You are an expert resume coach.

Your task is to analyze the resume against the job description.

IMPORTANT:
Return ONLY a VALID JSON array.
Do NOT include explanation.
Do NOT include markdown.
Do NOT include text before or after JSON.

STRICT RULES:
- Use ONLY double quotes (" ")
- No trailing commas
- No broken or multiline strings
- Output must be directly parsable using json.loads()

REQUIRED FORMAT:
[
  {{
    "section": "string",
    "priority": "high",
    "points": 10,
    "issue": "string",
    "label": "string",
    "rewrite": "string"
  }}
]

Job Description:
{jd_text}

Resume:
{resume_text}
"""
    import re
    raw = ""
    try:
        raw = call_gemini(prompt, max_tokens=4000)
        print("\n===== GEMINI RAW OUTPUT =====\n", raw)
        # 🔥 Extract ONLY JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return jsonify({
                "error": "No JSON found in AI response",
                "raw_output": raw[:300]
            }), 500
        json_text = match.group(0)
        # 🔧 Fix broken quotes/newlines
        json_text = json_text.replace("\n", " ").replace("\r", " ")
        data_out = json.loads(json_text)
        return jsonify(data_out)
    except Exception as e:
        return jsonify({
            "error": "AI response broken",
            "raw_output": raw[:300]
        }), 500


# ═══════════════════════════════════════════════════════════════
# GENERATE ROADMAP  (Gemini — 3-week skill-building plan)
# ═══════════════════════════════════════════════════════════════
@app.route("/generate_roadmap", methods=["POST"])
def generate_roadmap():
    data = request.get_json(force=True)
    missing_skills = data.get("missing_skills", [])
    job_title = data.get("job_title", "target role")

    if not missing_skills:
        return jsonify({"error": "No missing skills provided."}), 400

    # 🔥 Improved STRICT prompt
    prompt = f"""
You are an expert career coach.

Return ONLY VALID JSON.
No explanation.
No markdown.
No extra text.

STRICT RULES:
- All strings must be in double quotes
- No trailing commas
- No missing brackets
- Output must be directly parsable with json.loads()

FORMAT:
{{
  "week1": {{
    "goal": "string",
    "tasks": ["string", "string", "string", "string"],
    "output": "string"
  }},
  "week2": {{
    "goal": "string",
    "tasks": ["string", "string", "string", "string"],
    "output": "string"
  }},
  "week3": {{
    "goal": "string",
    "tasks": ["string", "string", "string", "string"],
    "output": "string"
  }}
}}

Job role: {job_title}
Missing skills: {', '.join(missing_skills)}
"""

    try:
        raw = call_gemini(prompt, max_tokens=1200)

        # 🔍 DEBUG (optional)
        print("\n===== GEMINI RAW OUTPUT =====\n", raw)

        clean = strip_json_fences(raw)

        # 🔥 Try parsing JSON safely
        try:
            roadmap = json.loads(clean)
            return jsonify({"roadmap": roadmap})

        except json.JSONDecodeError:
            print("❌ JSON FAILED, trying to fix...")

            # 🔧 Basic fix attempt (remove newlines inside strings)
            fixed = clean.replace("\n", " ").replace("\r", " ")

            try:
                roadmap = json.loads(fixed)
                return jsonify({"roadmap": roadmap})
            except:
                return jsonify({
                    "error": "AI returned invalid JSON. Try again.",
                    "raw_output": clean[:500]  # send partial for debugging
                }), 500

    except Exception as e:
        return jsonify({"error": f"Gemini API error: {str(e)}"}), 500


# ═══════════════════════════════════════════════════════════════
# SIMULATE  (what-if score if a skill is added to the resume)
# ═══════════════════════════════════════════════════════════════
@app.route("/simulate", methods=["POST"])
def simulate():
    """
    Accepts JSON: { "resume_text": "...", "job_text": "...", "skill_to_add": "..." }
    Returns: { "original_score": int, "new_score": int, "delta": int, "simulated_resume": str }
    """
    data = request.get_json(force=True)
    resume_text = data.get("resume_text", "").strip()
    job_text = data.get("job_text", "").strip()
    new_skill = data.get("skill_to_add", "").strip()

    if not resume_text or not job_text or not new_skill:
        return jsonify({"error": "Missing fields: resume_text, job_text, skill_to_add"}), 400

    original_score = analyze_text_score(resume_text, job_text)
    simulated_resume = resume_text + "\n" + new_skill
    new_score = analyze_text_score(simulated_resume, job_text)

    return jsonify({
        "original_score": original_score,
        "new_score": new_score,
        "delta": new_score - original_score,
        "simulated_resume": simulated_resume
    })


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True)
from groq import Groq
import json, sqlite3, os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/ui")
def serve_ui():
    return FileResponse("index.html")

client = Groq()

# ─── DATABASE SETUP ───────────────────────────────────────────
DB_FILE = "evaluations.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            domain      TEXT,
            question    TEXT,
            answer      TEXT,
            accuracy    REAL,
            clarity     REAL,
            completeness REAL,
            reasoning   REAL,
            overall     REAL,
            verdict     TEXT,
            summary     TEXT,
            suggestions TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(domain, question, answer, result):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evaluations 
        (timestamp, domain, question, answer, accuracy, clarity, completeness, reasoning, overall, verdict, summary, suggestions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        domain,
        question,
        answer,
        result.get("accuracy"),
        result.get("clarity"),
        result.get("completeness"),
        result.get("reasoning"),
        result.get("overall"),
        result.get("verdict"),
        result.get("summary"),
        json.dumps(result.get("suggestions", []))
    ))
    conn.commit()
    conn.close()

# initialise DB when server starts
init_db()

# ─── REQUEST MODEL ─────────────────────────────────────────────
class EvalRequest(BaseModel):
    question: str
    answer: str
    domain: str = "Mathematics"

# ─── ROUTES ────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "running"}


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert AI evaluator in {req.domain}.
Respond ONLY with valid JSON. No markdown, no backticks, nothing else.
Score 0-10 on: accuracy, clarity, completeness, reasoning.
Also give: overall (accuracy 40% + others 20% each), verdict (Excellent/Good/Fair/Poor), summary, suggestions (list of 2-3).
Exact shape:
{{"accuracy":7.5,"clarity":8.0,"completeness":6.0,"reasoning":7.0,"overall":7.2,"verdict":"Good","summary":"...","suggestions":["...","..."]}}"""
            },
            {
                "role": "user",
                "content": f"Domain: {req.domain}\nQuestion: {req.question}\nAnswer to evaluate:\n{req.answer}"
            }
        ],
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    print("=== GROQ RAW OUTPUT ===")
    print(raw)
    print("=======================")

    clean = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(clean)

    # save to database
    save_to_db(req.domain, req.question, req.answer, result)

    return result


@app.get("/history")
def get_history():
    """Returns all past evaluations, newest first."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluations ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        entry = dict(row)
        entry["suggestions"] = json.loads(entry["suggestions"])
        history.append(entry)

    return {"total": len(history), "evaluations": history}


@app.get("/history/stats")
def get_stats():
    """Returns summary statistics across all evaluations."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            ROUND(AVG(overall), 2) as avg_overall,
            ROUND(AVG(accuracy), 2) as avg_accuracy,
            ROUND(AVG(clarity), 2) as avg_clarity,
            ROUND(AVG(completeness), 2) as avg_completeness,
            ROUND(AVG(reasoning), 2) as avg_reasoning,
            MAX(overall) as highest_score,
            MIN(overall) as lowest_score
        FROM evaluations
    """)
    row = cursor.fetchone()

    cursor.execute("SELECT verdict, COUNT(*) as count FROM evaluations GROUP BY verdict")
    verdicts = {r[0]: r[1] for r in cursor.fetchall()}

    cursor.execute("SELECT domain, COUNT(*) as count FROM evaluations GROUP BY domain ORDER BY count DESC")
    domains = {r[0]: r[1] for r in cursor.fetchall()}

    conn.close()
    return {
        "total_evaluations": row[0],
        "average_scores": {
            "overall": row[1],
            "accuracy": row[2],
            "clarity": row[3],
            "completeness": row[4],
            "reasoning": row[5]
        },
        "highest_score": row[6],
        "lowest_score": row[7],
        "verdicts_breakdown": verdicts,
        "domains_breakdown": domains
    }


@app.delete("/history/clear")
def clear_history():
    """Clears all evaluation records."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM evaluations")
    conn.commit()
    conn.close()
    return {"message": "All records cleared."}
from groq import Groq
import json, sqlite3, os
from fastapi.responses import FileResponse
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/ui")
def serve_ui():
    return FileResponse("index.html")

client = None

def get_groq_client():
    global client
    if client is not None:
        return client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set.")
    client = Groq(api_key=api_key)
    return client

DB_FILE = "evaluations.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            evaluator    TEXT,
            domain       TEXT,
            question     TEXT,
            answer       TEXT,
            accuracy     REAL,
            clarity      REAL,
            completeness REAL,
            reasoning    REAL,
            overall      REAL,
            verdict      TEXT,
            summary      TEXT,
            suggestions  TEXT
        )
    """)
    existing_columns = [row[1] for row in cursor.execute("PRAGMA table_info(evaluations)")]
    if "evaluator" not in existing_columns:
        cursor.execute("ALTER TABLE evaluations ADD COLUMN evaluator TEXT DEFAULT 'evaluator1'")
    conn.commit()
    conn.close()

def save_to_db(evaluator, domain, question, answer, result):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evaluations
        (timestamp, evaluator, domain, question, answer,
         accuracy, clarity, completeness, reasoning,
         overall, verdict, summary, suggestions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        evaluator,
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

init_db()

class EvalRequest(BaseModel):
    question:  str
    answer:    str
    domain:    str = "Mathematics"
    evaluator: str = "evaluator1"

def build_prompt(evaluator: str, domain: str) -> str:
    if evaluator == "evaluator1":
        return f"""You are an expert AI evaluator in {domain}.

Respond ONLY with valid JSON. No markdown. No backticks. No explanations outside JSON.

Scoring rubric (0-10):
- accuracy:     is the answer factually correct?
- clarity:      is it clear and easy to follow?
- completeness: does it fully address the question?
- reasoning:    is the logic sound and well-justified?

Return ONLY these 4 scores plus summary and suggestions.
Do NOT compute overall or verdict.

Exact JSON shape:
{{"accuracy":8.5,"clarity":9.0,"completeness":8.0,"reasoning":7.5,"summary":"...","suggestions":["...","..."]}}"""

    elif evaluator == "evaluator2":
        return f"""You are an expert evaluator for hallucination and truthfulness detection in {domain}.

Respond ONLY with valid JSON. No markdown. No backticks. No explanations outside JSON.

Scoring rubric (0-10):
- factuality:    how factually accurate the answer is
- hallucination: how free from fabricated claims (10 = zero hallucination, 0 = fully fabricated)
- relevance:     how well the answer addresses the question
- reasoning:     logical consistency and evidence-based reasoning

Important:
- A factually WRONG answer must score factuality <= 3, hallucination <= 3
- If the answer states something false as fact, overall MUST be below 5.0
- Do not reward fluency or confidence in wrong answers
- Score based on correctness ONLY — style and length do not matter
- When in doubt whether an answer is wrong, assume it is wrong

Return ONLY these 4 scores plus summary and suggestions.
Do NOT compute overall or verdict.

Exact JSON shape:
{{"factuality":8.5,"hallucination":9.0,"relevance":8.0,"reasoning":7.5,"summary":"...","suggestions":["...","..."]}}"""

    else:
        raise ValueError(f"Unknown evaluator: {evaluator}")

def normalize(raw: dict, evaluator: str) -> dict:
    if evaluator == "evaluator1":
        accuracy     = raw.get("accuracy",     0)
        clarity      = raw.get("clarity",      0)
        completeness = raw.get("completeness", 0)
        reasoning    = raw.get("reasoning",    0)
        overall = round(
            accuracy     * 0.40 +
            clarity      * 0.20 +
            completeness * 0.20 +
            reasoning    * 0.20,
            2
        )

    elif evaluator == "evaluator2":
        accuracy     = raw.get("factuality",    0)
        clarity      = raw.get("hallucination", 0)
        completeness = raw.get("relevance",     0)
        reasoning    = raw.get("reasoning",     0)
        overall = round(
            accuracy     * 0.40 +
            clarity      * 0.30 +
            completeness * 0.15 +
            reasoning    * 0.15,
            2
        )

    else:
        raise ValueError(f"Unknown evaluator: {evaluator}")

    if evaluator == "evaluator1":
        if overall >= 8.5:
            verdict = "Excellent"
        elif overall >= 6.5:
            verdict = "Good"
        elif overall >= 4.5:
            verdict = "Fair"
        else:
            verdict = "Poor"
    else:
        if overall >= 8.0:
            verdict = "Excellent"
        elif overall >= 6.0:
            verdict = "Good"
        elif overall >= 4.0:
            verdict = "Fair"
        else:
            verdict = "Poor"

    return {
        "accuracy":     round(accuracy, 2),
        "clarity":      round(clarity, 2),
        "completeness": round(completeness, 2),
        "reasoning":    round(reasoning, 2),
        "overall":      overall,
        "verdict":      verdict,
        "summary":      raw.get("summary", ""),
        "suggestions":  raw.get("suggestions", []),
        "_evaluator":   evaluator
    }

@app.get("/")
def home():
    return {"status": "running", "evaluators": ["evaluator1", "evaluator2"]}

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    if req.evaluator not in ("evaluator1", "evaluator2"):
        return {"error": f"Unknown evaluator '{req.evaluator}'. Use evaluator1 or evaluator2."}

    try:
        response = get_groq_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": build_prompt(req.evaluator, req.domain)},
                {"role": "user",   "content": f"Domain: {req.domain}\n\nQuestion:\n{req.question}\n\nAnswer to evaluate:\n{req.answer}"}
            ],
            temperature=0.1
        )

        raw_text = response.choices[0].message.content
        print(f"=== [{req.evaluator}] GROQ RAW OUTPUT ===")
        print(raw_text)
        print("=" * 40)

        clean  = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(clean)
        except json.JSONDecodeError as je:
            return {"error": "JSON parse failed", "detail": str(je), "raw_output": raw_text}

        unified = normalize(result, req.evaluator)
        save_to_db(req.evaluator, req.domain, req.question, req.answer, unified)
        return unified

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": type(e).__name__, "detail": str(e)}

@app.get("/history")
def get_history():
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
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*)                    as total,
            ROUND(AVG(overall), 2)      as avg_overall,
            ROUND(AVG(accuracy), 2)     as avg_accuracy,
            ROUND(AVG(clarity), 2)      as avg_clarity,
            ROUND(AVG(completeness), 2) as avg_completeness,
            ROUND(AVG(reasoning), 2)    as avg_reasoning,
            MAX(overall)                as highest_score,
            MIN(overall)                as lowest_score
        FROM evaluations
    """)
    row = cursor.fetchone()
    cursor.execute("SELECT verdict, COUNT(*) as count FROM evaluations GROUP BY verdict")
    verdicts = {r[0]: r[1] for r in cursor.fetchall()}
    cursor.execute("SELECT domain, COUNT(*) as count FROM evaluations GROUP BY domain ORDER BY count DESC")
    domains = {r[0]: r[1] for r in cursor.fetchall()}
    cursor.execute("SELECT evaluator, COUNT(*) as count FROM evaluations GROUP BY evaluator")
    evaluators = {r[0]: r[1] for r in cursor.fetchall()}
    conn.close()
    return {
        "total_evaluations": row[0],
        "average_scores": {
            "overall":      row[1],
            "accuracy":     row[2],
            "clarity":      row[3],
            "completeness": row[4],
            "reasoning":    row[5]
        },
        "highest_score":       row[6],
        "lowest_score":        row[7],
        "verdicts_breakdown":  verdicts,
        "domains_breakdown":   domains,
        "evaluator_breakdown": evaluators
    }

@app.delete("/history/clear")
def clear_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM evaluations")
    conn.commit()
    conn.close()
    return {"message": "All records cleared."}
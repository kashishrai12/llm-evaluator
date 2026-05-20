from groq import Groq
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Groq()  # reads GROQ_API_KEY from environment

class EvalRequest(BaseModel):
    question: str
    answer: str
    domain: str = "Mathematics"

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
    return json.loads(clean)

@app.get("/")
def home():
    return {"status": "running"}
# LLM Response Evaluator

A REST API that evaluates the quality of AI-generated answers using an **LLM-as-judge** approach. Built with FastAPI and Groq. Given any question and an AI's answer, it returns a structured JSON rubric scoring the response across four criteria.

## What it does

- Takes a **question**, an **AI-generated answer**, and a **domain** (Math, Physics, Coding, etc.)
- Uses an LLM to score the answer on: Accuracy, Clarity, Completeness, and Reasoning Quality
- Returns a structured JSON rubric with scores, an overall verdict, a summary, and improvement suggestions

## Example

**Input:**
```json
{
  "question": "What is the derivative of x^2 + 5x + 3?",
  "answer": "The derivative is 2x + 5. Using the power rule, the derivative of x^2 is 2x, the derivative of 5x is 5, and the derivative of constant 3 is 0.",
  "domain": "Mathematics"
}
```

**Output:**
```json
{
  "accuracy": 9.0,
  "clarity": 9.0,
  "completeness": 8.0,
  "reasoning": 8.5,
  "overall": 8.5,
  "verdict": "Excellent",
  "summary": "The answer correctly applies the power rule and explains each step clearly.",
  "suggestions": [
    "Consider providing the general power rule formula",
    "Include a real-world application of derivatives",
    "Mention the linearity property of differentiation"
  ]
}
```

## Scoring Criteria

| Criterion | Weight | Description |
|---|---|---|
| Accuracy | 40% | Is the answer factually correct? |
| Clarity | 20% | Is it clear and easy to follow? |
| Completeness | 20% | Does it fully address the question? |
| Reasoning | 20% | Is the logic sound and well-justified? |

## Tech Stack

- **Python** 
- **FastAPI** — REST API framework
- **Groq API** — LLM inference (llama-3.3-70b-versatile)
- **Pydantic** — request validation

## Setup and Running

**1. Clone the repo**
```bash
git clone https://github.com/kashishrai12/llm-evaluator.git
cd llm-evaluator
```

**2. Install dependencies**
```bash
pip install groq fastapi uvicorn
```

**3. Get Groq API key**

Sign up at [console.groq.com](https://console.groq.com)

**4. Set your API key**

Mac/Linux:
```bash
export GROQ_API_KEY="your_key_here"
```

Windows:
```cmd
set GROQ_API_KEY=your_key_here
```

**5. Run the server**
```bash
uvicorn evaluator:app --reload
```

**6. Test it**

Open [http://localhost:8000/docs](http://localhost:8000/docs) (for the interactive API UI).

Or use curl:
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Newton second law?", "answer": "F = ma", "domain": "Physics"}'
```

## Supported Domains

Mathematics, Physics, Coding, Finance, Biology, Chemistry, General

## How it works

The evaluator uses an **LLM-as-judge** pattern — a second LLM call is made with a structured system prompt that instructs the model to act as a domain expert and return evaluation scores in strict JSON format. This is the same approach used in production RLHF (Reinforcement Learning from Human Feedback) pipelines to automate quality assessment of model outputs.

## API Reference

### `POST /evaluate`

| Field | Type | Required | Description |
|---|---|---|---|
| question | string | yes | The question that was asked |
| answer | string | yes | The AI-generated answer to evaluate |
| domain | string | no | Subject domain (default: Mathematics) |

### `GET /`

Health check — returns `{"status": "running"}`

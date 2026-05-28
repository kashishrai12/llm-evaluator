# LLM Response Evaluator

A dual-evaluator REST API that evaluates the quality of AI-generated answers using an **LLM-as-judge** approach. Built with FastAPI and Groq. Supports two evaluation modes — general quality scoring and hallucination detection, through a single unified interface.

## What it does

- Takes a **question**, an **AI-generated answer**, a **domain**, and an **evaluator type**
- Routes to the appropriate evaluator with its own scoring rubric and prompt
- Normalizes all outputs to a unified schema before returning to the frontend
- Returns a structured JSON rubric with scores, an overall verdict, a summary, and improvement suggestions
- Logs every evaluation to a persistent SQLite database with history and analytics endpoints

## Evaluators

### Evaluator 1 — General Quality

Scores AI answers across four general quality dimensions.

|Criterion   |Weight|Description                           |
|------------|------|--------------------------------------|
|Accuracy    |40%   |Is the answer factually correct?      |
|Clarity     |20%   |Is it clear and easy to follow?       |
|Completeness|20%   |Does it fully address the question?   |
|Reasoning   |20%   |Is the logic sound and well-justified?|

### Evaluator 2 — Hallucination Detection

Focused on detecting fabricated or misleading content.

|Criterion    |Weight|Description                                |
|-------------|------|-------------------------------------------|
|Factuality   |40%   |Is the answer factually accurate?          |
|Hallucination|30%   |Is it free from fabricated claims?         |
|Relevance    |15%   |Does it address the question?              |
|Reasoning    |15%   |Is the logic consistent and evidence-based?|

## Example

**Input:**

```json
{
  "question": "What is the derivative of x^2 + 5x + 3?",
  "answer": "The derivative is 2x + 5. Using the power rule, the derivative of x^2 is 2x, the derivative of 5x is 5, and the derivative of constant 3 is 0.",
  "domain": "Mathematics",
  "evaluator": "evaluator1"
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

## Tech Stack

- **Python**
- **FastAPI** — REST API framework
- **Groq API** — LLM inference (llama-3.3-70b-versatile)
- **SQLite** — persistent evaluation storage
- **Pydantic** — request validation
- **HTML/CSS/JavaScript** — frontend dashboard

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

**3. Get a free Groq API key**

Sign up at [console.groq.com](https://console.groq.com) — no credit card required.

**4. Set your API key**

Mac/Linux:

```bash
export GROQ_API_KEY="your_key_here"
```

Windows (Command Prompt):

```cmd
set GROQ_API_KEY=your_key_here
```

Windows (PowerShell):

```powershell
$env:GROQ_API_KEY="your_key_here"
```

**5. Run the server**

```bash
uvicorn combined:app --reload
```

**6. Open the UI**

Go to <http://localhost:8000/ui>

Or use the interactive API docs at <http://localhost:8000/docs>

## Supported Domains

Mathematics, Physics, Coding, Finance, Biology, Chemistry, General

## How it works

Each evaluator uses the **LLM-as-judge** pattern — a structured system prompt instructs the model to act as a domain expert and return raw scores in strict JSON format. A normalization layer then maps evaluator-specific field names to a unified frontend schema and computes the overall score and verdict deterministically. This decoupling means the frontend never changes regardless of how many evaluators are added.

This is the same approach used in production RLHF (Reinforcement Learning from Human Feedback) pipelines to automate quality assessment of model outputs.

## API Reference

### `POST /evaluate`

|Field    |Type  |Required|Default    |Description                        |
|---------|------|--------|-----------|-----------------------------------|
|question |string|yes     |—          |The question that was asked        |
|answer   |string|yes     |—          |The AI-generated answer to evaluate|
|domain   |string|no      |Mathematics|Subject domain                     |
|evaluator|string|no      |evaluator1 |`evaluator1` or `evaluator2`       |

### `GET /history`

Returns all past evaluations, newest first.

### `GET /history/stats`

Returns aggregate statistics: average scores, verdict breakdown, domain breakdown, evaluator breakdown.

### `DELETE /history/clear`

Clears all evaluation records.

### `GET /`

Health check — returns `{"status": "running", "evaluators": ["evaluator1", "evaluator2"]}`

## Benchmark Results

|Evaluator                            |Dataset       |Samples|Agreement Rate|Notes                                                        |
|-------------------------------------|--------------|-------|--------------|-------------------------------------------------------------|
|Evaluator 1 — General Quality        |Synthetic STEM|40     |**80%**       |6 domains: Math, Physics, Coding, Finance, Biology, Chemistry|
|Evaluator 2 — Hallucination Detection|TruthfulQA    |99     |**76.8%**     |87.9% binary accuracy · 75.5% hallucination catch rate       |

Evaluator 2 was benchmarked against [TruthfulQA](https://github.com/sylinrl/TruthfulQA), a dataset of questions where AI models commonly hallucinate. A **75.5% hallucination catch rate** means the evaluator correctly flagged 37 out of 49 factually incorrect answers as Poor or Fair. The remaining 24.5% miss rate consists primarily of fluently-written wrong answers — a known hard problem in hallucination detection research.

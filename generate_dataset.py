import json, time
from groq import Groq

client = Groq()

DOMAINS = ["Mathematics", "Physics", "Coding", "Finance", "Biology"]
QUALITIES = {
    "Excellent": """Write a complete, fully correct answer. Include:
- The exact correct result
- Step by step reasoning
- Why each step works
- Proper terminology
This must be a 9-10/10 answer with zero errors.""",

    "Good": """Write a correct answer but with a CLEAR weakness. You must do ONE of:
- Give the right answer but SKIP all explanation (just state the result)
- Give the right answer but use imprecise informal language throughout
- Give the right answer but miss one important edge case or condition
Do NOT write a thorough explanation. Score should be 7-8/10.""",

    "Fair": """Write a partially correct answer with a SIGNIFICANT flaw. You must do ONE of:
- Get the main concept right but make a clear error in the detail or formula
- Explain the wrong method but accidentally reach a correct conclusion
- Give only a vague hand-wavy answer with no actual content
Score should be 4-6/10. The flaw must be obvious.""",

    "Poor": """Write a confidently wrong answer. You must:
- State an incorrect fact as if certain
- Get the core answer completely wrong
- Do NOT accidentally say anything correct about the key concept
Score should be 0-3/10."""
}

QUESTIONS = {
    "Mathematics": [
        "What is the integral of 2x?",
        "What is the quadratic formula?",
        "What is the derivative of e^x?",
        "What is the sum of interior angles of a triangle?",
        "Define a prime number.",
        "What is the chain rule in calculus?",
        "What does it mean for a matrix to be invertible?",
        "State the Pythagorean theorem."
    ],
    "Physics": [
        "State Newton's Second Law of Motion.",
        "What is the speed of light?",
        "What is Ohm's Law?",
        "Why does the sky appear blue?",
        "What is the first law of thermodynamics?",
        "Define acceleration.",
        "What is the difference between mass and weight?",
        "What is kinetic energy?"
    ],
    "Coding": [
        "What is the time complexity of binary search?",
        "What is the time complexity of bubble sort?",
        "What does a hash map do?",
        "What is recursion?",
        "How do you reverse a string in Python?",
        "What is the difference between a stack and a queue?",
        "What does O(1) time complexity mean?",
        "What is a linked list?"
    ],
    "Finance": [
        "What is compound interest?",
        "What is the time value of money?",
        "What does NPV stand for and what does it measure?",
        "What is diversification in investing?",
        "What is inflation?",
        "Define return on investment (ROI).",
        "What is a stock?",
        "What is a bond?"
    ],
    "Biology": [
        "What is the powerhouse of the cell?",
        "What is DNA?",
        "What is photosynthesis?",
        "What is natural selection?",
        "What is the function of red blood cells?",
        "What is mitosis?",
        "What is an enzyme?",
        "What is a gene?"
    ]
}

def generate_answer(question, domain, quality_label, quality_instruction):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a student answering a {domain} question.
{quality_instruction}
Keep the answer to 2-4 sentences maximum.
Reply with ONLY the answer text, nothing else."""
            },
            {"role": "user", "content": question}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def generate_dataset():
    dataset = []
    sample_id = 1

    for domain, questions in QUESTIONS.items():
        print(f"\nGenerating: {domain}")
        # pick 2 questions per domain to keep dataset balanced
        selected = questions[:2]

        for question in selected:
            for verdict, instruction in QUALITIES.items():
                print(f"  [{sample_id}] {verdict} answer for: {question[:45]}...")
                try:
                    answer = generate_answer(question, domain, verdict, instruction)
                    dataset.append({
                        "id": sample_id,
                        "domain": domain,
                        "question": question,
                        "answer": answer,
                        "expected_verdict": verdict
                    })
                    sample_id += 1
                    time.sleep(0.4)  # avoid rate limit
                except Exception as e:
                    print(f"    ERROR: {e}")

    with open("benchmark.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDone. Generated {len(dataset)} samples → benchmark.json")

if __name__ == "__main__":
    generate_dataset()
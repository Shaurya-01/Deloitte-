import os
import uuid
import re
import json
from typing import Dict, Any, List

from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
import httpx
import asyncio

load_dotenv()

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.static_folder = 'static'
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-70b-8192"

quiz_sessions: Dict[str, Dict[str, Any]] = {}

DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced"]
QUESTION_TYPES = ["mixed", "mcq_only", "subjective_only"]

def clean_json_response(content: str) -> str:
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    start_idx = content.find('[')
    if start_idx != -1:
        content = content[start_idx:]
    end_idx = content.rfind(']')
    if end_idx != -1:
        content = content[:end_idx + 1]
    content = re.sub(r'"\s*\n\s*}', '"\n  }', content)
    content = re.sub(r'"\s*\n\s*]', '"\n  ]', content)
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)
    return content.strip()

def is_python_mcq_ambiguous(question: str, options: List[str], answer: str) -> bool:
    python_types = {"int", "float", "str", "list", "tuple", "set", "dict", "bool", "complex", "bytes", "frozenset", "bytearray", "range"}
    corrects = [opt for opt in options if opt.lower() in python_types]
    ambiguous_keywords = ["built-in type", "data type", "python type", "datatype"]
    if any(kw in question.lower() for kw in ambiguous_keywords) and len(corrects) > 1:
        return True
    return False

def validate_and_fix_questions(questions_data: List[Dict], prev_all_questions: List[str]=None) -> List[Dict]:
    fixed_questions = []
    seen_questions = set(prev_all_questions or [])
    for q in questions_data:
        if not isinstance(q, dict):
            continue
        if "question" not in q or "type" not in q or "answer" not in q:
            continue
        question_type = q.get("type", "").lower()
        if question_type not in ["mcq", "subjective"]:
            continue
        question_text = str(q["question"]).strip()
        if question_text in seen_questions:
            continue
        seen_questions.add(question_text)
        clean_question = {
            "type": question_type,
            "question": question_text,
            "answer": str(q["answer"]).strip(),
            "points": int(q.get("points", 2 if question_type == "mcq" else 4))
        }
        if question_type == "mcq":
            options = q.get("options", [])
            if not isinstance(options, list) or len(options) != 4:
                continue
            clean_options = [str(opt).strip() for opt in options]
            clean_question["options"] = clean_options
            if clean_question["answer"] not in clean_options:
                continue
            if is_python_mcq_ambiguous(question_text, clean_options, clean_question["answer"]):
                continue
        else:
            clean_question["options"] = None
            if not clean_question["answer"] or clean_question["answer"].lower() in ["", "n/a", "none"]:
                continue
        fixed_questions.append(clean_question)
    return fixed_questions

def generate_mixed_questions_prompt(topic, difficulty, all_prev_questions=None):
    difficulty_specs = {
        "beginner": "basic understanding, simple concepts, foundational knowledge",
        "intermediate": "moderate complexity, real-world scenarios, integration of concepts",
        "advanced": "complex analysis, critical thinking, expert-level insights"
    }
    spec = difficulty_specs.get(difficulty, "appropriate level")
    uniqueness = ""
    if all_prev_questions:
        uniqueness = (
            "\nAVOID REPETITION:\n"
            "Do NOT repeat any of these questions (or very similar ones):\n"
            + "\n".join(f"- {q}" for q in all_prev_questions if q)
            + "\n"
        )
    return f"""
You are a master quiz maker. Generate EXACTLY 5 unique, high-quality questions on "{topic}" for {difficulty} level ({spec}).

MANDATORY REQUIREMENTS:
- 2-3 MCQs and 2-3 Subjective (open-ended) questions, total exactly 5.
- MCQs: Each must have 4 plausible options and **exactly one unambiguously correct answer**; the other three must be clearly incorrect. Do NOT use questions where multiple options could be correct.
- Subjective: Must require analysis, synthesis, evaluation, creativity, and real-world application.
- Avoid simple recall or yes/no. All questions must be unique, non-trivial, and increase in cognitive demand with each level.
- Return ONLY a valid JSON array, NO explanations or markdown.

{uniqueness}

DIFFICULTY-SPECIFIC GUIDELINES:
{difficulty.title()} Level:
- {"Beginner: focus on core concepts and basic applications." if difficulty=="beginner" else
   "Intermediate: application, scenario-based, integration of concepts." if difficulty=="intermediate" else
   "Advanced: critical thinking, analysis, creative synthesis."}

FORMAT:
[
    {{"type": "mcq", "question": "...", "options": ["A", "B", "C", "D"], "answer": "...", "points": 2}},
    {{"type": "subjective", "question": "...", "options": null, "answer": "...", "points": 4}}
]

TOPIC: {topic}
LEVEL: {difficulty}
COUNT: Exactly 5 questions (verify count before responding)
"""

def generate_mcq_only_prompt(topic, difficulty, all_prev_questions=None):
    difficulty_specs = {
        "beginner": "foundational concepts, basic terminology, simple applications",
        "intermediate": "practical scenarios, moderate complexity, applied knowledge, integration of concepts", 
        "advanced": "complex scenarios, expert-level analysis, nuanced understanding, require reasoning"
    }
    uniqueness = ""
    if all_prev_questions:
        uniqueness = (
            "\nAVOID REPEATS:\n"
            "Do NOT repeat any of these questions (or similar):\n"
            + "\n".join(f"- {q}" for q in all_prev_questions if q)
            + "\n"
        )
    spec = difficulty_specs.get(difficulty, "appropriate level")
    return f"""
Create EXACTLY 5 unique, high-quality multiple-choice questions (MCQs) on "{topic}" for {difficulty} level.

{uniqueness}

REQUIREMENTS:
- Test conceptual understanding and application, not just memorization.
- Use realistic, scenario-based questions where possible.
- Each MCQ must have 4 plausible options and **exactly one unambiguously correct answer; the other three must be clearly incorrect**. Avoid any question where more than one answer could be correct (e.g., avoid “Which of these is a built-in type in Python?”).
- Do not use questions where multiple options could be reasonably correct.
- Use common misconceptions as distractors.
- Vary question stems (What, Why, How, Which, When).
- Avoid trivial, ambiguous, or repeated questions.

DIFFICULTY CALIBRATION ({difficulty.title()}):
{spec}

FORMAT: JSON array, exactly 5 MCQs
[{{"type": "mcq", "question": "...", "options": ["A", "B", "C", "D"], "answer": "...", "points": 2}}]

TOPIC: {topic}
VERIFY: Count must be exactly 5 and all must be unique and unambiguous.
"""

def generate_subjective_only_prompt(topic, difficulty, all_prev_questions=None):
    bloom_levels = {
        "beginner": "Understanding and Application - explain, describe, demonstrate, apply basic concepts",
        "intermediate": "Analysis and Synthesis - analyze, compare, contrast, organize, integrate concepts in real contexts",
        "advanced": "Evaluation and Creation - evaluate, critique, design, propose, create new solutions, integrate multiple topics"
    }
    uniqueness = ""
    if all_prev_questions:
        uniqueness = (
            "\nAVOID REPEATS:\n"
            "Do NOT repeat any of these questions (or similar):\n"
            + "\n".join(f"- {q}" for q in all_prev_questions if q)
            + "\n"
        )
    bloom_level = bloom_levels.get(difficulty, "appropriate cognitive level")
    return f"""
Generate EXACTLY 5 unique, high-quality subjective (open-ended) questions on "{topic}" for {difficulty} level.

{uniqueness}

COGNITIVE DEPTH REQUIRED ({difficulty.title()}):
{bloom_level}

REQUIREMENTS:
- Require extended, thoughtful responses (not one-word).
- Must demand critical thinking, deep analysis, and real-world application.
- NO repeats, NO simple recall.
- Each question must require multiple key points and perspectives.

MODEL ANSWER REQUIREMENTS:
- Provide comprehensive, well-structured sample answers showing depth, reasoning, and multiple viewpoints.

FORMAT: JSON array, exactly 5 subjective objects
[{{"type": "subjective", "question": "...", "options": null, "answer": "...", "points": 4}}]

TOPIC: {topic}
COGNITIVE LEVEL: {difficulty}
VERIFY: Must be exactly 5, all unique and non-repetitive.
"""

def evaluate_subjective_answer_prompt(question, model_answer, user_answer, points):
    return f"""
You are an expert educator evaluating a subjective answer. Use a comprehensive rubric to assess the response fairly.

RUBRIC:
- Content Knowledge (40%): Accuracy and depth of subject matter understanding
- Critical Thinking (30%): Analysis, reasoning, and insight
- Communication (20%): Clarity, organization, and coherence
- Examples/Evidence (10%): Relevant examples or supporting evidence

SCORING GUIDELINES (strict, no grade inflation):
{points} points (90-100%): Exceptional - Comprehensive, insightful, well-reasoned, excellent examples
{points-1} points (80-89%): Proficient - Good understanding, clear reasoning, adequate examples
{points-2} points (70-79%): Developing - Basic understanding, some reasoning, limited examples
{points-3} points (60-69%): Beginning - Minimal understanding, weak reasoning, poor examples
0-{max(0,points-4)} points (0-59%): Inadequate - Little to no understanding or reasoning

QUESTION: {question}
MODEL ANSWER: {model_answer}
STUDENT ANSWER: {user_answer}

EVALUATION CRITERIA:
1. Does the answer demonstrate understanding of key concepts?
2. Is there evidence of critical thinking and analysis?
3. Are ideas communicated clearly and logically?
4. Are examples or evidence provided?
5. Does the response address the question comprehensively?

Return ONLY this JSON:
{{"score": number_0_to_{points}, "feedback": "specific_constructive_feedback", "percentage": percentage_number, "rubric_breakdown": {{"content": score_out_of_40, "thinking": score_out_of_30, "communication": score_out_of_20, "examples": score_out_of_10}}}}
"""

def run_async(coro):
    return asyncio.run(coro)

async def fetch_questions(topic: str, difficulty: str, question_type: str = "mixed", retry_count: int = 0, all_prev_questions: List[str]=None):
    if retry_count >= 3:
        return generate_fallback_questions(topic, difficulty, question_type, all_prev_questions)
    if question_type == "mcq_only":
        prompt = generate_mcq_only_prompt(topic, difficulty, all_prev_questions)
    elif question_type == "subjective_only":
        prompt = generate_subjective_only_prompt(topic, difficulty, all_prev_questions)
    else:
        prompt = generate_mixed_questions_prompt(topic, difficulty, all_prev_questions)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.25 if difficulty == "advanced" else 0.35 if difficulty == "intermediate" else 0.4,
        "max_tokens": 1700,
    }
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers
            )
            if response.status_code != 200:
                raise Exception(f"GROQ API error: {response.status_code}")
            data = response.json()
            if "choices" not in data or not data["choices"]:
                raise Exception("No choices in API response")
            content = data["choices"][0]["message"]["content"]
            cleaned_content = clean_json_response(content)
            try:
                questions_data = json.loads(cleaned_content)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', cleaned_content, re.DOTALL)
                if json_match:
                    potential_json = json_match.group(0)
                    try:
                        questions_data = json.loads(potential_json)
                    except json.JSONDecodeError:
                        return await fetch_questions(topic, difficulty, question_type, retry_count + 1, all_prev_questions)
                else:
                    raise Exception("Could not extract valid JSON from response")
            if not isinstance(questions_data, list):
                raise Exception(f"Expected list, got {type(questions_data)}")
            fixed_questions = validate_and_fix_questions(questions_data, all_prev_questions)
            if len(fixed_questions) < 5:
                needed_questions = 5 - len(fixed_questions)
                additional_questions = generate_additional_questions(
                    topic, difficulty, question_type, needed_count=needed_questions,
                    existing_questions=fixed_questions, prev_all_questions=all_prev_questions
                )
                fixed_questions.extend(additional_questions)
            elif len(fixed_questions) > 5:
                fixed_questions = fixed_questions[:5]
            if len(fixed_questions) != 5:
                if retry_count < 2:
                    return await fetch_questions(topic, difficulty, question_type, retry_count + 1, all_prev_questions)
                else:
                    return generate_fallback_questions(topic, difficulty, question_type, all_prev_questions)
            return fixed_questions
    except Exception:
        if retry_count < 2:
            return await fetch_questions(topic, difficulty, question_type, retry_count + 1, all_prev_questions)
        else:
            return generate_fallback_questions(topic, difficulty, question_type, all_prev_questions)

def generate_additional_questions(topic: str, difficulty: str, question_type: str, needed_count: int, existing_questions: List[Dict], prev_all_questions: List[str]=None) -> List[Dict]:
    additional_questions = []
    existing_mcq_count = sum(1 for q in existing_questions if q["type"] == "mcq")
    used_questions = set(prev_all_questions or [])
    used_questions.update(q["question"] for q in existing_questions)
    if "python" in topic.lower():
        mcq_templates = [
            {
                "question": f"What is the output of print(2 + 3 * 2) in Python?",
                "options": ["8", "10", "12", "7"],
                "answer": "8"
            },
            {
                "question": f"Which of these is a valid Python variable name?",
                "options": ["my_var1", "1st_var", "var-2", "my var"],
                "answer": "my_var1"
            },
            {
                "question": f"What will be the result of: my_list = [1, 2, 3]; print(my_list[-1])?",
                "options": ["1", "3", "2", "Error"],
                "answer": "3"
            },
            {
                "question": f"What is the correct way to define a function in Python?",
                "options": ["def myfunc():", "function myfunc():", "define myfunc():", "func myfunc():"],
                "answer": "def myfunc():"
            },
            {
                "question": f"What will this code print: for i in range(3): print(i)?",
                "options": ["0 1 2", "1 2 3", "0 1 2 3", "1 2"],
                "answer": "0 1 2"
            },
        ]
        subjective_templates = [
            f"Explain the difference between a list and a tuple in Python.",
            f"Describe how you would handle errors in a Python program.",
            f"Discuss the importance of indentation in Python.",
            f"Describe how to use a for loop to iterate over a list in Python.",
            f"Explain what a dictionary is in Python and give an example."
        ]
    else:
        mcq_templates = [
            {
                "question": f"What is a key concept in {topic}?",
                "options": [f"A correct concept", f"Incorrect concept 1", f"Incorrect concept 2", f"Incorrect concept 3"],
                "answer": f"A correct concept"
            },
            {
                "question": f"Which of the following best explains {topic}?",
                "options": [f"Accurate summary", f"Misconception", f"Partial truth", f"Unrelated fact"],
                "answer": f"Accurate summary"
            },
            {
                "question": f"What is a common application of {topic}?",
                "options": [f"Real-world use", f"Not an application", f"Rarely used", f"Incorrect use"],
                "answer": f"Real-world use"
            },
            {
                "question": f"Which statement about {topic} is false?",
                "options": [f"True fact", f"False statement", f"Another fact", f"Another false"],
                "answer": f"False statement"
            },
            {
                "question": f"What is the main advantage of {topic}?",
                "options": [f"Correct advantage", f"Incorrect", f"Irrelevant", f"Partial"],
                "answer": f"Correct advantage"
            },
        ]
        subjective_templates = [
            f"Explain an important aspect of {topic} and why it matters.",
            f"Discuss challenges faced in {topic} and how to address them.",
            f"Describe a real-world scenario where {topic} is applied.",
            f"Compare {topic} to a related concept.",
            f"Predict how {topic} might evolve in the future."
        ]
    for i in range(needed_count):
        found = False
        if question_type == "mcq_only" or (question_type == "mixed" and (existing_mcq_count + len([q for q in additional_questions if q["type"] == "mcq"])) < 3):
            for template in mcq_templates:
                if template["question"] not in used_questions:
                    additional_questions.append({
                        "type": "mcq",
                        "question": template["question"],
                        "options": template["options"],
                        "answer": template["answer"],
                        "points": 2
                    })
                    used_questions.add(template["question"])
                    found = True
                    break
        else:
            for template in subjective_templates:
                if template not in used_questions:
                    additional_questions.append({
                        "type": "subjective",
                        "question": template,
                        "options": None,
                        "answer": f"A strong answer explains the topic with clear examples and reasoning.",
                        "points": 4
                    })
                    used_questions.add(template)
                    found = True
                    break
        if not found:
            break
    return additional_questions

def generate_fallback_questions(topic: str, difficulty: str, question_type: str, prev_all_questions: List[str]=None) -> List[Dict]:
    fallback_questions = []
    used_questions = set(prev_all_questions or [])
    fallback_mcqs = [
        {
            "question": f"Fallback MCQ: Which statement best describes {topic}?",
            "options": ["A broad overview", "A limited view", "A misconception", "An unrelated fact"],
            "answer": "A broad overview"
        },
        {
            "question": f"Fallback MCQ: What would be a key application of {topic} in real life?",
            "options": ["Correct use", "Incorrect use", "Irrelevant use", "Uncommon use"],
            "answer": "Correct use"
        }
    ]
    fallback_subjectives = [
        f"Fallback: Discuss why {topic} is important for {difficulty} learners.",
        f"Fallback: Analyze potential drawbacks and benefits of {topic}.",
        f"Fallback: Predict future developments in {topic} and their possible impact."
    ]
    if question_type in ["mixed", "mcq_only"]:
        for fallback in fallback_mcqs:
            if fallback["question"] not in used_questions:
                fallback_questions.append({
                    "type": "mcq",
                    "question": fallback["question"],
                    "options": fallback["options"],
                    "answer": fallback["answer"],
                    "points": 2
                })
                used_questions.add(fallback["question"])
    if question_type in ["mixed", "subjective_only"]:
        for fallback in fallback_subjectives:
            if fallback not in used_questions:
                fallback_questions.append({
                    "type": "subjective",
                    "question": fallback,
                    "options": None,
                    "answer": f"A strong answer covers all aspects with examples.",
                    "points": 4
                })
                used_questions.add(fallback)
    return fallback_questions[:5]

async def evaluate_subjective_answer(question: str, model_answer: str, user_answer: str, max_points: int):
    if not user_answer or not user_answer.strip():
        return {
            "score": 0, 
            "feedback": "No answer provided. Please provide a response to demonstrate your understanding.", 
            "percentage": 0,
            "rubric_breakdown": {"content": 0, "thinking": 0, "communication": 0, "examples": 0}
        }
    prompt = evaluate_subjective_answer_prompt(question, model_answer, user_answer, max_points)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.15,
        "max_tokens": 450,
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers
            )
            if response.status_code != 200:
                return apply_fallback_scoring(user_answer, max_points)
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            cleaned_content = clean_json_response(content)
            try:
                evaluation = json.loads(cleaned_content)
                score = max(0, min(int(evaluation.get("score", 0)), max_points))
                feedback = str(evaluation.get("feedback", "Evaluated automatically"))
                percentage = (score / max_points) * 100 if max_points > 0 else 0
                rubric_breakdown = evaluation.get("rubric_breakdown", {})
                if not isinstance(rubric_breakdown, dict):
                    rubric_breakdown = generate_rubric_breakdown(user_answer, score, max_points)
                return {
                    "score": score,
                    "feedback": feedback,
                    "percentage": round(percentage, 1),
                    "rubric_breakdown": rubric_breakdown
                }
            except (json.JSONDecodeError, ValueError, KeyError):
                return apply_fallback_scoring(user_answer, max_points)
    except Exception:
        return apply_fallback_scoring(user_answer, max_points)

def apply_fallback_scoring(user_answer: str, max_points: int):
    answer_words = user_answer.strip().split()
    answer_length = len(answer_words)
    content_score = 0
    thinking_score = 0 
    communication_score = 0
    examples_score = 0
    if answer_length >= 50:
        content_score = 35
    elif answer_length >= 25:
        content_score = 28
    elif answer_length >= 10:
        content_score = 20
    else:
        content_score = 10
    analytical_words = ['because', 'therefore', 'however', 'although', 'compare', 'contrast', 'analyze', 'evaluate', 'consider', 'furthermore', 'moreover', 'consequently']
    thinking_indicators = sum(1 for word in analytical_words if word.lower() in user_answer.lower())
    if thinking_indicators >= 3:
        thinking_score = 25
    elif thinking_indicators >= 2:
        thinking_score = 20
    elif thinking_indicators >= 1:
        thinking_score = 15
    else:
        thinking_score = 10
    sentences = len([s for s in user_answer.split('.') if s.strip()])
    if sentences >= 4 and answer_length >= 30:
        communication_score = 18
    elif sentences >= 2 and answer_length >= 15:
        communication_score = 14
    else:
        communication_score = 8
    example_words = ['example', 'for instance', 'such as', 'like', 'including', 'specifically']
    example_indicators = sum(1 for word in example_words if word.lower() in user_answer.lower())
    if example_indicators >= 2:
        examples_score = 9
    elif example_indicators >= 1:
        examples_score = 6
    else:
        examples_score = 3
    total_percentage = content_score + thinking_score + communication_score + examples_score
    final_score = int((total_percentage / 100) * max_points)
    feedback_parts = []
    if content_score < 25:
        feedback_parts.append("Consider providing more detailed content")
    if thinking_score < 20:
        feedback_parts.append("Include more analytical thinking and reasoning")
    if communication_score < 15:
        feedback_parts.append("Work on organizing ideas into clear sentences")
    if examples_score < 6:
        feedback_parts.append("Add specific examples to support your points")
    if not feedback_parts:
        feedback = "Good comprehensive response with clear reasoning and examples."
    else:
        feedback = "Suggestions for improvement: " + "; ".join(feedback_parts) + "."
    return {
        "score": final_score,
        "feedback": feedback,
        "percentage": round(total_percentage, 1),
        "rubric_breakdown": {
            "content": content_score,
            "thinking": thinking_score, 
            "communication": communication_score,
            "examples": examples_score
        }
    }

def generate_rubric_breakdown(user_answer: str, score: int, max_points: int):
    percentage = (score / max_points) * 100 if max_points > 0 else 0
    return {
        "content": int(percentage * 0.4),
        "thinking": int(percentage * 0.3),
        "communication": int(percentage * 0.2),
        "examples": int(percentage * 0.1)
    }

def generate_session_id():
    return str(uuid.uuid4())

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/start_quiz", methods=["POST"])
def start_quiz():
    topic = request.form.get("topic", "")
    question_type = request.form.get("question_type", "mixed")
    if question_type not in QUESTION_TYPES:
        question_type = "mixed"
    if not topic or len(topic.strip()) < 2:
        return jsonify({"error": "Please provide a valid topic (at least 2 characters)"}), 400
    session_id = generate_session_id()
    quiz_sessions[session_id] = {
        "topic": topic.strip(),
        "question_type": question_type,
        "current_level": 0,
        "current_questions": [],
        "user_answers": [],
        "subjective_evaluations": [],
        "scores": {"beginner": 0, "intermediate": 0, "advanced": 0},
        "completed_levels": [],
        "all_questions": [],
        "quiz_completed": False
    }
    questions = run_async(fetch_questions(topic.strip(), DIFFICULTY_LEVELS[0], question_type, 0, []))
    quiz_sessions[session_id]["current_questions"] = questions
    quiz_sessions[session_id]["user_answers"] = [None] * len(questions)
    quiz_sessions[session_id]["subjective_evaluations"] = [None] * len(questions)
    quiz_sessions[session_id]["all_questions"] = [q["question"] for q in questions]
    return jsonify({
        "session_id": session_id,
        "level": DIFFICULTY_LEVELS[0],
        "question_type": question_type,
        "questions": questions,
        "message": f"Starting {DIFFICULTY_LEVELS[0]} level quiz on {topic} ({question_type.replace('_', ' ')} questions)"
    })

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    session_id = request.form.get("session_id")
    question_index = int(request.form.get("question_index", -1))
    answer = request.form.get("answer", "")
    if not session_id or session_id not in quiz_sessions:
        return jsonify({"error": "Invalid session"}), 400
    session = quiz_sessions[session_id]
    if question_index >= len(session["current_questions"]) or question_index < 0:
        return jsonify({"error": "Invalid question index"}), 400
    session["user_answers"][question_index] = answer.strip() if answer else ""
    return jsonify({"success": True})

@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    session_id = request.form.get("session_id")
    if not session_id or session_id not in quiz_sessions:
        return jsonify({"error": "Invalid session"}), 400
    session = quiz_sessions[session_id]
    current_level_idx = session["current_level"]
    current_level = DIFFICULTY_LEVELS[current_level_idx]
    questions = session["current_questions"]
    user_answers = session["user_answers"]
    if not questions:
        return jsonify({"error": "No questions found for this session"}), 400
    total_score = 0
    max_possible_score = 0
    detailed_results = []
    for i, question in enumerate(questions):
        max_possible_score += question["points"]
        user_answer = user_answers[i] if i < len(user_answers) else None
        if question["type"] == "mcq":
            if user_answer == question["answer"]:
                score = question["points"]
                is_correct = True
                feedback = "Correct!"
            else:
                score = 0
                is_correct = False
                feedback = f"Incorrect. Correct answer: {question['answer']}"
            detailed_results.append({
                "question_index": i,
                "type": "mcq",
                "score": score,
                "max_score": question["points"],
                "is_correct": is_correct,
                "feedback": feedback,
                "correct_answer": question["answer"],
                "user_answer": user_answer
            })
        elif question["type"] == "subjective":
            evaluation = run_async(evaluate_subjective_answer(
                question["question"],
                question["answer"],
                user_answer or "",
                question["points"]
            ))
            score = evaluation["score"]
            feedback = evaluation["feedback"]
            session["subjective_evaluations"][i] = evaluation
            detailed_results.append({
                "question_index": i,
                "type": "subjective",
                "score": score,
                "max_score": question["points"],
                "feedback": feedback,
                "model_answer": question["answer"],
                "user_answer": user_answer,
                "evaluation": evaluation
            })
        total_score += score

    score_percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    session["scores"][current_level] = score_percentage

    session["all_questions"].extend([q["question"] for q in questions if q["question"] not in session["all_questions"]])

    if score_percentage > 50:
        session["completed_levels"].append(current_level)
        if session["current_level"] < len(DIFFICULTY_LEVELS) - 1:
            session["current_level"] += 1
            next_level = DIFFICULTY_LEVELS[session["current_level"]]
            questions = run_async(fetch_questions(
                session["topic"], next_level, session["question_type"], 0, session["all_questions"]
            ))
            quiz_sessions[session_id]["current_questions"] = questions
            quiz_sessions[session_id]["user_answers"] = [None] * len(questions)
            quiz_sessions[session_id]["subjective_evaluations"] = [None] * len(questions)
            quiz_sessions[session_id]["all_questions"].extend([q["question"] for q in questions if q["question"] not in quiz_sessions[session_id]["all_questions"]])
            return jsonify({
                "passed": True,
                "score": round(score_percentage, 1),
                "total_score": total_score,
                "max_score": max_possible_score,
                "detailed_results": detailed_results,
                "next_level": next_level,
                "questions": questions,
                "message": f"Congratulations! You scored {score_percentage:.1f}% ({total_score}/{max_possible_score} points). Moving to {next_level} level."
            })
        else:
            quiz_sessions[session_id]["quiz_completed"] = True
            return jsonify({
                "passed": True,
                "score": round(score_percentage, 1),
                "total_score": total_score,
                "max_score": max_possible_score,
                "detailed_results": detailed_results,
                "quiz_completed": True,
                "final_scores": quiz_sessions[session_id]["scores"],
                "message": f"Congratulations! You've completed all levels with a final score of {score_percentage:.1f}% ({total_score}/{max_possible_score} points)!"
            })
    else:
        questions = run_async(fetch_questions(
            session["topic"], current_level, session["question_type"], 0, session["all_questions"]
        ))
        quiz_sessions[session_id]["current_questions"] = questions
        quiz_sessions[session_id]["user_answers"] = [None] * len(questions)
        quiz_sessions[session_id]["subjective_evaluations"] = [None] * len(questions)
        quiz_sessions[session_id]["all_questions"].extend([q["question"] for q in questions if q["question"] not in quiz_sessions[session_id]["all_questions"]])
        return jsonify({
            "passed": False,
            "score": round(score_percentage, 1),
            "total_score": total_score,
            "max_score": max_possible_score,
            "detailed_results": detailed_results,
            "retry_level": current_level,
            "questions": questions,
            "message": f"You scored {score_percentage:.1f}% ({total_score}/{max_possible_score} points). You need >50% to advance. Try again with new questions!"
        })

@app.route("/quiz_status/<session_id>", methods=["GET"])
def get_quiz_status(session_id):
    if session_id not in quiz_sessions:
        return jsonify({"error": "Invalid session"}), 400
    session = quiz_sessions[session_id]
    current_level = DIFFICULTY_LEVELS[session["current_level"]]
    return jsonify({
        "topic": session["topic"],
        "question_type": session["question_type"],
        "current_level": current_level,
        "completed_levels": session["completed_levels"],
        "scores": session["scores"],
        "quiz_completed": session["quiz_completed"],
        "questions": session["current_questions"],
        "subjective_evaluations": session.get("subjective_evaluations", [])
    })

@app.route("/quiz_session/<session_id>", methods=["DELETE"])
def delete_quiz_session(session_id):
    if session_id in quiz_sessions:
        del quiz_sessions[session_id]
        return jsonify({"message": "Session deleted"})
    return jsonify({"error": "Session not found"}), 404

@app.route("/question_types", methods=["GET"])
def get_question_types():
    return jsonify({
        "question_types": QUESTION_TYPES,
        "descriptions": {
            "mixed": "Mix of multiple-choice and subjective questions",
            "mcq_only": "Only multiple-choice questions",
            "subjective_only": "Only subjective/essay questions"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "active_sessions": len(quiz_sessions),
        "api_key_configured": bool(GROQ_API_KEY)
    })

if __name__ == "__main__":
    app.run(debug=True)

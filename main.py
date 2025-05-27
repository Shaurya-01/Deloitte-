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

# ------------------ Helpers (copied from FastAPI code) ------------------

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

def validate_and_fix_questions(questions_data: List[Dict]) -> List[Dict]:
    fixed_questions = []
    for i, q in enumerate(questions_data):
        if not isinstance(q, dict):
            continue
        if "question" not in q or "type" not in q or "answer" not in q:
            continue
        question_type = q.get("type", "").lower()
        if question_type not in ["mcq", "subjective"]:
            continue
        clean_question = {
            "type": question_type,
            "question": str(q["question"]).strip(),
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
        else:
            clean_question["options"] = None
        fixed_questions.append(clean_question)
    return fixed_questions

def generate_mixed_questions_prompt(topic, difficulty):
    difficulty_specs = {
        "beginner": "basic understanding, simple concepts, foundational knowledge",
        "intermediate": "moderate complexity, practical applications, connecting concepts", 
        "advanced": "complex analysis, critical thinking, expert-level insights"
    }
    spec = difficulty_specs.get(difficulty, "appropriate level")
    return f"""
You are an expert educator creating a high-quality quiz. Generate EXACTLY 5 questions on "{topic}" for {difficulty} level ({spec}).

MANDATORY REQUIREMENTS:
- Exactly 5 questions total (2-3 MCQs + 2-3 subjective)
- Return ONLY valid JSON array, no explanations
- Ensure questions are pedagogically sound and engaging

MCQ QUALITY STANDARDS:
- Test genuine understanding, not just memorization
- Create plausible distractors (wrong options that seem reasonable)
- Avoid "all of the above" or "none of the above" options
- Make questions scenario-based when possible

SUBJECTIVE QUALITY STANDARDS:
- Require analysis, synthesis, evaluation, or creativity
- Use Bloom's taxonomy higher-order thinking skills
- Include real-world applications and critical thinking
- Avoid simple recall or yes/no questions

DIFFICULTY-SPECIFIC GUIDELINES:
{difficulty.title()} Level:
- {"Focus on fundamental concepts and basic applications" if difficulty == "beginner" else 
  "Include practical scenarios and moderate complexity" if difficulty == "intermediate" else
  "Require deep analysis, evaluation, and expert-level reasoning"}

FORMAT:
[
  {{"type": "mcq", "question": "scenario_based_question", "options": ["A", "B", "C", "D"], "answer": "correct_option", "points": 2}},
  {{"type": "subjective", "question": "analytical_question", "options": null, "answer": "comprehensive_model_answer", "points": 4}}
]

TOPIC: {topic}
LEVEL: {difficulty}
COUNT: Exactly 5 questions (verify count before responding)
"""

def generate_mcq_only_prompt(topic, difficulty):
    difficulty_specs = {
        "beginner": "foundational concepts, basic terminology, simple applications",
        "intermediate": "practical scenarios, moderate complexity, applied knowledge", 
        "advanced": "complex scenarios, expert-level analysis, nuanced understanding"
    }
    spec = difficulty_specs.get(difficulty, "appropriate level")
    return f"""
Create EXACTLY 5 high-quality multiple-choice questions on "{topic}" for {difficulty} level.

QUALITY REQUIREMENTS:
- Test conceptual understanding, not just memorization
- Use realistic scenarios and practical applications
- Create believable distractors (wrong answers that seem plausible)
- Vary question stems (What, Why, How, Which, When used appropriately)
- Avoid trivial or ambiguous questions

DIFFICULTY CALIBRATION ({difficulty.title()}):
{spec}

DISTRACTOR GUIDELINES:
- Make wrong options reasonable but clearly incorrect to experts
- Avoid obviously wrong or silly options
- Include common misconceptions as distractors
- Ensure only one clearly correct answer

FORMAT: JSON array with exactly 5 MCQ objects
[{{"type": "mcq", "question": "well_crafted_question", "options": ["A", "B", "C", "D"], "answer": "correct_option", "points": 2}}]

TOPIC: {topic}
VERIFY: Count must be exactly 5 questions
"""

def generate_subjective_only_prompt(topic, difficulty):
    bloom_levels = {
        "beginner": "Understanding and Application - explain, describe, demonstrate, apply basic concepts",
        "intermediate": "Analysis and Synthesis - analyze, compare, contrast, organize, integrate concepts",
        "advanced": "Evaluation and Creation - evaluate, critique, design, propose, create new solutions"
    }
    bloom_level = bloom_levels.get(difficulty, "appropriate cognitive level")
    return f"""
Generate EXACTLY 5 high-quality subjective questions on "{topic}" for {difficulty} level.

COGNITIVE DEPTH REQUIRED ({difficulty.title()}):
{bloom_level}

QUESTION QUALITY STANDARDS:
- Require extended responses (not one-word answers)
- Promote critical thinking and deep analysis  
- Include real-world applications and scenarios
- Encourage personal reflection and professional insight
- Avoid questions with single "correct" answers

QUESTION TYPES TO INCLUDE:
1. Analytical questions (Why/How analysis)
2. Comparative questions (Compare/Contrast)
3. Evaluative questions (Assess/Judge/Critique)
4. Application questions (Real-world scenarios)
5. Reflective questions (Personal insights/implications)

MODEL ANSWER REQUIREMENTS:
- Provide comprehensive, well-structured sample answers
- Include multiple key points and perspectives
- Demonstrate the depth expected from students
- Show clear reasoning and examples

FORMAT: JSON array with exactly 5 subjective objects
[{{"type": "subjective", "question": "thought_provoking_question", "options": null, "answer": "comprehensive_model_answer_with_multiple_points", "points": 4}}]

TOPIC: {topic}
COGNITIVE LEVEL: {difficulty}
VERIFY: Must be exactly 5 questions
"""

def evaluate_subjective_answer_prompt(question, model_answer, user_answer, points):
    return f"""
You are an expert educator evaluating a subjective answer. Use a comprehensive rubric to assess the response fairly.

EVALUATION RUBRIC:
- Content Knowledge (40%): Accuracy and depth of subject matter understanding
- Critical Thinking (30%): Analysis, reasoning, and insight demonstrated  
- Communication (20%): Clarity, organization, and coherence of response
- Examples/Evidence (10%): Use of relevant examples or supporting evidence

SCORING GUIDELINES:
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
4. Are examples or evidence provided to support points?
5. Does the response address the question comprehensively?

Return ONLY this JSON format:
{{"score": number_0_to_{points}, "feedback": "specific_constructive_feedback_with_strengths_and_improvements", "percentage": percentage_number, "rubric_breakdown": {{"content": score_out_of_40, "thinking": score_out_of_30, "communication": score_out_of_20, "examples": score_out_of_10}}}}

Provide specific, actionable feedback that helps the student improve.
"""

def run_async(coro):
    return asyncio.run(coro)

async def fetch_questions(topic: str, difficulty: str, question_type: str = "mixed", retry_count: int = 0):
    if retry_count >= 3:
        return generate_fallback_questions(topic, difficulty, question_type)
    if question_type == "mcq_only":
        prompt = generate_mcq_only_prompt(topic, difficulty)
    elif question_type == "subjective_only":
        prompt = generate_subjective_only_prompt(topic, difficulty)
    else:
        prompt = generate_mixed_questions_prompt(topic, difficulty)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 1500,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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
                        return await fetch_questions(topic, difficulty, question_type, retry_count + 1)
                else:
                    raise Exception("Could not extract valid JSON from response")
            if not isinstance(questions_data, list):
                raise Exception(f"Expected list, got {type(questions_data)}")
            fixed_questions = validate_and_fix_questions(questions_data)
            if len(fixed_questions) < 5:
                needed_questions = 5 - len(fixed_questions)
                additional_questions = generate_additional_questions(
                    topic, difficulty, question_type, needed_questions, fixed_questions
                )
                fixed_questions.extend(additional_questions)
            elif len(fixed_questions) > 5:
                fixed_questions = fixed_questions[:5]
            if len(fixed_questions) != 5:
                if retry_count < 2:
                    return await fetch_questions(topic, difficulty, question_type, retry_count + 1)
                else:
                    return generate_fallback_questions(topic, difficulty, question_type)
            return fixed_questions
    except Exception:
        if retry_count < 2:
            return await fetch_questions(topic, difficulty, question_type, retry_count + 1)
        else:
            return generate_fallback_questions(topic, difficulty, question_type)

def generate_additional_questions(topic: str, difficulty: str, question_type: str, needed_count: int, existing_questions: List[Dict]) -> List[Dict]:
    additional_questions = []
    existing_mcq_count = sum(1 for q in existing_questions if q["type"] == "mcq")
    existing_subjective_count = sum(1 for q in existing_questions if q["type"] == "subjective")
    for i in range(needed_count):
        if question_type == "mcq_only":
            additional_questions.append({
                "type": "mcq",
                "question": f"Additional multiple choice question {i+1} about {topic} ({difficulty} level). Choose the best answer:",
                "options": [f"Option A for {topic}", f"Option B for {topic}", f"Option C for {topic}", f"Option D for {topic}"],
                "answer": f"Option A for {topic}",
                "points": 2
            })
        elif question_type == "subjective_only":
            additional_questions.append({
                "type": "subjective",
                "question": f"Discuss your thoughts on an important aspect of {topic} that you find most relevant for {difficulty} level learners. Explain your reasoning with examples.",
                "options": None,
                "answer": f"This question asks for personal reflection on {topic}, considering its relevance and importance. A good answer should include specific examples, clear reasoning, and demonstrate understanding appropriate for {difficulty} level.",
                "points": 4
            })
        else:
            if (existing_mcq_count + len([q for q in additional_questions if q["type"] == "mcq"])) < 3:
                additional_questions.append({
                    "type": "mcq",
                    "question": f"Additional MCQ about {topic} ({difficulty} level). What is the most important aspect?",
                    "options": [f"Aspect A of {topic}", f"Aspect B of {topic}", f"Aspect C of {topic}", f"Aspect D of {topic}"],
                    "answer": f"Aspect A of {topic}",
                    "points": 2
                })
            else:
                additional_questions.append({
                    "type": "subjective",
                    "question": f"Explain how {topic} impacts daily life at a {difficulty} level. Provide specific examples and discuss both benefits and challenges.",
                    "options": None,
                    "answer": f"The impact of {topic} on daily life includes both positive effects such as improved efficiency and convenience, as well as challenges like complexity and adaptation requirements. Specific examples would vary based on individual circumstances and the particular aspects of {topic} being considered.",
                    "points": 4
                })
    return additional_questions

def generate_fallback_questions(topic: str, difficulty: str, question_type: str) -> List[Dict]:
    fallback_questions = []
    if question_type in ["mixed", "mcq_only"]:
        fallback_questions.extend([
            {
                "type": "mcq",
                "question": f"This is a sample multiple choice question about {topic}. What is the correct answer?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Option A",
                "points": 2
            },
            {
                "type": "mcq",
                "question": f"Another sample MCQ about {topic} for {difficulty} level. Choose the best answer:",
                "options": ["Choice 1", "Choice 2", "Choice 3", "Choice 4"],
                "answer": "Choice 1",
                "points": 2
            }
        ])
    if question_type in ["mixed", "subjective_only"]:
        fallback_questions.extend([
            {
                "type": "subjective",
                "question": f"Explain why you think {topic} is important in today's world. Discuss at least three key reasons with examples.",
                "options": None,
                "answer": f"The importance of {topic} in today's world can be seen through multiple perspectives: 1) Its practical applications in daily life, 2) Its impact on society and culture, 3) Its role in shaping future developments. Each aspect contributes to our understanding and interaction with {topic}.",
                "points": 4
            },
            {
                "type": "subjective",
                "question": f"What are the main benefits and challenges associated with {topic}? Analyze both positive and negative aspects.",
                "options": None,
                "answer": f"The benefits of {topic} include improved understanding, practical applications, and societal advancement. However, challenges may include complexity in implementation, resource requirements, and potential negative consequences that need careful consideration.",
                "points": 4
            },
            {
                "type": "subjective",
                "question": f"How do you think {topic} will evolve in the future? Discuss potential developments and their implications.",
                "options": None,
                "answer": f"The future evolution of {topic} likely involves technological advancement, changing societal needs, and emerging challenges. These developments could lead to new opportunities while requiring adaptive approaches to maximize benefits and minimize risks.",
                "points": 4
            }
        ])
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
        "temperature": 0.2,
        "max_tokens": 400,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
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

# ------------------ Flask routes ------------------

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
        "quiz_completed": False
    }
    questions = run_async(fetch_questions(topic.strip(), DIFFICULTY_LEVELS[0], question_type))
    quiz_sessions[session_id]["current_questions"] = questions
    quiz_sessions[session_id]["user_answers"] = [None] * len(questions)
    quiz_sessions[session_id]["subjective_evaluations"] = [None] * len(questions)
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
    current_level = DIFFICULTY_LEVELS[session["current_level"]]
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
    if score_percentage > 50:
        session["completed_levels"].append(current_level)
        if session["current_level"] < len(DIFFICULTY_LEVELS) - 1:
            session["current_level"] += 1
            next_level = DIFFICULTY_LEVELS[session["current_level"]]
            questions = run_async(fetch_questions(session["topic"], next_level, session["question_type"]))
            session["current_questions"] = questions
            session["user_answers"] = [None] * len(questions)
            session["subjective_evaluations"] = [None] * len(questions)
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
            session["quiz_completed"] = True
            return jsonify({
                "passed": True,
                "score": round(score_percentage, 1),
                "total_score": total_score,
                "max_score": max_possible_score,
                "detailed_results": detailed_results,
                "quiz_completed": True,
                "final_scores": session["scores"],
                "message": f"Congratulations! You've completed all levels with a final score of {score_percentage:.1f}% ({total_score}/{max_possible_score} points)!"
            })
    else:
        questions = run_async(fetch_questions(session["topic"], current_level, session["question_type"]))
        session["current_questions"] = questions
        session["user_answers"] = [None] * len(questions)
        session["subjective_evaluations"] = [None] * len(questions)
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
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
import os
from dotenv import load_dotenv
import json
import uuid
import re
from typing import Dict, Any, List

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama3-70b-8192"

# Store quiz sessions in memory (in production, use a database)
quiz_sessions: Dict[str, Dict[str, Any]] = {}

DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced"]
QUESTION_TYPES = ["mixed", "mcq_only", "subjective_only"]

def clean_json_response(content: str) -> str:
    """Clean and fix common JSON formatting issues"""
    # Remove any markdown formatting
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    
    # Remove any text before the first [
    start_idx = content.find('[')
    if start_idx != -1:
        content = content[start_idx:]
    
    # Remove any text after the last ]
    end_idx = content.rfind(']')
    if end_idx != -1:
        content = content[:end_idx + 1]
    
    # Fix common JSON issues
    # Fix missing commas before closing brackets/braces
    content = re.sub(r'"\s*\n\s*}', '"\n  }', content)
    content = re.sub(r'"\s*\n\s*]', '"\n  ]', content)
    
    # Fix trailing commas
    content = re.sub(r',\s*}', '}', content)
    content = re.sub(r',\s*]', ']', content)
    
    return content.strip()

def validate_and_fix_questions(questions_data: List[Dict]) -> List[Dict]:
    """Validate and fix question structure"""
    fixed_questions = []
    
    for i, q in enumerate(questions_data):
        if not isinstance(q, dict):
            print(f"Question {i} is not a dict: {type(q)}")
            continue
        
        # Check required fields
        if "question" not in q or "type" not in q or "answer" not in q:
            print(f"Question {i} missing required fields: {list(q.keys())}")
            continue
        
        question_type = q.get("type", "").lower()
        if question_type not in ["mcq", "subjective"]:
            print(f"Question {i} has invalid type: {question_type}")
            continue
        
        # Create clean question object
        clean_question = {
            "type": question_type,
            "question": str(q["question"]).strip(),
            "answer": str(q["answer"]).strip(),
            "points": int(q.get("points", 2 if question_type == "mcq" else 4))
        }
        
        if question_type == "mcq":
            # Validate MCQ structure
            options = q.get("options", [])
            if not isinstance(options, list) or len(options) != 4:
                print(f"MCQ Question {i} has invalid options: {options}")
                continue
            
            clean_options = [str(opt).strip() for opt in options]
            clean_question["options"] = clean_options
            
            # Ensure answer is one of the options
            if clean_question["answer"] not in clean_options:
                print(f"MCQ Question {i} answer not in options: {clean_question['answer']} not in {clean_options}")
                continue
                
        else:  # subjective
            clean_question["options"] = None
        
        fixed_questions.append(clean_question)
    
    return fixed_questions

def generate_mixed_questions_prompt(topic, difficulty):
    return f"""
Generate exactly 5 questions on the topic "{topic}" suitable for a {difficulty} learner.
Include BOTH multiple-choice and subjective questions (mix them).

IMPORTANT: Return ONLY a valid JSON array. No explanations, no markdown, no extra text.

Format each question as:
- MCQ: {{"type": "mcq", "question": "text", "options": ["A", "B", "C", "D"], "answer": "correct_option", "points": 2}}
- Subjective: {{"type": "subjective", "question": "text", "options": null, "answer": "sample_answer", "points": 4}}

CRITICAL: Subjective questions must require analysis, explanation, opinion, or discussion - NOT single factual answers.

Examples of GOOD subjective questions:
- "Explain the importance of education in personal development"
- "Discuss the advantages and disadvantages of social media"
- "What do you think are the main challenges facing the environment today? Explain your reasoning"
- "Compare and contrast different forms of government"

Examples of BAD subjective questions (these should be MCQ instead):
- "What is the capital of France?" (factual answer)
- "Who wrote Romeo and Juliet?" (single correct answer)
- "What is the largest planet?" (objective fact)

Generate a mix of 2-3 MCQs and 2-3 subjective questions on "{topic}" for {difficulty} level.
MCQs should test factual knowledge, subjective questions should require thinking and explanation.
"""

def generate_mcq_only_prompt(topic, difficulty):
    return f"""
Generate exactly 5 multiple-choice questions on the topic "{topic}" suitable for a {difficulty} learner.

IMPORTANT: Return ONLY a valid JSON array. No explanations, no markdown, no extra text.

Format: [{{"type": "mcq", "question": "text", "options": ["A", "B", "C", "D"], "answer": "correct_option", "points": 2}}]

Generate 5 MCQ questions on "{topic}" for {difficulty} level.
"""

def generate_subjective_only_prompt(topic, difficulty):
    return f"""
Generate exactly 5 subjective questions on the topic "{topic}" suitable for a {difficulty} learner.

IMPORTANT: Return ONLY a valid JSON array. No explanations, no markdown, no extra text.

Format: [{{"type": "subjective", "question": "text", "options": null, "answer": "comprehensive_answer", "points": 4}}]

CRITICAL: All questions must be truly subjective requiring analysis, explanation, opinion, or discussion - NOT factual answers.

Examples of GOOD subjective questions:
- "Explain why you think {topic} is important in today's world"
- "Discuss the main benefits and challenges of {topic}"
- "What are your thoughts on how {topic} affects daily life? Provide examples"
- "Compare different aspects of {topic} and explain which you find most significant"
- "Analyze the role of {topic} in society and its future implications"

Examples of BAD subjective questions (avoid these):
- "What is..." (factual questions)
- "Who invented..." (single correct answers)
- "When did..." (date-based facts)
- "Where is..." (location-based facts)

Generate 5 subjective questions on "{topic}" for {difficulty} level that require thoughtful responses.
"""

def evaluate_subjective_answer_prompt(question, model_answer, user_answer, points):
    return f"""
Evaluate this subjective answer and return ONLY a JSON object.

Question: {question}
Model Answer: {model_answer}
Student Answer: {user_answer}
Max Points: {points}

Return format: {{"score": number_0_to_{points}, "feedback": "brief_feedback", "percentage": number}}

Be fair in evaluation. Award partial credit for partially correct answers.
"""

async def fetch_questions(topic: str, difficulty: str, question_type: str = "mixed", retry_count: int = 0):
    """Fetch questions with improved error handling and retries"""
    if retry_count >= 3:
        raise Exception("Maximum retry attempts reached for question generation")
    
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
        "temperature": 0.4,  # Lower temperature for more consistent formatting
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
                error_data = response.json() if response.text else {"error": "Unknown error"}
                print(f"GROQ API error {response.status_code}: {error_data}")
                raise Exception(f"GROQ API error: {response.status_code}")
            
            data = response.json()
            if "choices" not in data or not data["choices"]:
                raise Exception("No choices in API response")
            
            content = data["choices"][0]["message"]["content"]
            print(f"Raw API response: {content[:200]}...")
            
            # Clean the JSON response
            cleaned_content = clean_json_response(content)
            print(f"Cleaned content: {cleaned_content[:200]}...")
            
            # Parse JSON
            try:
                questions_data = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                print(f"Problematic content: {cleaned_content}")
                
                # Try to extract and fix JSON manually
                json_match = re.search(r'\[.*\]', cleaned_content, re.DOTALL)
                if json_match:
                    potential_json = json_match.group(0)
                    try:
                        questions_data = json.loads(potential_json)
                    except json.JSONDecodeError:
                        # Last resort: retry with different prompt
                        print(f"Retrying question generation (attempt {retry_count + 1})")
                        return await fetch_questions(topic, difficulty, question_type, retry_count + 1)
                else:
                    raise Exception("Could not extract valid JSON from response")
            
            if not isinstance(questions_data, list):
                raise Exception(f"Expected list, got {type(questions_data)}")
            
            # Validate and fix questions
            fixed_questions = validate_and_fix_questions(questions_data)
            
            if len(fixed_questions) < 3:
                print(f"Too few valid questions ({len(fixed_questions)}), retrying...")
                return await fetch_questions(topic, difficulty, question_type, retry_count + 1)
            
            # Return exactly 5 questions (or all if less than 5)
            return fixed_questions[:5]
            
    except Exception as e:
        print(f"Error in fetch_questions (attempt {retry_count + 1}): {str(e)}")
        if retry_count < 2:
            return await fetch_questions(topic, difficulty, question_type, retry_count + 1)
        else:
            # Return fallback questions if all retries fail
            return generate_fallback_questions(topic, difficulty, question_type)

def generate_fallback_questions(topic: str, difficulty: str, question_type: str) -> List[Dict]:
    """Generate fallback questions when API fails"""
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
    """Evaluate a subjective answer using AI with better error handling"""
    if not user_answer or not user_answer.strip():
        return {"score": 0, "feedback": "No answer provided", "percentage": 0}
    
    prompt = evaluate_subjective_answer_prompt(question, model_answer, user_answer, max_points)
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"Evaluation API error: {response.status_code}")
                return {"score": max_points // 2, "feedback": "Auto-scored due to evaluation service error", "percentage": 50}
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Clean and parse evaluation response
            cleaned_content = clean_json_response(content)
            
            try:
                evaluation = json.loads(cleaned_content)
                
                # Validate and fix evaluation
                score = max(0, min(int(evaluation.get("score", 0)), max_points))
                feedback = str(evaluation.get("feedback", "Evaluated automatically"))
                percentage = (score / max_points) * 100 if max_points > 0 else 0
                
                return {
                    "score": score,
                    "feedback": feedback,
                    "percentage": percentage
                }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Evaluation parsing error: {e}")
                # Simple keyword-based fallback scoring
                answer_length = len(user_answer.strip().split())
                if answer_length >= 20:
                    score = max_points * 0.7  # Give 70% for substantial answers
                elif answer_length >= 10:
                    score = max_points * 0.5  # Give 50% for moderate answers
                else:
                    score = max_points * 0.3  # Give 30% for short answers
                
                return {
                    "score": int(score),
                    "feedback": "Auto-scored based on answer length and content",
                    "percentage": (score / max_points) * 100
                }
                
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        # Fallback scoring
        return {"score": max_points // 2, "feedback": "Auto-scored due to technical error", "percentage": 50}

def generate_session_id():
    return str(uuid.uuid4())

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start_quiz")
async def start_quiz(
    topic: str = Form(...), 
    question_type: str = Form(default="mixed")
):
    """Start a new quiz session with specified question type"""
    try:
        if question_type not in QUESTION_TYPES:
            question_type = "mixed"
        
        # Validate topic
        if not topic or len(topic.strip()) < 2:
            return JSONResponse(
                content={"error": "Please provide a valid topic (at least 2 characters)"}, 
                status_code=400
            )
            
        session_id = generate_session_id()
        
        # Initialize quiz session
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
        
        # Generate first set of questions
        questions = await fetch_questions(topic.strip(), DIFFICULTY_LEVELS[0], question_type)
        quiz_sessions[session_id]["current_questions"] = questions
        quiz_sessions[session_id]["user_answers"] = [None] * len(questions)
        quiz_sessions[session_id]["subjective_evaluations"] = [None] * len(questions)
        
        return JSONResponse(content={
            "session_id": session_id,
            "level": DIFFICULTY_LEVELS[0],
            "question_type": question_type,
            "questions": questions,
            "message": f"Starting {DIFFICULTY_LEVELS[0]} level quiz on {topic} ({question_type.replace('_', ' ')} questions)"
        })
        
    except Exception as e:
        print(f"Error in start_quiz: {str(e)}")
        return JSONResponse(
            content={"error": f"Failed to start quiz: {str(e)}"}, 
            status_code=500
        )

@app.post("/submit_answer")
async def submit_answer(
    session_id: str = Form(...),
    question_index: int = Form(...),
    answer: str = Form(...)
):
    """Submit an answer for a specific question"""
    if session_id not in quiz_sessions:
        return JSONResponse(content={"error": "Invalid session"}, status_code=400)
    
    session = quiz_sessions[session_id]
    
    if question_index >= len(session["current_questions"]) or question_index < 0:
        return JSONResponse(content={"error": "Invalid question index"}, status_code=400)
    
    # Store the answer
    session["user_answers"][question_index] = answer.strip() if answer else ""
    
    return JSONResponse(content={"success": True})

@app.post("/submit_quiz")
async def submit_quiz(session_id: str = Form(...)):
    """Submit the entire quiz for current level and calculate score"""
    try:
        if session_id not in quiz_sessions:
            return JSONResponse(content={"error": "Invalid session"}, status_code=400)
        
        session = quiz_sessions[session_id]
        current_level = DIFFICULTY_LEVELS[session["current_level"]]
        questions = session["current_questions"]
        user_answers = session["user_answers"]
        
        if not questions:
            return JSONResponse(content={"error": "No questions found for this session"}, status_code=400)
        
        # Calculate score for both MCQ and subjective questions
        total_score = 0
        max_possible_score = 0
        detailed_results = []
        
        for i, question in enumerate(questions):
            max_possible_score += question["points"]
            user_answer = user_answers[i] if i < len(user_answers) else None
            
            if question["type"] == "mcq":
                # MCQ scoring
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
                # Subjective scoring using AI evaluation
                evaluation = await evaluate_subjective_answer(
                    question["question"], 
                    question["answer"], 
                    user_answer or "", 
                    question["points"]
                )
                
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
        
        # Calculate overall percentage
        score_percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        session["scores"][current_level] = score_percentage
        
        # Check if user passed (>50%)
        if score_percentage > 50:
            session["completed_levels"].append(current_level)
            
            # Move to next level or complete quiz
            if session["current_level"] < len(DIFFICULTY_LEVELS) - 1:
                session["current_level"] += 1
                next_level = DIFFICULTY_LEVELS[session["current_level"]]
                
                # Generate questions for next level
                questions = await fetch_questions(session["topic"], next_level, session["question_type"])
                session["current_questions"] = questions
                session["user_answers"] = [None] * len(questions)
                session["subjective_evaluations"] = [None] * len(questions)
                
                return JSONResponse(content={
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
                # Quiz completed!
                session["quiz_completed"] = True
                return JSONResponse(content={
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
            # Failed - generate new questions for same level
            questions = await fetch_questions(session["topic"], current_level, session["question_type"])
            session["current_questions"] = questions
            session["user_answers"] = [None] * len(questions)
            session["subjective_evaluations"] = [None] * len(questions)
            
            return JSONResponse(content={
                "passed": False,
                "score": round(score_percentage, 1),
                "total_score": total_score,
                "max_score": max_possible_score,
                "detailed_results": detailed_results,
                "retry_level": current_level,
                "questions": questions,
                "message": f"You scored {score_percentage:.1f}% ({total_score}/{max_possible_score} points). You need >50% to advance. Try again with new questions!"
            })
            
    except Exception as e:
        print(f"Error in submit_quiz: {str(e)}")
        return JSONResponse(
            content={"error": f"Failed to submit quiz: {str(e)}"}, 
            status_code=500
        )

@app.get("/quiz_status/{session_id}")
async def get_quiz_status(session_id: str):
    """Get current status of a quiz session"""
    if session_id not in quiz_sessions:
        return JSONResponse(content={"error": "Invalid session"}, status_code=400)
    
    session = quiz_sessions[session_id]
    current_level = DIFFICULTY_LEVELS[session["current_level"]]
    
    return JSONResponse(content={
        "topic": session["topic"],
        "question_type": session["question_type"],
        "current_level": current_level,
        "completed_levels": session["completed_levels"],
        "scores": session["scores"],
        "quiz_completed": session["quiz_completed"],
        "questions": session["current_questions"],
        "subjective_evaluations": session.get("subjective_evaluations", [])
    })

@app.delete("/quiz_session/{session_id}")
async def delete_quiz_session(session_id: str):
    """Delete a quiz session"""
    if session_id in quiz_sessions:
        del quiz_sessions[session_id]
        return JSONResponse(content={"message": "Session deleted"})
    return JSONResponse(content={"error": "Session not found"}, status_code=404)

@app.get("/question_types")
async def get_question_types():
    """Get available question types"""
    return JSONResponse(content={
        "question_types": QUESTION_TYPES,
        "descriptions": {
            "mixed": "Mix of multiple-choice and subjective questions",
            "mcq_only": "Only multiple-choice questions",
            "subjective_only": "Only subjective/essay questions"
        }
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "active_sessions": len(quiz_sessions),
        "api_key_configured": bool(GROQ_API_KEY)
    })
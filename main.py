from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Dict, Any, List, Optional
from enum import Enum
import httpx
import os
import json
import uuid
import re
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL = "llama3-70b-8192"
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 30.0
    SESSION_CLEANUP_INTERVAL = 3600  # 1 hour
    SESSION_EXPIRY = 7200  # 2 hours

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")

# Enums for better type safety
class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class QuestionType(str, Enum):
    MIXED = "mixed"
    MCQ_ONLY = "mcq_only"
    SUBJECTIVE_ONLY = "subjective_only"

class QuestionFormat(str, Enum):
    MCQ = "mcq"
    SUBJECTIVE = "subjective"

# Pydantic models for validation
class QuizStartRequest(BaseModel):
    topic: str
    question_type: QuestionType = QuestionType.MIXED
    difficulty_level: DifficultyLevel = DifficultyLevel.BEGINNER

    @validator('topic')
    def validate_topic(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Topic must be at least 2 characters long")
        return v.strip()

class AnswerSubmissionRequest(BaseModel):
    session_id: str
    question_index: int
    answer: str

class QuizSubmissionRequest(BaseModel):
    session_id: str

@dataclass
class Question:
    type: QuestionFormat
    question: str
    answer: str
    points: int
    options: Optional[List[str]] = None

    def to_dict(self):
        return asdict(self)

@dataclass
class EvaluationResult:
    score: int
    feedback: str
    percentage: float
    rubric_breakdown: Dict[str, int]

@dataclass
class QuizSession:
    session_id: str
    topic: str
    question_type: QuestionType
    current_level: int
    current_questions: List[Question]
    user_answers: List[Optional[str]]
    subjective_evaluations: List[Optional[EvaluationResult]]
    scores: Dict[str, float]
    completed_levels: List[str]
    quiz_completed: bool
    created_at: datetime
    last_activity: datetime

    def is_expired(self) -> bool:
        return datetime.now() - self.last_activity > timedelta(seconds=Config.SESSION_EXPIRY)

    def update_activity(self):
        self.last_activity = datetime.now()

# Application setup
app = FastAPI(
    title="Adaptive Quiz System",
    description="An intelligent quiz system with adaptive difficulty",
    version="2.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory storage
quiz_sessions: Dict[str, QuizSession] = {}

# Replace your QuestionGenerator and QuestionValidator classes with these improved versions

class QuestionGenerator:
    """Simplified question generation with clearer prompts"""

    @staticmethod
    def generate_mixed_prompt(topic: str, difficulty: DifficultyLevel) -> str:
        return f"""Generate exactly 5 quiz questions about "{topic}" at {difficulty.value} level.

Return ONLY a valid JSON array with this exact format:
[
  {{"type": "mcq", "question": "Your MCQ question?", "options": ["A", "B", "C", "D"], "answer": "A", "points": 2}},
  {{"type": "subjective", "question": "Your subjective question?", "options": null, "answer": "Model answer here", "points": 4}}
]

Requirements:
- Mix of 2-3 MCQ and 2-3 subjective questions
- For MCQ: answer must be one of the 4 options
- For subjective: provide comprehensive model answer
- All questions must be unique and relevant to {topic}
- No extra text, markdown, or explanations - JSON only"""

    @staticmethod
    def generate_mcq_prompt(topic: str, difficulty: DifficultyLevel) -> str:
        return f"""Generate exactly 5 multiple choice questions about "{topic}" at {difficulty.value} level.

Return ONLY a valid JSON array:
[
  {{"type": "mcq", "question": "Question 1?", "options": ["A", "B", "C", "D"], "answer": "A", "points": 2}},
  {{"type": "mcq", "question": "Question 2?", "options": ["A", "B", "C", "D"], "answer": "B", "points": 2}}
]

Requirements:
- Exactly 4 options per question
- Answer must match one of the options exactly
- All questions unique and about {topic}
- JSON only, no markdown or extra text"""

    @staticmethod
    def generate_subjective_prompt(topic: str, difficulty: DifficultyLevel) -> str:
        return f"""Generate exactly 5 subjective questions about "{topic}" at {difficulty.value} level.

Return ONLY a valid JSON array:
[
  {{"type": "subjective", "question": "Question requiring detailed explanation?", "options": null, "answer": "Complete model answer here", "points": 4}}
]

Requirements:
- Questions need detailed, analytical answers
- Provide comprehensive model answers
- All questions unique and about {topic}
- JSON only, no markdown or extra text"""


class QuestionValidator:
    """Improved validation with better error handling"""

    @staticmethod
    def clean_json_response(content: str) -> str:
        """More robust JSON cleaning"""
        if not content or not content.strip():
            raise ValueError("Empty response content")
        
        # Remove common markdown artifacts
        content = re.sub(r'```json\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'```\s*', '', content)
        content = re.sub(r'^[^[{]*', '', content)  # Remove text before JSON
        content = re.sub(r'[^}\]]*$', '', content)  # Remove text after JSON
        
        # Find JSON array bounds
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            raise ValueError(f"No valid JSON array found in: {content[:100]}...")
        
        content = content[start_idx:end_idx + 1]
        
        # Fix common JSON issues
        content = re.sub(r',\s*}', '}', content)  # Remove trailing commas in objects
        content = re.sub(r',\s*]', ']', content)  # Remove trailing commas in arrays
        content = re.sub(r'([^\\])\\([^"\\nrtbf/u])', r'\1\\\\2', content)  # Fix bad escapes
        
        return content.strip()

    @staticmethod
    def validate_questions(questions_data: List[Dict]) -> List[Question]:
        """Improved validation with better error messages"""
        if not isinstance(questions_data, list):
            raise ValueError(f"Expected list, got {type(questions_data)}")
        
        validated_questions = []
        seen_questions = set()
        
        for i, q_data in enumerate(questions_data):
            try:
                # Basic structure validation
                if not isinstance(q_data, dict):
                    logger.warning(f"Question {i}: Not a dictionary, skipping")
                    continue
                
                # Required fields check
                required_fields = ["question", "type", "answer"]
                missing_fields = [field for field in required_fields if field not in q_data]
                if missing_fields:
                    logger.warning(f"Question {i}: Missing fields {missing_fields}")
                    continue
                
                # Type validation
                q_type = str(q_data["type"]).lower().strip()
                if q_type not in ["mcq", "subjective"]:
                    logger.warning(f"Question {i}: Invalid type '{q_type}'")
                    continue
                
                # Question text validation
                q_text = str(q_data["question"]).strip()
                if len(q_text) < 10:
                    logger.warning(f"Question {i}: Question too short")
                    continue
                
                if q_text in seen_questions:
                    logger.warning(f"Question {i}: Duplicate question")
                    continue
                seen_questions.add(q_text)
                
                # Answer validation
                answer = str(q_data["answer"]).strip()
                if len(answer) < 1:
                    logger.warning(f"Question {i}: Empty answer")
                    continue
                
                # Create question object
                question = Question(
                    type=QuestionFormat(q_type),
                    question=q_text,
                    answer=answer,
                    points=int(q_data.get("points", 2 if q_type == "mcq" else 4))
                )
                
                # MCQ-specific validation
                if q_type == "mcq":
                    options = q_data.get("options", [])
                    if not isinstance(options, list):
                        logger.warning(f"Question {i}: Options not a list")
                        continue
                    
                    if len(options) != 4:
                        logger.warning(f"Question {i}: Expected 4 options, got {len(options)}")
                        continue
                    
                    # Clean and validate options
                    clean_options = [str(opt).strip() for opt in options if str(opt).strip()]
                    if len(clean_options) != 4:
                        logger.warning(f"Question {i}: Some options are empty")
                        continue
                    
                    question.options = clean_options
                    
                    # Check if answer matches any option
                    if answer not in clean_options:
                        # Try to find close match
                        close_match = None
                        for opt in clean_options:
                            if opt.lower() == answer.lower():
                                close_match = opt
                                break
                        
                        if close_match:
                            question.answer = close_match
                            logger.info(f"Question {i}: Fixed answer case mismatch")
                        else:
                            logger.warning(f"Question {i}: Answer '{answer}' not in options {clean_options}")
                            # Force fix by replacing first option
                            question.options[0] = answer
                            logger.info(f"Question {i}: Forced answer into options")
                
                validated_questions.append(question)
                logger.info(f"Question {i}: Validated successfully")
                
            except Exception as e:
                logger.error(f"Question {i}: Validation error - {e}")
                continue
        
        logger.info(f"Validated {len(validated_questions)} out of {len(questions_data)} questions")
        return validated_questions


class QuizService:
    """Updated service with better error handling"""

    def __init__(self):
        self.http_client = None

    async def __aenter__(self):
        self.http_client = httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            await self.http_client.aclose()

    async def generate_questions(
        self,
        topic: str,
        difficulty: DifficultyLevel,
        question_type: QuestionType,
        retry_count: int = 0
    ) -> List[Question]:
        """Improved question generation with better retry logic"""
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(f"Generating questions - Attempt {attempt + 1}/{max_attempts}")
                
                # Get prompt
                prompt = self._get_prompt(topic, difficulty, question_type)
                
                # Call API
                response_content = await self._call_api(prompt)
                logger.info(f"API Response length: {len(response_content)} chars")
                
                # Clean and parse JSON
                cleaned_content = QuestionValidator.clean_json_response(response_content)
                logger.info(f"Cleaned JSON: {cleaned_content[:200]}...")
                
                questions_data = json.loads(cleaned_content)
                logger.info(f"Parsed {len(questions_data)} questions from JSON")
                
                # Validate questions
                validated_questions = QuestionValidator.validate_questions(questions_data)
                logger.info(f"Successfully validated {len(validated_questions)} questions")
                
                # If we have enough questions, return them
                if len(validated_questions) >= 3:  # Minimum acceptable
                    # Pad to 5 if needed
                    while len(validated_questions) < 5:
                        validated_questions.append(self._create_fallback_question(
                            topic, difficulty, question_type, len(validated_questions)
                        ))
                    return validated_questions[:5]
                else:
                    logger.warning(f"Only got {len(validated_questions)} valid questions, retrying...")
                    continue
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    logger.error("All attempts failed, using fallback questions")
                    break
            except Exception as e:
                logger.error(f"Generation error on attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    break
        
        # Fallback: create 5 simple questions
        logger.warning("Using fallback questions")
        return [
            self._create_fallback_question(topic, difficulty, question_type, i)
            for i in range(5)
        ]

    def _create_fallback_question(
        self, 
        topic: str, 
        difficulty: DifficultyLevel, 
        question_type: QuestionType, 
        index: int
    ) -> Question:
        """Create a simple fallback question"""
        
        if question_type == QuestionType.SUBJECTIVE_ONLY or (
            question_type == QuestionType.MIXED and index % 2 == 1
        ):
            return Question(
                type=QuestionFormat.SUBJECTIVE,
                question=f"Explain an important concept related to {topic} (Question {index + 1})",
                options=None,
                answer=f"A comprehensive explanation about {topic} covering key concepts, applications, and importance in the field.",
                points=4
            )
        else:
            return Question(
                type=QuestionFormat.MCQ,
                question=f"Which of the following is most relevant to {topic}? (Question {index + 1})",
                options=[
                    f"Basic concept of {topic}",
                    f"Advanced application of {topic}",
                    f"Unrelated topic",
                    f"Complex theory in {topic}"
                ],
                answer=f"Basic concept of {topic}",
                points=2
            )

    def _get_prompt(self, topic: str, difficulty: DifficultyLevel, question_type: QuestionType) -> str:
        if question_type == QuestionType.MCQ_ONLY:
            return QuestionGenerator.generate_mcq_prompt(topic, difficulty)
        elif question_type == QuestionType.SUBJECTIVE_ONLY:
            return QuestionGenerator.generate_subjective_prompt(topic, difficulty)
        else:
            return QuestionGenerator.generate_mixed_prompt(topic, difficulty)

    async def _call_api(self, prompt: str) -> str:
        """API call with better error handling"""
        headers = {
            "Authorization": f"Bearer {Config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": Config.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # Lower temperature for more consistent JSON
            "max_tokens": 2000,
        }
        
        try:
            response = await self.http_client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                error_text = response.text[:500] if response.text else "No error details"
                logger.error(f"API error {response.status_code}: {error_text}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"API error: {error_text}"
                )
            
            data = response.json()
            
            if "choices" not in data or not data["choices"]:
                raise ValueError("No choices in API response")
            
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("Empty content in API response")
            
            return content.strip()
            
        except httpx.TimeoutException:
            logger.error("API request timeout")
            raise HTTPException(status_code=504, detail="API request timeout")
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

    # Keep the rest of your existing QuizService methods (evaluate_subjective_answer, etc.)
    # ... [rest of the methods remain the same]

    async def evaluate_subjective_answer(
        self,
        question: str,
        model_answer: str,
        user_answer: str,
        max_points: int
    ) -> EvaluationResult:
        if not user_answer or not user_answer.strip():
            return EvaluationResult(
                score=0,
                feedback="No answer provided. Please provide a response to demonstrate your understanding.",
                percentage=0.0,
                rubric_breakdown={"content": 0, "thinking": 0, "communication": 0, "examples": 0}
            )
        try:
            prompt = self._get_evaluation_prompt(question, model_answer, user_answer, max_points)
            response_content = await self._call_api(prompt)
            cleaned_content = QuestionValidator.clean_json_response(response_content)
            evaluation_data = json.loads(cleaned_content)
            return EvaluationResult(
                score=max(0, min(int(evaluation_data.get("score", 0)), max_points)),
                feedback=str(evaluation_data.get("feedback", "Evaluated automatically")),
                percentage=round((evaluation_data.get("score", 0) / max_points) * 100, 1),
                rubric_breakdown=evaluation_data.get("rubric_breakdown", {})
            )
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return self._fallback_evaluation(user_answer, max_points)

    def _get_evaluation_prompt(self, question: str, model_answer: str, user_answer: str, max_points: int) -> str:
        return f"""
Evaluate this subjective answer using a comprehensive rubric.
RUBRIC (out of 100%):
- Content Knowledge (40%): Accuracy and depth
- Critical Thinking (30%): Analysis and reasoning
- Communication (20%): Clarity and organization
- Examples/Evidence (10%): Supporting details
SCORING SCALE:
{max_points} points (90-100%): Exceptional
{max_points-1} points (80-89%): Proficient
{max_points-2} points (70-79%): Developing
{max_points-3} points (60-69%): Beginning
0-{max(0,max_points-4)} points (0-59%): Inadequate
QUESTION: {question}
MODEL ANSWER: {model_answer}
STUDENT ANSWER: {user_answer}
Return JSON:
{{"score": number, "feedback": "constructive_feedback", "percentage": number, "rubric_breakdown": {{"content": number, "thinking": number, "communication": number, "examples": number}}}}
"""

    def _fallback_evaluation(self, user_answer: str, max_points: int) -> EvaluationResult:
        words = len(user_answer.strip().split())
        if words >= 50:
            score = max_points - 1
        elif words >= 25:
            score = max_points - 2
        elif words >= 10:
            score = max_points - 3
        else:
            score = 1
        percentage = (score / max_points) * 100
        return EvaluationResult(
            score=score,
            feedback=f"Answer evaluated based on length and content. Consider adding more detail and examples for higher scores.",
            percentage=round(percentage, 1),
            rubric_breakdown={
                "content": int(percentage * 0.4),
                "thinking": int(percentage * 0.3),
                "communication": int(percentage * 0.2),
                "examples": int(percentage * 0.1)
            }
        )

class SessionManager:
    """Manages quiz sessions and cleanup"""

    @staticmethod
    def create_session(
        topic: str,
        question_type: QuestionType,
        difficulty_level: DifficultyLevel
    ) -> str:
        session_id = str(uuid.uuid4())
        quiz_sessions[session_id] = QuizSession(
            session_id=session_id,
            topic=topic,
            question_type=question_type,
            current_level=list(DifficultyLevel).index(difficulty_level),
            current_questions=[],
            user_answers=[],
            subjective_evaluations=[],
            scores={level.value: 0.0 for level in DifficultyLevel},
            completed_levels=[],
            quiz_completed=False,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        return session_id

    @staticmethod
    def get_session(session_id: str) -> Optional[QuizSession]:
        session = quiz_sessions.get(session_id)
        if session and not session.is_expired():
            session.update_activity()
            return session
        elif session:
            del quiz_sessions[session_id]
        return None

    @staticmethod
    def cleanup_expired_sessions():
        expired_sessions = [
            session_id for session_id, session in quiz_sessions.items()
            if session.is_expired()
        ]
        for session_id in expired_sessions:
            del quiz_sessions[session_id]
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Background task for session cleanup
async def periodic_cleanup():
    while True:
        try:
            SessionManager.cleanup_expired_sessions()
            await asyncio.sleep(Config.SESSION_CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
            await asyncio.sleep(Config.SESSION_CLEANUP_INTERVAL)

@app.on_event("startup")
async def startup_event():
    try:
        Config.validate()
        logger.info("Application started successfully")
        asyncio.create_task(periodic_cleanup())
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start_quiz")
async def start_quiz(request: QuizStartRequest):
    try:
        session_id = SessionManager.create_session(
            request.topic,
            request.question_type,
            request.difficulty_level
        )
        async with QuizService() as quiz_service:
            questions = await quiz_service.generate_questions(
                request.topic,
                request.difficulty_level,
                request.question_type
            )
        session = SessionManager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=500, detail="Failed to create session")
        session.current_questions = questions
        session.user_answers = [None] * len(questions)
        session.subjective_evaluations = [None] * len(questions)
        return JSONResponse(content={
            "session_id": session_id,
            "level": request.difficulty_level.value,
            "question_type": request.question_type.value,
            "questions": [q.to_dict() for q in questions],
            "message": f"Starting {request.difficulty_level.value} level quiz on {request.topic}"
        })
    except Exception as e:
        logger.error(f"Error starting quiz: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start quiz: {str(e)}")

@app.post("/submit_answer")
async def submit_answer(request: AnswerSubmissionRequest):
    session = SessionManager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")
    if request.question_index >= len(session.current_questions) or request.question_index < 0:
        raise HTTPException(status_code=400, detail="Invalid question index")
    while len(session.user_answers) <= request.question_index:
        session.user_answers.append(None)
    session.user_answers[request.question_index] = request.answer.strip()
    session.update_activity()
    return JSONResponse(content={"success": True})

@app.post("/submit_quiz")
async def submit_quiz(request: QuizSubmissionRequest):
    session = SessionManager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")
    try:
        current_level = list(DifficultyLevel)[session.current_level]
        questions = session.current_questions
        user_answers = session.user_answers
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found")
        total_score = 0
        max_possible_score = sum(q.points for q in questions)
        detailed_results = []
        async with QuizService() as quiz_service:
            for i, question in enumerate(questions):
                user_answer = user_answers[i] if i < len(user_answers) else None
                if question.type == QuestionFormat.MCQ:
                    is_correct = user_answer == question.answer
                    score = question.points if is_correct else 0
                    detailed_results.append({
                        "question_index": i,
                        "type": "mcq",
                        "score": score,
                        "max_score": question.points,
                        "is_correct": is_correct,
                        "feedback": "Correct!" if is_correct else f"Incorrect. Correct answer: {question.answer}",
                        "correct_answer": question.answer,
                        "user_answer": user_answer
                    })
                elif question.type == QuestionFormat.SUBJECTIVE:
                    evaluation = await quiz_service.evaluate_subjective_answer(
                        question.question,
                        question.answer,
                        user_answer or "",
                        question.points
                    )
                    score = evaluation.score
                    while len(session.subjective_evaluations) <= i:
                        session.subjective_evaluations.append(None)
                    session.subjective_evaluations[i] = evaluation
                    detailed_results.append({
                        "question_index": i,
                        "type": "subjective",
                        "score": score,
                        "max_score": question.points,
                        "feedback": evaluation.feedback,
                        "model_answer": question.answer,
                        "user_answer": user_answer,
                        "evaluation": asdict(evaluation)
                    })
                total_score += score
        score_percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        session.scores[current_level.value] = score_percentage
        if score_percentage > 50:
            session.completed_levels.append(current_level.value)
            if session.current_level < len(DifficultyLevel) - 1:
                session.current_level += 1
                next_level = list(DifficultyLevel)[session.current_level]
                async with QuizService() as quiz_service:
                    questions = await quiz_service.generate_questions(
                        session.topic,
                        next_level,
                        session.question_type
                    )
                session.current_questions = questions
                session.user_answers = [None] * len(questions)
                session.subjective_evaluations = [None] * len(questions)
                return JSONResponse(content={
                    "passed": True,
                    "score": round(score_percentage, 1),
                    "total_score": total_score,
                    "max_score": max_possible_score,
                    "detailed_results": detailed_results,
                    "next_level": next_level.value,
                    "questions": [q.to_dict() for q in questions],
                    "message": f"Congratulations! Moving to {next_level.value} level."
                })
            else:
                session.quiz_completed = True
                return JSONResponse(content={
                    "passed": True,
                    "score": round(score_percentage, 1),
                    "total_score": total_score,
                    "max_score": max_possible_score,
                    "detailed_results": detailed_results,
                    "quiz_completed": True,
                    "final_scores": session.scores,
                    "message": "Congratulations! You've completed all levels!"
                })
        else:
            async with QuizService() as quiz_service:
                questions = await quiz_service.generate_questions(
                    session.topic,
                    current_level,
                    session.question_type
                )
            session.current_questions = questions
            session.user_answers = [None] * len(questions)
            session.subjective_evaluations = [None] * len(questions)
            return JSONResponse(content={
                "passed": False,
                "score": round(score_percentage, 1),
                "total_score": total_score,
                "max_score": max_possible_score,
                "detailed_results": detailed_results,
                "retry": True,
                "current_level": current_level.value,
                "questions": [q.to_dict() for q in questions],
                "message": f"Score too low. Try again at {current_level.value} level."
            })
    except Exception as e:
        logger.error(f"Error submitting quiz: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit quiz: {str(e)}")

@app.get("/quiz_status/{session_id}")
async def get_quiz_status(session_id: str):
    session = SessionManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    current_level = list(DifficultyLevel)[session.current_level]
    return JSONResponse(content={
        "session_id": session_id,
        "topic": session.topic,
        "current_level": current_level.value,
        "quiz_completed": session.quiz_completed,
        "completed_levels": session.completed_levels,
        "scores": session.scores,
        "questions_count": len(session.current_questions),
        "answered_count": sum(1 for ans in session.user_answers if ans is not None)
    })

@app.get("/session/{session_id}/results")
async def get_session_results(session_id: str):
    session = SessionManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    results = {
        "session_id": session_id,
        "topic": session.topic,
        "completed_levels": session.completed_levels,
        "scores": session.scores,
        "quiz_completed": session.quiz_completed,
        "detailed_evaluations": []
    }
    for i, evaluation in enumerate(session.subjective_evaluations):
        if evaluation:
            results["detailed_evaluations"].append({
                "question_index": i,
                "evaluation": asdict(evaluation)
            })
    return JSONResponse(content=results)

@app.post("/reset_session/{session_id}")
async def reset_session(session_id: str):
    session = SessionManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.current_level = 0
    session.current_questions = []
    session.user_answers = []
    session.subjective_evaluations = []
    session.scores = {level.value: 0.0 for level in DifficultyLevel}
    session.completed_levels = []
    session.quiz_completed = False
    session.update_activity()
    return JSONResponse(content={"message": "Session reset successfully"})

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in quiz_sessions:
        del quiz_sessions[session_id]
        return JSONResponse(content={"message": "Session deleted successfully"})
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "active_sessions": len(quiz_sessions),
        "timestamp": datetime.now().isoformat()
    })

@app.get("/api/stats")
async def get_stats():
    active_sessions = len(quiz_sessions)
    completed_quizzes = sum(1 for session in quiz_sessions.values() if session.quiz_completed)
    topics = {}
    for session in quiz_sessions.values():
        topic = session.topic.lower()
        topics[topic] = topics.get(topic, 0) + 1
    levels = {level.value: 0 for level in DifficultyLevel}
    for session in quiz_sessions.values():
        current_level = list(DifficultyLevel)[session.current_level]
        levels[current_level.value] += 1
    return JSONResponse(content={
        "active_sessions": active_sessions,
        "completed_quizzes": completed_quizzes,
        "topic_distribution": topics,
        "level_distribution": levels,
        "uptime": datetime.now().isoformat()
    })

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

class QuizAnalytics:
    @staticmethod
    def calculate_performance_metrics(session: QuizSession) -> Dict[str, Any]:
        metrics = {
            "completion_rate": 0.0,
            "average_score": 0.0,
            "level_progression": len(session.completed_levels),
            "time_spent": 0,
            "question_accuracy": {}
        }
        if session.scores:
            completed_scores = [score for score in session.scores.values() if score > 0]
            if completed_scores:
                metrics["average_score"] = sum(completed_scores) / len(completed_scores)
        if session.created_at and session.last_activity:
            time_diff = session.last_activity - session.created_at
            metrics["time_spent"] = int(time_diff.total_seconds())
        mcq_correct = mcq_total = 0
        subj_avg_score = subj_total = 0
        for i, question in enumerate(session.current_questions):
            if i < len(session.user_answers) and session.user_answers[i]:
                if question.type == QuestionFormat.MCQ:
                    mcq_total += 1
                    if session.user_answers[i] == question.answer:
                        mcq_correct += 1
                elif question.type == QuestionFormat.SUBJECTIVE:
                    if i < len(session.subjective_evaluations) and session.subjective_evaluations[i]:
                        subj_total += 1
                        subj_avg_score += session.subjective_evaluations[i].percentage
        if mcq_total > 0:
            metrics["question_accuracy"]["mcq"] = (mcq_correct / mcq_total) * 100
        if subj_total > 0:
            metrics["question_accuracy"]["subjective"] = subj_avg_score / subj_total
        return metrics

    @staticmethod
    def generate_recommendations(session: QuizSession) -> List[str]:
        recommendations = []
        metrics = QuizAnalytics.calculate_performance_metrics(session)
        if metrics["average_score"] < 60:
            recommendations.append(f"Focus on foundational concepts in {session.topic}")
            recommendations.append("Consider reviewing basic materials before advancing")
        if "mcq" in metrics["question_accuracy"] and metrics["question_accuracy"]["mcq"] < 70:
            recommendations.append("Practice more multiple-choice questions to improve recall")
        if "subjective" in metrics["question_accuracy"] and metrics["question_accuracy"]["subjective"] < 70:
            recommendations.append("Work on providing more detailed explanations and examples")
            recommendations.append("Practice structuring your written responses")
        if metrics["level_progression"] == 0:
            recommendations.append("Take your time to understand each concept thoroughly")
        if not recommendations:
            recommendations.append("Great job! Continue practicing to maintain your skills")
            recommendations.append("Consider exploring advanced topics in this subject")
        return recommendations

@app.get("/session/{session_id}/analytics")
async def get_session_analytics(session_id: str):
    session = SessionManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    metrics = QuizAnalytics.calculate_performance_metrics(session)
    recommendations = QuizAnalytics.generate_recommendations(session)
    return JSONResponse(content={
        "session_id": session_id,
        "metrics": metrics,
        "recommendations": recommendations,
        "session_summary": {
            "topic": session.topic,
            "question_type": session.question_type.value,
            "levels_completed": len(session.completed_levels),
            "quiz_completed": session.quiz_completed,
            "total_questions_answered": sum(1 for ans in session.user_answers if ans is not None)
        }
    })

try:
    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.websockets import WebSocketState

    class ConnectionManager:
        def __init__(self):
            self.active_connections: Dict[str, WebSocket] = {}

        async def connect(self, websocket: WebSocket, session_id: str):
            await websocket.accept()
            self.active_connections[session_id] = websocket

        def disconnect(self, session_id: str):
            if session_id in self.active_connections:
                del self.active_connections[session_id]

        async def send_personal_message(self, message: dict, session_id: str):
            if session_id in self.active_connections:
                websocket = self.active_connections[session_id]
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)

    manager = ConnectionManager()

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        await manager.connect(websocket, session_id)
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_text(f"Message received: {data}")
        except WebSocketDisconnect:
            manager.disconnect(session_id)

except ImportError:
    logger.warning("WebSocket support not available")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
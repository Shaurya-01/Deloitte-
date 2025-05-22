from flask import Flask, request, jsonify, render_template, session
import ollama
import json
import random

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Predefined question bank as fallback
FALLBACK_QUESTIONS = {
    'general knowledge': {
        'beginner': [
            {"question": "What is the capital of France?", "options": {"A": "London", "B": "Berlin", "C": "Paris", "D": "Madrid"}, "answer": "C"},
            {"question": "How many days are in a week?", "options": {"A": "5", "B": "6", "C": "7", "D": "8"}, "answer": "C"},
            {"question": "What color do you get when you mix red and white?", "options": {"A": "Purple", "B": "Pink", "C": "Orange", "D": "Yellow"}, "answer": "B"},
            {"question": "Which animal is known as the King of the Jungle?", "options": {"A": "Tiger", "B": "Elephant", "C": "Lion", "D": "Bear"}, "answer": "C"},
            {"question": "How many continents are there?", "options": {"A": "5", "B": "6", "C": "7", "D": "8"}, "answer": "C"},
            {"question": "What is 10 + 5?", "options": {"A": "13", "B": "14", "C": "15", "D": "16"}, "answer": "C"},
            {"question": "Which planet is closest to the Sun?", "options": {"A": "Venus", "B": "Earth", "C": "Mercury", "D": "Mars"}, "answer": "C"},
            {"question": "What is the largest ocean on Earth?", "options": {"A": "Atlantic", "B": "Pacific", "C": "Indian", "D": "Arctic"}, "answer": "B"},
            {"question": "How many sides does a triangle have?", "options": {"A": "2", "B": "3", "C": "4", "D": "5"}, "answer": "B"}
        ],
        'intermediate': [
            {"question": "What is the chemical symbol for gold?", "options": {"A": "Go", "B": "Gd", "C": "Au", "D": "Ag"}, "answer": "C"},
            {"question": "Who wrote 'Romeo and Juliet'?", "options": {"A": "Charles Dickens", "B": "William Shakespeare", "C": "Jane Austen", "D": "Mark Twain"}, "answer": "B"},
            {"question": "What is the square root of 64?", "options": {"A": "6", "B": "7", "C": "8", "D": "9"}, "answer": "C"},
            {"question": "Which country invented pizza?", "options": {"A": "France", "B": "Spain", "C": "Greece", "D": "Italy"}, "answer": "D"},
            {"question": "What is the hardest natural substance?", "options": {"A": "Gold", "B": "Iron", "C": "Diamond", "D": "Silver"}, "answer": "C"},
            {"question": "In which year did World War II end?", "options": {"A": "1944", "B": "1945", "C": "1946", "D": "1947"}, "answer": "B"},
            {"question": "What is the currency of Japan?", "options": {"A": "Yuan", "B": "Won", "C": "Yen", "D": "Rupee"}, "answer": "C"},
            {"question": "Which organ in the human body produces insulin?", "options": {"A": "Liver", "B": "Kidney", "C": "Heart", "D": "Pancreas"}, "answer": "D"},
            {"question": "What is 15% of 200?", "options": {"A": "25", "B": "30", "C": "35", "D": "40"}, "answer": "B"}
        ],
        'hard': [
            {"question": "What is the atomic number of carbon?", "options": {"A": "4", "B": "6", "C": "8", "D": "12"}, "answer": "B"},
            {"question": "Who developed the theory of relativity?", "options": {"A": "Isaac Newton", "B": "Galileo Galilei", "C": "Albert Einstein", "D": "Stephen Hawking"}, "answer": "C"},
            {"question": "What is the derivative of x²?", "options": {"A": "x", "B": "2x", "C": "x²", "D": "2x²"}, "answer": "B"},
            {"question": "Which philosophical school was founded by Zeno of Citium?", "options": {"A": "Epicureanism", "B": "Stoicism", "C": "Cynicism", "D": "Skepticism"}, "answer": "B"},
            {"question": "What is the smallest prime number greater than 10?", "options": {"A": "11", "B": "12", "C": "13", "D": "15"}, "answer": "A"},
            {"question": "In which layer of the atmosphere do most weather phenomena occur?", "options": {"A": "Stratosphere", "B": "Mesosphere", "C": "Troposphere", "D": "Thermosphere"}, "answer": "C"},
            {"question": "What is the half-life of Carbon-14?", "options": {"A": "5,730 years", "B": "10,000 years", "C": "1,000 years", "D": "50,000 years"}, "answer": "A"},
            {"question": "Who wrote 'The Wealth of Nations'?", "options": {"A": "Karl Marx", "B": "John Maynard Keynes", "C": "Adam Smith", "D": "David Ricardo"}, "answer": "C"},
            {"question": "What is the molecular formula for glucose?", "options": {"A": "C6H12O6", "B": "H2O", "C": "CO2", "D": "C2H6O"}, "answer": "A"}
        ]
    }
}

def generate_custom_questions(topic, difficulty, count=9):
    """Generate questions using AI or return fallback questions"""
    
    # Try different models in order of preference
    models_to_try = ['llama3.1:8b', 'llama3.2:3b', 'llama3.2:1b', 'phi3:mini']
    
    for model in models_to_try:
        try:
            print(f"Trying model: {model}")
            
            # Check if model exists
            response = ollama.chat(model=model, messages=[
                {"role": "user", "content": "Hello"}
            ])
            
            # If successful, use this model for questions
            prompt = f"""Generate {count} multiple choice questions about {topic} at {difficulty} difficulty level.

IMPORTANT: Return ONLY a valid JSON array. No other text.

Example format:
[
  {{"question": "What is 2+2?", "options": {{"A": "3", "B": "4", "C": "5", "D": "6"}}, "answer": "B"}},
  {{"question": "What is the capital of France?", "options": {{"A": "London", "B": "Berlin", "C": "Paris", "D": "Madrid"}}, "answer": "C"}}
]

Generate {count} questions about {topic} ({difficulty} level):"""

            response = ollama.chat(model=model, messages=[
                {"role": "user", "content": prompt}
            ])
            
            text = response['message']['content'].strip()
            
            # Extract JSON
            start_idx = text.find('[')
            end_idx = text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_text = text[start_idx:end_idx]
                questions = json.loads(json_text)
                
                # Validate questions
                valid_questions = []
                for q in questions:
                    if (isinstance(q, dict) and 
                        'question' in q and 'options' in q and 'answer' in q and
                        isinstance(q['options'], dict) and
                        all(opt in q['options'] for opt in ['A', 'B', 'C', 'D'])):
                        valid_questions.append(q)
                
                if len(valid_questions) >= count:
                    print(f"Successfully generated {len(valid_questions)} questions with {model}")
                    return valid_questions[:count]
            
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue
    
    # If all models fail, return fallback questions
    print("All models failed, using fallback questions")
    return get_fallback_questions(topic, difficulty, count)

def get_fallback_questions(topic, difficulty, count=9):
    """Get fallback questions from predefined bank"""
    
    # Use general knowledge questions if topic not found
    topic_key = topic.lower() if topic.lower() in FALLBACK_QUESTIONS else 'general knowledge'
    
    if topic_key in FALLBACK_QUESTIONS and difficulty in FALLBACK_QUESTIONS[topic_key]:
        questions = FALLBACK_QUESTIONS[topic_key][difficulty].copy()
        random.shuffle(questions)
        
        # If we need more questions, duplicate and modify
        while len(questions) < count:
            base_questions = FALLBACK_QUESTIONS[topic_key][difficulty]
            for q in base_questions:
                if len(questions) >= count:
                    break
                modified_q = q.copy()
                modified_q['question'] = f"[Variation] {q['question']}"
                questions.append(modified_q)
        
        return questions[:count]
    else:
        # Generate basic fallback questions
        return [
            {
                "question": f"Sample {difficulty} question {i+1} about {topic}?",
                "options": {
                    "A": f"Option A for question {i+1}",
                    "B": f"Option B for question {i+1}",
                    "C": f"Option C for question {i+1}",
                    "D": f"Option D for question {i+1}"
                },
                "answer": ["A", "B", "C", "D"][i % 4]
            }
            for i in range(count)
        ]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_quiz():
    topic = request.json.get('topic', 'general knowledge')
    
    print(f"Starting quiz generation for topic: {topic}")
    
    # Generate questions for all difficulty levels
    difficulties = ['beginner', 'intermediate', 'hard']
    all_questions = {}
    
    for difficulty in difficulties:
        print(f"Generating questions for {difficulty} level...")
        questions = generate_custom_questions(topic, difficulty, 9)
        all_questions[difficulty] = questions
        print(f"Got {len(questions)} questions for {difficulty}")

    # Initialize session data
    session['all_questions'] = all_questions
    session['current_difficulty'] = 'beginner'
    session['difficulty_index'] = 0
    session['question_set'] = 0
    session['score'] = 0
    session['total_score'] = 0
    session['difficulty_scores'] = {'beginner': 0, 'intermediate': 0, 'hard': 0}
    
    # Get first question from beginner level
    first_question = all_questions['beginner'][0]
    
    return jsonify({
        "question": first_question,
        "difficulty": "beginner",
        "question_number": 1,
        "total_questions": 3,
        "message": "Starting with Beginner level"
    })

@app.route('/answer', methods=['POST'])
def check_answer():
    data = request.json
    user_answer = data.get('answer', '').strip().upper()

    all_questions = session.get('all_questions', {})
    current_difficulty = session.get('current_difficulty', 'beginner')
    difficulty_index = session.get('difficulty_index', 0)
    question_set = session.get('question_set', 0)
    score = session.get('score', 0)
    total_score = session.get('total_score', 0)
    difficulty_scores = session.get('difficulty_scores', {'beginner': 0, 'intermediate': 0, 'hard': 0})

    if current_difficulty not in all_questions:
        return jsonify({"error": "Quiz data not found."}), 400

    current_questions = all_questions[current_difficulty]
    question_index = question_set * 3 + difficulty_index
    
    if question_index >= len(current_questions):
        return jsonify({"error": "No more questions available."}), 400

    correct_answer = current_questions[question_index]['answer']
    is_correct = user_answer == correct_answer
    
    if is_correct:
        score += 1
        total_score += 1

    difficulty_index += 1
    
    # Check if we've completed a set of 3 questions
    if difficulty_index >= 3:
        difficulty_scores[current_difficulty] = score
        percentage = (score / 3) * 100
        
        if percentage >= 50:  # 2 out of 3 correct or better
            # Move to next difficulty
            if current_difficulty == 'beginner':
                next_difficulty = 'intermediate'
            elif current_difficulty == 'intermediate':
                next_difficulty = 'hard'
            else:
                # Quiz completed
                session.update({
                    'difficulty_scores': difficulty_scores,
                    'total_score': total_score
                })
                return jsonify({
                    "message": "Quiz completed! Congratulations!",
                    "final_scores": difficulty_scores,
                    "total_score": total_score,
                    "max_score": 9,
                    "done": True
                })
            
            # Progress to next difficulty
            session.update({
                'current_difficulty': next_difficulty,
                'difficulty_index': 0,
                'question_set': 0,
                'score': 0,
                'total_score': total_score,
                'difficulty_scores': difficulty_scores
            })
            
            next_question = all_questions[next_difficulty][0]
            return jsonify({
                "question": next_question,
                "difficulty": next_difficulty,
                "question_number": 1,
                "total_questions": 3,
                "message": f"Great job! Moving to {next_difficulty.title()} level. You scored {score}/3 ({percentage:.1f}%) on {current_difficulty}.",
                "previous_score": f"{score}/3",
                "done": False
            })
        else:
            # Failed current difficulty - give new set of questions
            if question_set >= 2:  # Already tried 3 sets
                return jsonify({
                    "message": f"Quiz ended. You need to score above 50% to progress. Final score on {current_difficulty}: {score}/3 ({percentage:.1f}%)",
                    "final_scores": difficulty_scores,
                    "total_score": total_score,
                    "done": True
                })
            
            # Move to next set of questions for same difficulty
            question_set += 1
            session.update({
                'difficulty_index': 0,
                'question_set': question_set,
                'score': 0
            })
            
            next_question_index = question_set * 3
            next_question = current_questions[next_question_index]
            
            return jsonify({
                "question": next_question,
                "difficulty": current_difficulty,
                "question_number": 1,
                "total_questions": 3,
                "message": f"You scored {score}/3 ({percentage:.1f}%) on {current_difficulty}. Try again with new questions!",
                "previous_score": f"{score}/3",
                "done": False
            })
    
    # Continue with next question in current set
    session.update({
        'difficulty_index': difficulty_index,
        'score': score,
        'total_score': total_score
    })
    
    next_question_index = question_set * 3 + difficulty_index
    next_question = current_questions[next_question_index]
    
    return jsonify({
        "question": next_question,
        "difficulty": current_difficulty,
        "question_number": difficulty_index + 1,
        "total_questions": 3,
        "current_score": score,
        "is_correct": is_correct,
        "correct_answer": correct_answer if not is_correct else None,
        "done": False
    })

if __name__ == '__main__':
    app.run(debug=True)
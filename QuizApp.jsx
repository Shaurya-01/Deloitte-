import React, { useState } from "react";

// Utility for classNames
function cn(...classes) {
  return classes.filter(Boolean).join(" ");
}

const questionTypeDescriptions = {
  mixed: "Mix of multiple-choice and subjective questions for comprehensive assessment",
  mcq_only: "Only multiple-choice questions with instant feedback",
  subjective_only: "Only essay/short answer questions with AI-powered evaluation",
};

const LEVEL_DISPLAY = {
  beginner: "Beginner",
  intermediate: "Intermediate",
  advanced: "Advanced",
};

const LEVEL_BADGE_CLASS = {
  beginner: "level-badge level-beginner",
  intermediate: "level-badge level-intermediate",
  advanced: "level-badge level-advanced",
};

function ProgressBar({ percent }) {
  return (
    <div className="progress-bar">
      <div
        className="progress-fill"
        id="progressFill"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}

function Loading({ show, text = "Loading..." }) {
  if (!show) return null;
  return (
    <div className="loading">
      <div className="spinner"></div>
      <p id="loadingText">{text}</p>
    </div>
  );
}

function QuizInfo({ topic, questionType, level }) {
  return (
    <div id="quizInfo" className="quiz-info">
      <h4>Quiz Information</h4>
      <p>
        <strong>Topic:</strong> <span id="quizTopic">{topic}</span>
      </p>
      <p>
        <strong>Question Type:</strong>{" "}
        <span id="currentQuestionType">{questionTypeDescriptions[questionType]}</span>
      </p>
      <p>
        <strong>Level:</strong>{" "}
        <span id="currentLevel" className={LEVEL_BADGE_CLASS[level]}>
          {LEVEL_DISPLAY[level]}
        </span>
      </p>
    </div>
  );
}

function LevelIndicator() {
  return (
    <div id="levelIndicator" className="level-indicator">
      <p>Score more than 50% to advance to the next level!</p>
    </div>
  );
}

function QuestionCard({ q, index, total, value, onChange }) {
  return (
    <div className="question-card">
      <div className="question-header">
        <div>
          <div className="question-number">
            Question {index + 1} of {total}
          </div>
          <span
            className={cn(
              "question-type-badge",
              q.type === "mcq" ? "type-mcq" : "type-subjective"
            )}
          >
            {q.type === "mcq" ? "MCQ" : "ESSAY"}
          </span>
        </div>
        <div className="question-points">{q.points} pts</div>
      </div>
      <div className="question-text">{q.question}</div>
      {q.type === "mcq" ? (
        <ul className="options">
          {q.options.map((option, optIdx) => (
            <li className="option" key={optIdx}>
              <label>
                <input
                  type="radio"
                  name={`question_${index}`}
                  value={option}
                  checked={value === option}
                  onChange={() => onChange(option)}
                />
                {option}
              </label>
            </li>
          ))}
        </ul>
      ) : (
        <>
          <textarea
            className="subjective-answer"
            name={`question_${index}`}
            placeholder="Type your answer here... (Be detailed and specific)"
            value={value || ""}
            onChange={e => onChange(e.target.value)}
            onKeyUp={e => onChange(e.target.value)}
          />
          <div
            className="answer-counter"
            id={`counter_${index}`}
            style={{
              color:
                !value || value.length < 50
                  ? "#dc3545"
                  : value.length < 150
                  ? "#ffc107"
                  : "#28a745",
            }}
          >
            {value ? value.length : 0} characters
          </div>
        </>
      )}
    </div>
  );
}

function ScoreSummary({ score, total_score, max_score, passed, message }) {
  return (
    <div className="score-summary">
      <div className={`score ${passed ? "pass" : "fail"}`}>
        {score.toFixed(1)}%
      </div>
      <div className="points-display">
        {typeof total_score !== "undefined" && typeof max_score !== "undefined"
          ? `${total_score}/${max_score} points`
          : ""}
      </div>
      <div className={`message ${passed ? "success" : "error"}`}>{message}</div>
    </div>
  );
}

function DetailedResults({ results, questions }) {
  return (
    <div className="detailed-results">
      <h3>Detailed Results</h3>
      {results.map((result, idx) => {
        let className = "result-item";
        if (result.type === "mcq") {
          className += result.is_correct ? " correct" : " incorrect";
        } else {
          const percentage = (result.score / result.max_score) * 100;
          if (percentage >= 80) className += " correct";
          else if (percentage >= 50) className += " partial";
          else className += " incorrect";
        }
        return (
          <div className={className} key={idx}>
            <div className="result-question">
              Question {idx + 1}: {questions[idx].question}
              <span style={{ float: "right" }}>
                {result.score}/{result.max_score} pts
              </span>
            </div>
            <div className="result-answer">
              <strong>Your answer:</strong>{" "}
              {result.user_answer || "Not answered"}
              <br />
              <strong>
                {result.type === "mcq" ? "Correct answer" : "Model answer"}:
              </strong>{" "}
              {result.type === "mcq"
                ? result.correct_answer
                : result.model_answer}
            </div>
            <div className="result-feedback">{result.feedback}</div>
          </div>
        );
      })}
    </div>
  );
}

function FinalScores({ final_scores }) {
  return (
    <div className="final-scores">
      <h3>Final Scores</h3>
      {Object.entries(final_scores).map(([level, score]) => (
        <div className="score-item" key={level}>
          <span>{LEVEL_DISPLAY[level]}</span>
          <span>{score.toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

export default function QuizApp() {
  // State
  const [page, setPage] = useState("start"); // start | quiz | result
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("");
  // quiz state
  const [topic, setTopic] = useState("");
  const [questionType, setQuestionType] = useState("mixed");
  const [sessionId, setSessionId] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [level, setLevel] = useState("beginner");
  const [answers, setAnswers] = useState([]);
  // result state
  const [result, setResult] = useState(null);
  const [detailedVisible, setDetailedVisible] = useState(false);
  const [currentProgress, setCurrentProgress] = useState(0);

  // Handlers
  async function handleStartQuiz(e) {
    e.preventDefault();
    if (!topic.trim()) {
      alert("Please enter a topic");
      return;
    }
    setLoading(true);
    setLoadingText("Generating questions...");
    setPage("quiz");
    try {
      const formData = new FormData();
      formData.append("topic", topic.trim());
      formData.append("question_type", questionType);

      const res = await fetch("/start_quiz", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      setSessionId(data.session_id);
      setQuestions(data.questions);
      setLevel(data.level);
      setAnswers(Array(data.questions.length).fill(""));
      setResult(null);
      setCurrentProgress(0);
    } catch (err) {
      alert(err.message);
      setPage("start");
    } finally {
      setLoading(false);
      setLoadingText("");
    }
  }

  function handleAnswerChange(idx, value) {
    setAnswers((old) => {
      const next = old.slice();
      next[idx] = value;
      return next;
    });
  }

  async function handleSubmitQuiz() {
    setLoading(true);
    setLoadingText("Evaluating answers...");
    try {
      // Submit all answers per question (as in original logic)
      for (let i = 0; i < questions.length; i++) {
        const answer = answers[i];
        if (answer) {
          const formData = new FormData();
          formData.append("session_id", sessionId);
          formData.append("question_index", i);
          formData.append("answer", answer);
          await fetch("/submit_answer", {
            method: "POST",
            body: formData,
          });
        }
      }
      // Submit the quiz
      const formData = new FormData();
      formData.append("session_id", sessionId);
      const res = await fetch("/submit_quiz", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);

      // Update progress
      if (data.quiz_completed) setCurrentProgress(100);
      else if (data.next_level === "intermediate") setCurrentProgress(33);
      else if (data.next_level === "advanced") setCurrentProgress(66);
      else setCurrentProgress(0);

      setPage("result");
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
      setLoadingText("");
    }
  }

  function handleContinueQuiz() {
    if (!result) return;
    setQuestions(result.questions);
    setLevel(result.next_level);
    setAnswers(Array(result.questions.length).fill(""));
    setResult(null);
    setDetailedVisible(false);
    setPage("quiz");
  }

  function handleRetryLevel() {
    if (!result) return;
    setQuestions(result.questions);
    setAnswers(Array(result.questions.length).fill(""));
    setResult(null);
    setDetailedVisible(false);
    setPage("quiz");
  }

  function handleNewQuiz() {
    setPage("start");
    setTopic("");
    setQuestionType("mixed");
    setSessionId(null);
    setQuestions([]);
    setLevel("beginner");
    setAnswers([]);
    setResult(null);
    setDetailedVisible(false);
    setCurrentProgress(0);
  }

  // ------------------- JSX -------------------

  return (
    <div className="container">
      <div className="header">
        <h1>Progressive Quiz</h1>
        <p>
          Master topics through progressive difficulty levels with mixed question
          types
        </p>
        <ProgressBar percent={currentProgress} />
      </div>
      <div className="content">
        <Loading show={loading} text={loadingText} />

        {/* Start Form */}
        {page === "start" && !loading && (
          <form
            id="startForm"
            className="start-form"
            onSubmit={handleStartQuiz}
            autoComplete="off"
          >
            <div className="input-group">
              <label htmlFor="topic">Enter Quiz Topic:</label>
              <input
                type="text"
                id="topic"
                value={topic}
                placeholder="e.g., JavaScript, Python, History, Science..."
                required
                onChange={e => setTopic(e.target.value)}
              />
            </div>
            <div className="input-group">
              <label htmlFor="questionType">Question Type:</label>
              <select
                id="questionType"
                value={questionType}
                onChange={e => setQuestionType(e.target.value)}
              >
                <option value="mixed">Mixed (MCQ + Subjective)</option>
                <option value="mcq_only">Multiple Choice Only</option>
                <option value="subjective_only">Subjective Only</option>
              </select>
              <div className="question-type-info" id="questionTypeInfo">
                {questionTypeDescriptions[questionType]}
              </div>
            </div>
            <button className="btn" type="submit">
              Start Progressive Quiz
            </button>
          </form>
        )}

        {/* Quiz Area */}
        {page === "quiz" && !loading && (
          <div id="quizArea" className="quiz-area" style={{ display: "block" }}>
            <QuizInfo topic={topic} questionType={questionType} level={level} />
            <LevelIndicator />
            <div id="questionsContainer">
              {questions.map((q, idx) => (
                <QuestionCard
                  key={idx}
                  q={q}
                  index={idx}
                  total={questions.length}
                  value={answers[idx]}
                  onChange={v => handleAnswerChange(idx, v)}
                />
              ))}
            </div>
            <div className="quiz-controls">
              <button
                className="btn"
                onClick={handleSubmitQuiz}
                id="submitBtn"
                type="button"
              >
                Submit Quiz
              </button>
            </div>
          </div>
        )}

        {/* Result Card */}
        {page === "result" && result && !loading && (
          <div id="resultCard" className="result-card" style={{ display: "block" }}>
            <ScoreSummary
              score={result.score}
              total_score={result.total_score}
              max_score={result.max_score}
              passed={result.passed}
              message={result.message}
            />
            {/* Detailed Results Toggle */}
            {result.detailed_results && (
              <div style={{ textAlign: "center" }}>
                <button
                  className="btn"
                  onClick={() => setDetailedVisible(v => !v)}
                  id="toggleResultsBtn"
                  type="button"
                >
                  {detailedVisible ? "Hide Detailed Results" : "Show Detailed Results"}
                </button>
              </div>
            )}
            {/* Detailed Results */}
            {detailedVisible && result.detailed_results && (
              <DetailedResults results={result.detailed_results} questions={questions} />
            )}
            {/* Final Scores */}
            {result.quiz_completed && result.final_scores && (
              <FinalScores final_scores={result.final_scores} />
            )}
            {/* Controls */}
            <div style={{ textAlign: "center" }}>
              {result.quiz_completed ? (
                <button className="btn" onClick={handleNewQuiz} id="newQuizBtn" type="button">
                  Start New Quiz
                </button>
              ) : result.passed && result.next_level ? (
                <button className="btn" onClick={handleContinueQuiz} id="continueBtn" type="button">
                  Continue to Next Level
                </button>
              ) : (
                <button className="btn" onClick={handleRetryLevel} id="retryBtn" type="button">
                  Try Again
                </button>
              )}
            </div>
          </div>
        )}
      </div>
      {/* Style Tag: put your CSS here or import as separate file */}
      <style>{`
        /* (Paste your CSS here, unchanged from your HTML) */
        /* ... */
      `}</style>
    </div>
  );
}
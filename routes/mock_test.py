from flask import Blueprint, request, jsonify
import json
import random

mock_test_bp = Blueprint("mock_test", __name__)

# START TEST
@mock_test_bp.route("/mock-test/start", methods=["POST"])
def start_mock_test():
    role = request.json.get("role")

    try:
        with open(f"mock_data/{role.replace(' ', '_')}.json") as f:
            data = json.load(f)

        # pick 5 random questions
        questions = random.sample(data["questions"], 5)

        # remove answers before sending to frontend
        for q in questions:
            q.pop("answer")

        return jsonify({"questions": questions})

    except Exception as e:
        return jsonify({"error": str(e)}), 404


# SUBMIT TEST
@mock_test_bp.route("/mock-test/submit", methods=["POST"])
def submit_mock_test():
    role = request.json.get("role")
    user_answers = request.json.get("answers")

    with open(f"mock_data/{role.replace(' ', '_')}.json") as f:
        data = json.load(f)

    score = 0
    results = []

    for i, q in enumerate(data["questions"][:len(user_answers)]):
        correct = q["answer"]
        user_ans = user_answers[i]

        if user_ans == correct:
            score += 1

        results.append({
            "question": q["question"],
            "correct": correct,
            "your_answer": user_ans,
            "explanation": q["explanation"]
        })

    return jsonify({
        "score": score,
        "total": len(user_answers),
        "results": results
    })
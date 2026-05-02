from flask import Blueprint, request, jsonify
import json

mock_interview_bp = Blueprint("mock_interview", __name__)

# Start Interview
@mock_interview_bp.route("/mock-interview/start", methods=["POST"])
def start_interview():
    role = request.json.get("role")

    try:
        with open(f"mock_data/interview_{role.replace(' ', '_')}.json") as f:
            data = json.load(f)

        return jsonify({
            "questions": data["questions"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 404


# Feedback (simple logic for now)
@mock_interview_bp.route("/mock-interview/feedback", methods=["POST"])
def interview_feedback():
    answer = request.json.get("answer")

    # simple feedback logic (no AI)
    if len(answer) < 20:
        feedback = "Answer is too short. Try to explain more."
        rating = 4
    else:
        feedback = "Good answer. Try adding real examples."
        rating = 7

    return jsonify({
        "feedback": feedback,
        "rating": rating
    })
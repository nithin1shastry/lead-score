from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from langfuse import Langfuse
from datetime import datetime
from typing import Dict, Any
import time

app = Flask(__name__)
CORS(app, resources={r"/score-lead": {"origins": "*"}})



# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST_URL")
)

# LiteLLM API endpoint
OPENAI_API_URL = os.getenv("OPENAI_API_URL")
AUTH_API_PWD = os.getenv("AUTH_API_PWD")

API_KEY = os.getenv("API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def create_lead_scoring_prompt(comment: str) -> str:
    """
    Creates an advanced prompt for lead scoring, calibrated to handle
    simple product inquiries with appropriate scoring levels.
    
    Args:
        comment (str): Customer inquiry or comment
    
    Returns:
        str: Formatted prompt for lead scoring
    """
    if not comment or comment.lower() in ["nothing", "none", "n/a", "no comment", "nil", "blank"]:
        base_prompt = f"""You are a lead scoring assistant. The following customer inquiry is too vague or non-specific:
    INQUIRY: {comment}"""
        score_instruction = "Since the inquiry is too vague or missing critical details, assign a score below 5, indicating no immediate interest or priority."
    else:
        base_prompt = f"""You are a lead scoring assistant. Given the following customer inquiry, analyze it and provide a lead score between 0-100:
    INQUIRY: {comment}"""
        score_instruction = "Please analyze the inquiry and provide only a numeric score."

    return f"""{base_prompt}
    SCORING CRITERIA:
    - Urgency (35%): 
        * Time-Critical Words:
            - High Urgency: urgent, urgently, emergency, asap, right now, immediately, this week, critical, top priority, time-sensitive, without delay, straight away, at once, pressing, immediate attention, act fast, can't wait, by EOD (End of Day), within hours, by tomorrow
            - Medium Urgency: need it, soon, shortly, this month, at your earliest convenience, coming up, in a timely manner, by next week, within a few days, before long, in the near future, not too long from now
            - Low Urgency: when available, sometime, eventually, no rush, whenever possible, at some point, in due time, down the line, in the future, long-term
        * Context Clues:
            - Direct Questions: "do you have", "is it available"
            - Time Mentions: today, tomorrow, next week
            - Situational: for event, for meeting, before date
    
    - Purchase Intent (25%):
        * Strong Intent Indicators:
            - Direct Questions: "do you have", "is X available", "can I get"
            - Action Words: need, want, looking for, searching for
            - Specific Product Mentions: exact color, size, type
        * Medium Intent:
            - General Inquiries: "what about", "how about"
            - Product Information: asking about features, colors
        * Weak Intent:
            - Browsing: just looking, browsing, exploring
            - Hypothetical: might want, maybe, perhaps
    
    - Request Specificity (20%):
        * High Specificity:
            - Exact Product Details: specific color, size, model
            - Clear Requirements: must be green, has to be
        * Medium Specificity:
            - General Product Category: carpet, furniture
            - Basic Features: color, material
        * Low Specificity:
            - Vague Mentions: something like, kind of
            - Unclear Requirements: nice, good
    
    - Context Quality (20%):
        * High Quality:
            - Complete Questions
            - Clear Purpose
            - Additional Context
        * Medium Quality:
            - Basic Questions
            - Simple Requirements
        * Low Quality:
            - Very Short Queries
            - Unclear Purpose

    RESPONSE FORMAT:
    Only provide a number between 0-100.

    EXAMPLES:
    - "Do you guys have green carpet?" → 60
    - "Do you guys have green carpet that I need it urgently?" → 85
    - "Looking for a green carpet for tomorrow's event" → 90
    - "Just browsing carpet options" → 30
    - "Need green carpet asap for office" → 88
    - "What types of carpets do you have?" → 40
    - "Do you have the green carpet in stock for immediate purchase?" → 82
    - "Wondering about carpet colors" → 35
    - "Need urgent delivery of green carpet for stage setup" → 95
    - "Is the green carpet available?" → 55

    STRICT RULES:
    1. Only respond with a number between 0-100
    2. Direct product inquiries ("do you have X") start at base score of 55-65
    3. Adding urgency terms increases score by 20-30 points
    4. Specific product details add 5-10 points
    5. Context or purpose adds 5-10 points
    6. Multiple urgency indicators can push score into 90+ range
    7. Vague or browsing queries score below 40
    8. Simple availability questions without urgency score 55-65

    {score_instruction}"""






def get_llm_score(comment: str, trace_id: str) -> Dict[str, Any]:
    span = langfuse.span(
        name="llm_scoring",
        trace_id=trace_id,
    )
    
    retries = 3
    delay = 5
    for attempt in range(retries):
        try:
            prompt = create_lead_scoring_prompt(comment)
            
            # Calculate input tokens for cost tracking
            system_message = "You are an AI lead scoring assistant."
            input_content = system_message + prompt
            
            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 10
            }

            response = requests.post(
                OPENAI_API_URL,
                json=payload,
                headers=HEADERS
            )

            if response.status_code == 503:
                
                time.sleep(delay)
                continue

            if response.status_code != 200:
                raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

            response_json = response.json()
            

            # Extract usage details from response
            usage = response_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Calculate costs (GPT-4 pricing)
            input_cost = (prompt_tokens * 0.03) / 1000  # $0.03 per 1K tokens
            output_cost = (completion_tokens * 0.06) / 1000  # $0.06 per 1K tokens
            total_cost = input_cost + output_cost

            # Record generation with usage and cost details
            generation = langfuse.generation(
                name="lead_score_generation",
                trace_id=trace_id,
                model="gpt-4",
                model_parameters={
                    "temperature": 0.3,
                    "max_tokens": 10
                },
                usage_details={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": total_tokens,
                },
                cost_details={
                    "input": input_cost,
                    "output": output_cost,
                    "total": total_cost
                },
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                metadata={
                    "attempt": attempt + 1,
                    "prompt": prompt
                }
            )

            # Extract response content
            score_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if not score_text:
                raise ValueError("Received an empty response from the model.")

            # Extract numeric score from response
            score = None
            for word in score_text.split():
                if word.isdigit():
                    score = int(word)
                    break

            if score is None:
                
                if any(word in comment.lower() for word in ["urgent", "asap", "immediately"]):
                    score = 90
                elif any(word in comment.lower() for word in ["exploring", "future", "maybe"]):
                    score = 30
                elif any(word in comment.lower() for word in ["high-end", "luxury", "premium"]):
                    score = 80
                else:
                    score = 50

            score = max(0, min(100, score))
            

            # End the span with success status and metadata
            span.end(
                status="success",
                metadata={
                    "score": score,
                    "prompt": prompt,
                    "response": response_json,
                    "usage": usage,
                    "cost": {
                        "input": input_cost,
                        "output": output_cost,
                        "total": total_cost
                    }
                }
            )

            return {"score": score, "status": "success"}

        except Exception as e:
            
            span.end(
                status="error",
                metadata={
                    "error": str(e),
                    "prompt": prompt
                }
            )

            if attempt == retries - 1:
                return {"score": 0, "status": "error", "error": str(e)}

    return {"score": 0, "status": "error", "error": "Max retries exceeded"}

def authenticate(request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header != f"Bearer {AUTH_API_PWD}":
        return False
    return True



@app.route('/api/score-lead', methods=['POST'])

def score_lead():
    if not authenticate(request):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    
    comment = data.get('comment', '')
    
    
    if not comment:
        return jsonify({"error": "Comment is required"}), 400

    # Create a new trace for this request
    trace = langfuse.trace(
        name="lead_scoring",
        metadata={
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat(),
            "user_agent": request.headers.get('User-Agent')
        }
    )

    
    result = get_llm_score(comment, trace.id)
    

    if result["status"] == "success":
        score = result["score"]
        

        # Log scores in Langfuse
        langfuse.score(
            name="lead_score",
            value=score,
            trace_id=trace.id,
            metadata={
                "comment": comment,
                "score": score
            }
        )

        # Add observations about the scoring process using `track`
        langfuse.trace(
            name="score_analysis",
            metadata={
                "score_category": "high" if score >= 80 else "medium" if score >= 50 else "low",
                "contains_urgency": any(word in comment.lower() for word in ["urgent", "asap", "immediately"]),
                "project_scope": "large" if "complete" in comment.lower() or "full" in comment.lower() else "medium"
            }
        )

        return jsonify({
            "score": score,
            "trace_id": trace.id
        })
    else:
        return jsonify({
            "error": "Failed to generate score",
            "details": result.get("error"),
            "trace_id": trace.id
        }), 500




    

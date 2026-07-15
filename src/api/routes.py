"""
API routes.
"""

from flask import (
    Blueprint,
    jsonify,
    request
)

from src.api.predictor import (
    SentimentPredictor
)

from src.models.model_utils import (
    get_available_models,
    validate_model_name
)

api_blueprint = Blueprint(
    "api",
    __name__
)

predictor = SentimentPredictor()

# =============================================================================
# Health
# =============================================================================

@api_blueprint.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify(
        {
            "status": "healthy"
        }
    )

# =============================================================================
# Information
# =============================================================================

@api_blueprint.route(
    "/info",
    methods=["GET"]
)
def info():

    return jsonify(
        {
            "task": "sentiment-analysis",
            "available_models": get_available_models()
        }
    )

# =============================================================================
# Prediction
# =============================================================================

@api_blueprint.route(
    "/predict",
    methods=["POST"]
)
def predict():

    data = request.get_json()

    if data is None:

        return jsonify(
            {
                "error": "Request body must be JSON."
            }
        ), 400

    review = data.get(
        "review"
    )

    if not isinstance(review, str):

        return jsonify(
            {
                "error": "review must be a string."
            }
        ), 400

    review = review.strip()

    if not review:

        return jsonify(
            {
                "error": "review cannot be empty."
            }
        ), 400

    model_name = data.get(
        "model",
        "best_model"
    )

    try:

        if model_name != "best_model":

            validate_model_name(
                model_name
            )

        result = predictor.predict(
            review=review,
            model_name=model_name
        )

        return jsonify(
            {
                "review": review,
                **result
            }
        )

    except ValueError as error:

        return jsonify(
            {
                "error": str(error)
            }
        ), 400

    except FileNotFoundError as error:

        return jsonify(
            {
                "error": str(error)
            }
        ), 404

    except Exception as error:

        return jsonify(
            {
                "error": "Prediction failed.",
                "details": str(error)
            }
        ), 500
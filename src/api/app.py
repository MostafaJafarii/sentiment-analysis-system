"""
Flask application entry point.
"""

from flask import (
    Flask,
    jsonify,
    render_template
)

from src.api.routes import (
    api_blueprint
)

def create_app() -> Flask:
    """
    Create and configure Flask application.
    """

    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static"
    )

    # Register API routes
    app.register_blueprint(
        api_blueprint
    )

    # ---------------------------------------------------------------------
    # Web Interface
    # ---------------------------------------------------------------------

    @app.route("/")
    def index():
        """
        Render the web interface.
        """

        return render_template(
            "index.html"
        )

    # ---------------------------------------------------------------------
    # Error Handlers
    # ---------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(error):
        """
        Handle 404 errors.
        """

        return jsonify(
            {
                "error": "Resource not found."
            }
        ), 404

    @app.errorhandler(500)
    def internal_error(error):
        """
        Handle internal server errors.
        """

        return jsonify(
            {
                "error": "Internal server error."
            }
        ), 500

    return app

app = create_app()

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
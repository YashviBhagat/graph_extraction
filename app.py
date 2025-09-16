import os
from flask import Flask

# Import blueprints
from routes.upload import upload_bp
from routes.gallery import gallery_bp
from routes.api import api_bp
from routes.graphs import graphs_bp

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = 'supersecretkey'  # Change for production
    app.config['UPLOAD_FOLDER'] = 'static/files/'
    app.config['GRAPH_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'graphs')

    # Create directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)

    # Register blueprints
    app.register_blueprint(upload_bp)
    app.register_blueprint(gallery_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(graphs_bp)  # Graph routes
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

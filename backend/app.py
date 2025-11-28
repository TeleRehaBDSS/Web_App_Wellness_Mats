from flask import Flask
from backend.routes import app  # Import the app instance from routes.py

if __name__ == "__main__":
    # Ensure the uploads directory exists
    UPLOAD_DIRECTORY = "uploads"
    import os
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)
    
    # Run the Flask server
    app.run(host="0.0.0.0", port=5000, debug=True)

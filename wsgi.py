from waitress import serve
from app import app

if __name__ == "__main__":
    # Instead of app.run(), use Waitress to serve the app
    serve(app, host="0.0.0.0", port=8080) 
from backend import app
# waitress-serve --listen=127.0.0.1:5000 wsgi:app
# gunicorn wsgi:app
if __name__ == '__main__':
    app.run()

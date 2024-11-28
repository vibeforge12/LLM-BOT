import os

from flask import Flask, request, send_from_directory, render_template, Blueprint

from logger import logger
from common import *
from config import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, UPLOAD_FOLDER)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'
app.json.sort_keys = True

@app.route("/")
def index():
    return 'text'

if __name__ == '__main__':
    logger.info('Start XAI API Server')

    upload_folder = os.path.join(APP_ROOT, UPLOAD_FOLDER)
    os.makedirs(upload_folder, exist_ok=True)

    app.run(host='0.0.0.0', port=SERVER_PORT, debug=DEBUG)
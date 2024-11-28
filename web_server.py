import os

from flask import Flask, request, Blueprint, jsonify

from langchain_rag import DialogAgent
from logger import logger
from common import *
import config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, UPLOAD_FOLDER)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'
app.json.sort_keys = True

api = Blueprint('api', __name__, url_prefix='/api')


@app.route("/")
def index():
    return 'text'


@api.route("/chat", methods=['POST'])
def chat():
    session_id = request.form.get('session_id')
    input_text = request.form.get('text')

    logger.info(f'Current session_id: {session_id}')

    print(f'User: {input_text}')

    dialog_agent = DialogAgent(session_id)

    result = dialog_agent.generate_response(input_text)

    logger.debug(f'LLM result: {result}')

    return jsonify(result)


app.register_blueprint(api)

if __name__ == '__main__':
    logger.info('Start XAI API Server')

    upload_folder = os.path.join(APP_ROOT, UPLOAD_FOLDER)
    os.makedirs(upload_folder, exist_ok=True)

    app.run(host='0.0.0.0', port=SERVER_PORT, debug=DEBUG)

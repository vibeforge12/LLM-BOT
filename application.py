import os

from flask import Flask, request, Blueprint, jsonify

from langchain_rag import DialogAgent
from logger import logger
from common import *
import config

application = Flask(__name__)
application.config['SECRET_KEY'] = 'secret!'
application.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, UPLOAD_FOLDER)
application.config['JSON_AS_ASCII'] = False
application.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'
application.json.sort_keys = True

api = Blueprint('api', __name__, url_prefix='/api')


@application.route("/")
def index():
    return 'text'


@api.route("/chat", methods=['POST'])
def chat():
    session_id = request.form.get('session_id')
    input_text = request.form.get('text')

    logger.info(f'Current session_id: {session_id}')

    print(f'User: {input_text}')

    dialog_agent = DialogAgent(session_id)

    agent_result = dialog_agent.generate_response(input_text)
    agent_answer = agent_result['answer']

    logger.debug(f'LLM result: {agent_result}')

    return jsonify(agent_result)


application.register_blueprint(api)

if __name__ == '__main__':
    logger.info('Start XAI API Server')

    upload_folder = os.path.join(APP_ROOT, UPLOAD_FOLDER)
    os.makedirs(upload_folder, exist_ok=True)

    application.run(host='0.0.0.0', port=SERVER_PORT, debug=DEBUG)

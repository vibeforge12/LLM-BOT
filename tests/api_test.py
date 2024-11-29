import random
import string
import unittest
from datetime import datetime

import requests
from langchain_core.messages import HumanMessage, AIMessage

import config
from langchain_rag import UserSimulator
from logger import logger
from utils.utils import generate_session_id


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.url = config.SERVER_URL
        pass

    def test_request_chat(self):
        url = self.url + "/api/chat"

        # 세션 아이디 생성
        session_id = generate_session_id()

        # session_id = 'test_session_id'

        input_text = '안녕하세요'

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(url,
                                 data={'session_id': session_id, 'text': input_text}, verify=False)

        self.assertEqual(200, response.status_code)
        res = response.json()
        print(res)

    def test_request_user_simulator(self):
        url = self.url + "/api/chat"

        # 세션 아이디 생성
        session_id = generate_session_id()

        user_simulator = UserSimulator(session_id=session_id, temperature=1.0)

        input_text = '안녕하세요'

        headers = {
            'Content-Type': 'application/json'
        }

        max_turn = 20  # 최대 대화 가능한 턴
        turn = 0  # 각 발화 당 1개의 턴

        while True:
            logger.info(f'User: {input_text}')
            turn += 1
            response = requests.post(url,
                                     data={'session_id': session_id, 'text': input_text}, verify=False)

            self.assertEqual(200, response.status_code)
            res = response.json()
            # print(res)
            turn += 1

            answer = res['answer']
            logger.info(f'Agent: {answer}')

            if turn >= max_turn:
                logger.info('Max turn reached. Conversation end.')
                break

            is_finished = res['is_finished']
            if is_finished:
                category = res['category']
                break

            chat_history = res['history']

            # convert history format
            input_history = []
            for history in chat_history:
                if history['role'] == 'User':
                    input_history.append(HumanMessage(history['content']))
                else:
                    input_history.append(AIMessage(history['content']))

            input_text = user_simulator.generate_response(input_history)

        logger.info(f'Category: {category}')

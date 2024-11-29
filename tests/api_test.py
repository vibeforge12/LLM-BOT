import random
import string
import unittest
from datetime import datetime

import requests

import config
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
        pass

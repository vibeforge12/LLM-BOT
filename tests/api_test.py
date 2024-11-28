import random
import string
import unittest
from datetime import datetime

import requests

import config


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.url = config.SERVER_URL
        pass

    def test_request_chat(self):
        url = self.url + "/api/chat"

        # 세션 아이디 생성
        date_str = datetime.now().strftime("%Y%m%d")
        rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
        session_id = f"{date_str}_{rand_str}"

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

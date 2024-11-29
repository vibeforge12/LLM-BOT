import random
import string
import unittest
from datetime import datetime

import requests

import config
from langchain_rag import DialogAgent, UserSimulator
from logger import logger
from utils.utils import reverse_history_role


class TestUnit(unittest.TestCase):
    def setUp(self):
        pass

    def test_simulated_chat(self):
        # 세션 아이디 생성
        date_str = datetime.now().strftime("%Y%m%d")
        rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
        session_id = f"{date_str}_{rand_str}"

        dialog_agent = DialogAgent(session_id)
        user_simulator = UserSimulator()

        input_text = '안녕하세요. 진로에 대해 고민이 있어서 상담을 신청했어요.'

        is_finished = False
        while True:
            logger.info(f'User: {input_text}')
            agent_result = dialog_agent.generate_response(input_text)
            agent_answer = agent_result['answer']
            logger.info(f'Agent: {agent_answer}')

            if '[EOF]' in agent_answer:
                logger.info('Conversation end.')
                is_finished = True
                break

            chat_history = dialog_agent.get_chat_history(return_type='object')

            # history = [
            #     AIMessage(input_text),
            #     HumanMessage("진로에 대해 고민이 많은가보구나.")
            # ]
            input_history = reverse_history_role(chat_history)

            input_text = user_simulator.generate_response(input_history)

        # chat_history가 10턴 이내에서 종료되었는지 확인
        self.assertLessEqual(len(chat_history), 10)
        # [EOF]가 있는지 확인
        self.assertTrue(is_finished)



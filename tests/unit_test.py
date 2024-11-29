import unittest

import config
from langchain_rag import DialogAgent, UserSimulator
from logger import logger
from utils.utils import reverse_history_role, generate_session_id


class TestUnit(unittest.TestCase):
    def setUp(self):
        pass

    def test_simulated_chat(self):
        end_token = '[EOF]'
        # 세션 아이디 생성
        session_id = generate_session_id()
        dialog_agent = DialogAgent(session_id)
        user_simulator = UserSimulator(session_id=session_id, temperature=1.0)

        input_text = '안녕하세요. 진로에 대해 고민이 있어서 상담을 신청했어요.'

        is_finished = False  # EOF가 있는지 확인하기 위한 변수
        max_turn = 20  # 최대 대화 가능한 턴
        turn = 0  # 각 발화 당 1개의 턴

        while True:
            logger.info(f'User: {input_text}')
            turn += 1
            agent_result = dialog_agent.generate_response(input_text)
            agent_answer = agent_result['answer']
            logger.info(f'Agent: {agent_answer}')
            turn += 1

            if turn >= max_turn:
                logger.info('Max turn reached. Conversation end.')
                break

            is_finished = agent_result['is_finished']
            if is_finished:
                category = agent_result['category']
                break

            chat_history = dialog_agent.get_chat_history(return_type='object')

            # history = [
            #     AIMessage(input_text),
            #     HumanMessage("진로에 대해 고민이 많은가보구나.")
            # ]
            input_history = reverse_history_role(chat_history)

            input_text = user_simulator.generate_response(input_history)

        # [EOF]가 있는지 확인
        self.assertTrue(is_finished)

        logger.info(f'Category: {category}')

        # chat_history가 10턴 이내에서 종료되었는지 확인
        self.assertLessEqual(len(chat_history), 10)

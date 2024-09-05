from fastapi import Body
from configs import logger, log_verbose
from server.utils import BaseResponse
from server.db.repository import feedback_message_to_db

def chat_feedback(message_id: str = Body("", max_length=32, description="Chat record id"),
            score: int = Body(0, max=100, description="User rating, full score is 100, the larger the score, the higher the rating"),
            reason: str = Body("", description="Reasons for user ratings, such as not being consistent with the facts, etc.")
            ):
    try:
        feedback_message_to_db(message_id, score, reason)
    except Exception as e:
        msg = f"Feedback chat history error: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg=f"Chat history has been reported {message_id}")

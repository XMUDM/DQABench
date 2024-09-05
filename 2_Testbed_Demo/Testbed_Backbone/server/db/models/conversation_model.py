from sqlalchemy import Column, Integer, String, DateTime, JSON, func
from server.db.base import Base


class ConversationModel(Base):
    """
    Chat history model
    """
    __tablename__ = 'conversation'
    id = Column(String(32), primary_key=True, comment='Dialog ID')
    name = Column(String(50), comment='Dialog Box Name')
    # chat/agent_chat etc.
    chat_type = Column(String(50), comment='Chat Type')
    create_time = Column(DateTime, default=func.now(), comment='Creation time')

    def __repr__(self):
        return f"<Conversation(id='{self.id}', name='{self.name}', chat_type='{self.chat_type}', create_time='{self.create_time}')>"

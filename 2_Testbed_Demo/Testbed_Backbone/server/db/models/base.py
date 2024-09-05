from datetime import datetime
from sqlalchemy import Column, DateTime, String, Integer


class BaseModel:
    """
    Basic Model
    """
    id = Column(Integer, primary_key=True, index=True, comment="Primary Key ID")
    create_time = Column(DateTime, default=datetime.utcnow, comment="Creation time")
    update_time = Column(DateTime, default=None, onupdate=datetime.utcnow, comment="Update time")
    create_by = Column(String, default=None, comment="creator")
    update_by = Column(String, default=None, comment="updater")

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine=create_engine(os.getenv('DATABASE_URL','sqlite:///./windscope.db'),pool_pre_ping=True)
SessionLocal=sessionmaker(engine,expire_on_commit=False)


def db_session():
    with SessionLocal() as db:
        yield db

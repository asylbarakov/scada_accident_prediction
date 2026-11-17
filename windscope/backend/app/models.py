from datetime import datetime
from sqlalchemy import String, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from app.domain import utc_now

J=JSON().with_variant(JSONB,'postgresql')


class Base(DeclarativeBase):pass


class Scenario(Base):
    __tablename__='scenarios'
    id:Mapped[str]=mapped_column(String,primary_key=True)
    asset_id:Mapped[str]=mapped_column(String,index=True)
    source_hash:Mapped[str]=mapped_column(String)
    rows:Mapped[int]=mapped_column(Integer)
    start:Mapped[datetime]=mapped_column(DateTime)
    end:Mapped[datetime]=mapped_column(DateTime)
    prediction_start:Mapped[datetime]=mapped_column(DateTime)
    prediction_end:Mapped[datetime]=mapped_column(DateTime)
    prediction_points:Mapped[int]=mapped_column(Integer)
    annotation:Mapped[dict]=mapped_column(J)


class Sensor(Base):
    __tablename__='sensors'
    key:Mapped[str]=mapped_column(String,primary_key=True)
    description:Mapped[str]=mapped_column(String)
    unit:Mapped[str]=mapped_column(String)


class Measurement(Base):
    __tablename__='measurements'
    scenario_id:Mapped[str]=mapped_column(ForeignKey('scenarios.id'),primary_key=True)
    row_id:Mapped[int]=mapped_column(Integer,primary_key=True)
    source_at:Mapped[datetime]=mapped_column(DateTime)
    partition:Mapped[str]=mapped_column(String)
    values:Mapped[dict]=mapped_column(J)
    status_type_id:Mapped[int]=mapped_column(Integer)
    __table_args__=(Index('ix_measurement_time','scenario_id','source_at'),)


class ReplaySession(Base):
    __tablename__='replay_sessions'
    id:Mapped[str]=mapped_column(String,primary_key=True)
    scenario_id:Mapped[str]=mapped_column(ForeignKey('scenarios.id'),index=True)
    model_id:Mapped[str]=mapped_column(String,default='swtfps-scada-xgboost-v1.0.0',server_default='swtfps-scada-xgboost-v1.0.0')
    continuous:Mapped[bool]=mapped_column(Boolean,default=False,server_default='false')
    cycle:Mapped[int]=mapped_column(Integer,default=1,server_default='1')
    status:Mapped[str]=mapped_column(String,default='created')
    speed:Mapped[float]=mapped_column(Float,default=600)
    generation:Mapped[int]=mapped_column(Integer,default=0)
    revision:Mapped[int]=mapped_column(Integer,default=0)
    sequence:Mapped[int]=mapped_column(Integer,default=0)
    position:Mapped[int]=mapped_column(Integer,default=0)
    total:Mapped[int]=mapped_column(Integer)
    cursor:Mapped[int]=mapped_column(Integer,default=-1)
    start:Mapped[datetime]=mapped_column(DateTime)
    end:Mapped[datetime]=mapped_column(DateTime)
    virtual_time:Mapped[datetime|None]=mapped_column(DateTime,nullable=True)
    due_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)
    remaining:Mapped[float]=mapped_column(Float,default=0)
    lease:Mapped[str|None]=mapped_column(String,nullable=True)
    lease_until:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utc_now)
    __table_args__=(Index('ix_replay_due','status','due_at'),)


class Prediction(Base):
    __tablename__='predictions'
    id:Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    session_id:Mapped[str]=mapped_column(ForeignKey('replay_sessions.id'))
    generation:Mapped[int]=mapped_column(Integer)
    row_id:Mapped[int]=mapped_column(Integer)
    source_at:Mapped[datetime]=mapped_column(DateTime)
    processed_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utc_now)
    result:Mapped[dict]=mapped_column(J)
    __table_args__=(UniqueConstraint('session_id','generation','row_id'),Index('ix_prediction_session','session_id','generation','source_at'))


class Alert(Base):
    __tablename__='alerts'
    id:Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    session_id:Mapped[str]=mapped_column(ForeignKey('replay_sessions.id'))
    generation:Mapped[int]=mapped_column(Integer)
    start:Mapped[datetime]=mapped_column(DateTime)
    last:Mapped[datetime]=mapped_column(DateTime)
    end:Mapped[datetime|None]=mapped_column(DateTime,nullable=True)
    points:Mapped[int]=mapped_column(Integer,default=1)
    severity:Mapped[str]=mapped_column(String,default='warning')
    model_id:Mapped[str]=mapped_column(String)
    __table_args__=(Index('ix_alert_session','session_id','generation','start'),)


class ReplayEvent(Base):
    __tablename__='replay_events'
    session_id:Mapped[str]=mapped_column(ForeignKey('replay_sessions.id'),primary_key=True)
    sequence:Mapped[int]=mapped_column(Integer,primary_key=True)
    generation:Mapped[int]=mapped_column(Integer)
    type:Mapped[str]=mapped_column(String)
    payload:Mapped[dict]=mapped_column(J)
    emitted_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utc_now)


class AlertRuntime(Base):
    __tablename__='alert_runtimes'
    session_id:Mapped[str]=mapped_column(ForeignKey('replay_sessions.id'),primary_key=True)
    generation:Mapped[int]=mapped_column(Integer,primary_key=True)
    policy:Mapped[dict]=mapped_column(J)
    state:Mapped[dict]=mapped_column(J,default=dict)


class Diagnostic(Base):
    __tablename__='diagnostics'
    session_id:Mapped[str]=mapped_column(ForeignKey('replay_sessions.id'),primary_key=True)
    generation:Mapped[int]=mapped_column(Integer,primary_key=True)
    row_id:Mapped[int]=mapped_column(Integer,primary_key=True)
    source_at:Mapped[datetime]=mapped_column(DateTime)
    result:Mapped[dict]=mapped_column(J)


class OperationalAlert(Base):
    __tablename__='operational_alerts'
    id:Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    session_id:Mapped[str]=mapped_column(ForeignKey('replay_sessions.id'))
    generation:Mapped[int]=mapped_column(Integer)
    asset_id:Mapped[str]=mapped_column(String)
    first_signal:Mapped[datetime]=mapped_column(DateTime)
    start:Mapped[datetime]=mapped_column(DateTime)
    last:Mapped[datetime]=mapped_column(DateTime)
    end:Mapped[datetime|None]=mapped_column(DateTime,nullable=True)
    status:Mapped[str]=mapped_column(String,default='active')
    severity:Mapped[str]=mapped_column(String)
    subsystem:Mapped[str|None]=mapped_column(String,nullable=True)
    points:Mapped[int]=mapped_column(Integer,default=1)
    detail:Mapped[dict]=mapped_column(J)
    policy:Mapped[dict]=mapped_column(J)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=utc_now)
    acknowledged_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)
    resolved_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)
    resolution_reason:Mapped[str|None]=mapped_column(String,nullable=True)
    __table_args__=(Index('ix_operational_alert_session','session_id','generation','status'),Index('ix_operational_alert_asset','asset_id','start'))

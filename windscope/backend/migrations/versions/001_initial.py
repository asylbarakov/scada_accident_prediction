"""Initial MVP schema; existing research and raw data remain external."""
from alembic import op
from app.models import Base
revision='001'
down_revision=None
branch_labels=None
depends_on=None


def upgrade():
    Base.metadata.create_all(op.get_bind(),tables=[Base.metadata.tables[name] for name in ('scenarios','sensors','measurements','replay_sessions','predictions','alerts','replay_events')])


def downgrade():
    Base.metadata.drop_all(op.get_bind(),tables=[Base.metadata.tables[name] for name in ('scenarios','sensors','measurements','replay_sessions','predictions','alerts','replay_events')])

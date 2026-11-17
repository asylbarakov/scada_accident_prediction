"""Add v2 diagnostics and alert lifecycle without modifying validated predictions."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision='002'
down_revision='001'
branch_labels=None
depends_on=None


def upgrade():
    op.create_table('alert_runtimes',sa.Column('session_id',sa.String(),sa.ForeignKey('replay_sessions.id'),primary_key=True),
        sa.Column('generation',sa.Integer(),primary_key=True),sa.Column('policy',JSONB(),nullable=False),sa.Column('state',JSONB(),nullable=False))
    op.create_table('diagnostics',sa.Column('session_id',sa.String(),sa.ForeignKey('replay_sessions.id'),primary_key=True),
        sa.Column('generation',sa.Integer(),primary_key=True),sa.Column('row_id',sa.Integer(),primary_key=True),
        sa.Column('source_at',sa.DateTime(),nullable=False),sa.Column('result',JSONB(),nullable=False))
    op.create_table('operational_alerts',sa.Column('id',sa.Integer(),primary_key=True),
        sa.Column('session_id',sa.String(),sa.ForeignKey('replay_sessions.id'),nullable=False),sa.Column('generation',sa.Integer(),nullable=False),
        sa.Column('asset_id',sa.String(),nullable=False),sa.Column('first_signal',sa.DateTime(),nullable=False),
        sa.Column('start',sa.DateTime(),nullable=False),sa.Column('last',sa.DateTime(),nullable=False),sa.Column('end',sa.DateTime()),
        sa.Column('status',sa.String(),nullable=False),sa.Column('severity',sa.String(),nullable=False),sa.Column('subsystem',sa.String()),
        sa.Column('points',sa.Integer(),nullable=False),sa.Column('detail',JSONB(),nullable=False),sa.Column('policy',JSONB(),nullable=False),
        sa.Column('created_at',sa.DateTime(timezone=True),nullable=False),sa.Column('acknowledged_at',sa.DateTime(timezone=True)),
        sa.Column('resolved_at',sa.DateTime(timezone=True)),sa.Column('resolution_reason',sa.String()))
    op.create_index('ix_operational_alert_session','operational_alerts',['session_id','generation','status'])
    op.create_index('ix_operational_alert_asset','operational_alerts',['asset_id','start'])


def downgrade():
    op.drop_table('operational_alerts');op.drop_table('diagnostics');op.drop_table('alert_runtimes')

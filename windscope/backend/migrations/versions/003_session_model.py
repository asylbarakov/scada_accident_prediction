"""Pin each replay to its model; old sessions and predictions retain their identity."""
from alembic import op
import sqlalchemy as sa

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade():
    # Migration 001 uses current metadata when installing a fresh database.
    names = {c['name'] for c in sa.inspect(op.get_bind()).get_columns('replay_sessions')}
    if 'model_id' not in names:
        op.add_column('replay_sessions', sa.Column('model_id', sa.String(), nullable=False,
                      server_default='swtfps-scada-xgboost-v1.0.0'))


def downgrade():
    op.drop_column('replay_sessions', 'model_id')

"""Explicit historical repeat mode; existing finite sessions stay finite."""
from alembic import op
import sqlalchemy as sa
revision='004'
down_revision='003'
branch_labels=None
depends_on=None


def upgrade():
    names={c['name'] for c in sa.inspect(op.get_bind()).get_columns('replay_sessions')}
    if 'continuous' not in names:
        op.add_column('replay_sessions',sa.Column('continuous',sa.Boolean(),nullable=False,server_default=sa.false()))
    if 'cycle' not in names:
        op.add_column('replay_sessions',sa.Column('cycle',sa.Integer(),nullable=False,server_default='1'))


def downgrade():
    op.drop_column('replay_sessions','cycle')
    op.drop_column('replay_sessions','continuous')

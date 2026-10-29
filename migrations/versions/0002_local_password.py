"""Add local password authentication."""
from alembic import op
import sqlalchemy as sa

revision = "0002_local_password"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("users", sa.Column("password_hash", sa.String(256), nullable=True))

def downgrade():
    op.drop_column("users", "password_hash")

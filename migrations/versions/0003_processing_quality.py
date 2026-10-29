"""Add processing status and event deduplication metadata."""
from alembic import op
import sqlalchemy as sa

revision = "0003_processing_quality"
down_revision = "0002_local_password"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("processing_sessions", sa.Column("status", sa.String(20), nullable=False, server_default="running"))
    op.add_column("processing_sessions", sa.Column("error_message", sa.String(1000), nullable=True))
    op.add_column("vehicle_events", sa.Column("event_key", sa.String(64), nullable=True))
    op.add_column("vehicle_events", sa.Column("source_timestamp", sa.Float(), nullable=False, server_default="0"))
    op.execute("UPDATE vehicle_events SET event_key = md5(id::text || ':' || session_id::text || ':' || track_id::text || ':' || event_time::text || ':' || x::text || ':' || y::text) WHERE event_key IS NULL")
    op.alter_column("vehicle_events", "event_key", nullable=False)
    op.create_index("ix_vehicle_events_event_key", "vehicle_events", ["event_key"], unique=True)

def downgrade():
    op.drop_index("ix_vehicle_events_event_key", table_name="vehicle_events")
    op.drop_column("vehicle_events", "source_timestamp")
    op.drop_column("vehicle_events", "event_key")
    op.drop_column("processing_sessions", "error_message")
    op.drop_column("processing_sessions", "status")

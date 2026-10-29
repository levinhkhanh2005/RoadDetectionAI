"""Initial PostgreSQL schema."""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table("roles", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("name", sa.String(32), nullable=False, unique=True))
    op.create_table("users", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("email", sa.String(320), nullable=False, unique=True), sa.Column("name", sa.String(200), nullable=False, server_default=""), sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.create_table("user_roles", sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True), sa.Column("role_id", sa.Integer(), sa.ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True))
    op.create_table("locations", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("name", sa.String(200), nullable=False, unique=True), sa.Column("address", sa.String(500), nullable=False, server_default=""), sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.create_table("cameras", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("location_id", sa.Integer(), sa.ForeignKey("locations.id"), nullable=False), sa.Column("name", sa.String(200), nullable=False), sa.Column("source", sa.String(500), nullable=False, server_default=""), sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.create_index("ix_cameras_location_id", "cameras", ["location_id"])
    op.create_table("processing_sessions", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("camera_id", sa.Integer(), sa.ForeignKey("cameras.id"), nullable=False), sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id")), sa.Column("source_name", sa.String(500), nullable=False, server_default=""), sa.Column("started_at", sa.DateTime(timezone=True), nullable=False), sa.Column("finished_at", sa.DateTime(timezone=True)))
    op.create_index("ix_processing_sessions_camera_id", "processing_sessions", ["camera_id"])
    op.create_table("vehicle_events", sa.Column("id", sa.BigInteger(), primary_key=True), sa.Column("session_id", sa.Integer(), sa.ForeignKey("processing_sessions.id", ondelete="CASCADE"), nullable=False), sa.Column("camera_id", sa.Integer(), sa.ForeignKey("cameras.id"), nullable=False), sa.Column("location_id", sa.Integer(), sa.ForeignKey("locations.id"), nullable=False), sa.Column("event_time", sa.DateTime(timezone=True), nullable=False), sa.Column("track_id", sa.Integer(), nullable=False), sa.Column("vehicle_type", sa.String(32), nullable=False), sa.Column("vehicle_type_vi", sa.String(64), nullable=False, server_default="Xe"), sa.Column("direction", sa.String(32), nullable=False, server_default=""), sa.Column("x", sa.Integer(), nullable=False), sa.Column("y", sa.Integer(), nullable=False), sa.Column("speed_kmh", sa.Float(), nullable=False, server_default="0"), sa.Column("speed_avg_kmh", sa.Float(), nullable=False, server_default="0"), sa.Column("speed_max_kmh", sa.Float(), nullable=False, server_default="0"))
    for name, cols in [("ix_vehicle_events_session_id", ["session_id"]), ("ix_vehicle_events_camera_id", ["camera_id"]), ("ix_vehicle_events_location_id", ["location_id"]), ("ix_vehicle_events_event_time", ["event_time"]), ("ix_vehicle_events_vehicle_type", ["vehicle_type"]), ("ix_vehicle_events_filter", ["event_time", "camera_id", "vehicle_type"])]:
        op.create_index(name, "vehicle_events", cols)
    op.execute("INSERT INTO roles (name) VALUES ('admin'), ('viewer')")

def downgrade():
    op.drop_table("vehicle_events")
    op.drop_table("processing_sessions")
    op.drop_table("cameras")
    op.drop_table("locations")
    op.drop_table("user_roles")
    op.drop_table("users")
    op.drop_table("roles")

"""PostgreSQL persistence for vehicle events and application users."""

from __future__ import annotations

import os
import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, create_engine, select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Role(Base):
    __tablename__ = "roles"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(32), unique=True)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    password_hash: Mapped[str | None] = mapped_column(String(256), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    roles: Mapped[list[Role]] = relationship(secondary="user_roles", lazy="selectin")


class UserRole(Base):
    __tablename__ = "user_roles"
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)


class Location(Base):
    __tablename__ = "locations"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    address: Mapped[str] = mapped_column(String(500), default="")
    is_active: Mapped[bool] = mapped_column(default=True)


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(500), default="")
    is_active: Mapped[bool] = mapped_column(default=True)
    location: Mapped[Location] = relationship()


class ProcessingSession(Base):
    __tablename__ = "processing_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    source_name: Mapped[str] = mapped_column(String(500), default="")
    status: Mapped[str] = mapped_column(String(20), default="running", index=True)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class VehicleEvent(Base):
    __tablename__ = "vehicle_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("processing_sessions.id", ondelete="CASCADE"), index=True)
    event_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), index=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    track_id: Mapped[int] = mapped_column()
    vehicle_type: Mapped[str] = mapped_column(String(32), index=True)
    vehicle_type_vi: Mapped[str] = mapped_column(String(64), default="Xe")
    direction: Mapped[str] = mapped_column(String(32), default="")
    x: Mapped[int] = mapped_column()
    y: Mapped[int] = mapped_column()
    speed_kmh: Mapped[float] = mapped_column(default=0.0)
    speed_avg_kmh: Mapped[float] = mapped_column(default=0.0)
    speed_max_kmh: Mapped[float] = mapped_column(default=0.0)


Index("ix_vehicle_events_filter", VehicleEvent.event_time, VehicleEvent.camera_id, VehicleEvent.vehicle_type)


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("database_url")
        except Exception:
            url = None
    if not url:
        raise RuntimeError("Chưa cấu hình DATABASE_URL hoặc database_url trong .streamlit/secrets.toml")
    return str(url)


def make_session_factory():
    engine = create_engine(database_url(), pool_pre_ping=True, pool_size=10, max_overflow=20)
    return sessionmaker(bind=engine, expire_on_commit=False)


def sync_user(session, email: str, name: str, admin_emails: set[str]) -> User:
    user = session.scalar(select(User).where(User.email == email.lower()))
    if user is None:
        user = User(email=email.lower(), name=name or email)
        session.add(user)
        session.flush()
    else:
        user.name = name or user.name

    role_name = "admin" if email.lower() in admin_emails else "viewer"
    role = session.scalar(select(Role).where(Role.name == role_name))
    if role is None:
        raise RuntimeError("Database chưa có role admin/viewer; hãy chạy alembic upgrade head")
    user.roles = [role]
    session.commit()
    return user


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Mật khẩu phải có ít nhất 8 ký tự")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str | None) -> bool:
    if not encoded or not encoded.startswith("scrypt$"):
        return False
    try:
        _, salt_hex, digest_hex = encoded.split("$")
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def authenticate_local_user(session, username: str, password: str, bootstrap_username: str, bootstrap_password: str) -> User | None:
    username = username.strip().lower()
    user = session.scalar(select(User).where(User.email == username))
    if user is None and username == bootstrap_username.strip().lower():
        role = session.scalar(select(Role).where(Role.name == "admin"))
        if role is None:
            raise RuntimeError("Database chưa có role admin; hãy chạy migration")
        user = User(email=username, name=username, password_hash=hash_password(bootstrap_password))
        user.roles = [role]
        session.add(user)
        session.commit()
    if user and user.is_active and verify_password(password, user.password_hash):
        return user
    return None


def create_local_user(session, username: str, password: str, role_name: str = "viewer") -> User:
    username = username.strip().lower()
    if not username or "@" not in username:
        raise ValueError("Username phải là email hợp lệ")
    if session.scalar(select(User).where(User.email == username)):
        raise ValueError("Username đã tồn tại")
    role = session.scalar(select(Role).where(Role.name == role_name))
    if role is None:
        raise RuntimeError("Role không tồn tại")
    user = User(email=username, name=username, password_hash=hash_password(password))
    user.roles = [role]
    session.add(user)
    session.commit()
    return user


def user_has_role(user: User, role: str) -> bool:
    return any(item.name == role for item in user.roles)


def create_event(session, processing_session_id: int, camera_id: int, location_id: int, event: dict[str, Any]) -> None:
    create_events(session, processing_session_id, camera_id, location_id, [event])


def create_events(session, processing_session_id: int, camera_id: int, location_id: int, events: list[dict[str, Any]]) -> None:
    if not events:
        return
    rows = []
    for event in events:
        timestamp = event.get("timestamp")
        if isinstance(timestamp, str):
            event_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        else:
            event_time = datetime.now(timezone.utc)
        x, y = event.get("point", (0, 0))
        source_timestamp = float(event.get("frame_timestamp", 0.0))
        import hashlib
        event_key = hashlib.sha256(f"{processing_session_id}:{event['track_id']}:{event_time.isoformat()}:{x}:{y}".encode()).hexdigest()
        rows.append({
            "session_id": processing_session_id, "event_key": event_key, "camera_id": camera_id,
            "location_id": location_id, "event_time": event_time, "track_id": int(event["track_id"]),
            "vehicle_type": event["class_en"], "vehicle_type_vi": event.get("class_vi", "Xe"),
            "direction": event.get("direction", ""), "x": int(x), "y": int(y),
            "source_timestamp": source_timestamp, "speed_kmh": float(event.get("speed_kmh", 0)),
            "speed_avg_kmh": float(event.get("speed_avg_kmh", 0)), "speed_max_kmh": float(event.get("speed_max_kmh", 0)),
        })
    session.execute(pg_insert(VehicleEvent).values(rows).on_conflict_do_nothing(index_elements=["event_key"]))


def _legacy_create_event(session, processing_session_id: int, camera_id: int, location_id: int, event: dict[str, Any]) -> None:
    timestamp = event.get("timestamp")
    if isinstance(timestamp, str):
        event_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    else:
        event_time = datetime.now(timezone.utc)
    x, y = event.get("point", (0, 0))
    session.add(VehicleEvent(
        event_key=secrets.token_hex(32),
        session_id=processing_session_id, camera_id=camera_id, location_id=location_id,
        event_time=event_time, track_id=int(event["track_id"]), vehicle_type=event["class_en"],
        vehicle_type_vi=event.get("class_vi", "Xe"), direction=event.get("direction", ""),
        x=int(x), y=int(y), speed_kmh=float(event.get("speed_kmh", 0)),
        speed_avg_kmh=float(event.get("speed_avg_kmh", 0)), speed_max_kmh=float(event.get("speed_max_kmh", 0)),
    ))


def create_processing_session(session, camera_id: int, user_id: int | None, source_name: str) -> ProcessingSession:
    item = ProcessingSession(camera_id=camera_id, user_id=user_id, source_name=source_name)
    session.add(item)
    session.commit()
    return item


def finish_processing_session(session, item: ProcessingSession) -> None:
    item.status = "completed"
    item.finished_at = datetime.now(timezone.utc)
    session.commit()


def fail_processing_session(session, item: ProcessingSession, error_message: str) -> None:
    item.status = "failed"
    item.error_message = error_message[:1000]
    item.finished_at = datetime.now(timezone.utc)
    session.commit()


def list_locations(session) -> list[Location]:
    return list(session.scalars(select(Location).where(Location.is_active.is_(True)).order_by(Location.name)))


def list_cameras(session, location_id: int | None = None) -> list[Camera]:
    query = select(Camera).where(Camera.is_active.is_(True)).order_by(Camera.name)
    if location_id is not None:
        query = query.where(Camera.location_id == location_id)
    return list(session.scalars(query))


def list_all_cameras(session) -> list[Camera]:
    return list(session.scalars(select(Camera).order_by(Camera.name)))


def search_events(session, start: datetime, end: datetime, camera_id: int | None = None, location_id: int | None = None, vehicle_type: str | None = None, limit: int = 500, offset: int = 0):
    from sqlalchemy import text
    query = text("""
        SELECT e.event_time, l.name AS location, c.name AS camera,
               e.vehicle_type, e.vehicle_type_vi, e.direction, e.track_id, e.source_timestamp,
               e.speed_kmh, e.speed_avg_kmh, e.speed_max_kmh
        FROM vehicle_events e
        JOIN locations l ON l.id = e.location_id
        JOIN cameras c ON c.id = e.camera_id
        WHERE e.event_time >= :start AND e.event_time < :end
          AND (:camera_id IS NULL OR e.camera_id = :camera_id)
          AND (:location_id IS NULL OR e.location_id = :location_id)
          AND (:vehicle_type IS NULL OR e.vehicle_type = :vehicle_type)
        ORDER BY e.event_time DESC
        LIMIT :limit OFFSET :offset
    """)
    return session.execute(query, {"start": start, "end": end, "camera_id": camera_id, "location_id": location_id, "vehicle_type": vehicle_type, "limit": limit, "offset": offset}).mappings().all()


def count_events(session, start: datetime, end: datetime, camera_id: int | None = None, location_id: int | None = None, vehicle_type: str | None = None) -> int:
    query = select(func.count(VehicleEvent.id)).where(VehicleEvent.event_time >= start, VehicleEvent.event_time < end)
    if camera_id is not None:
        query = query.where(VehicleEvent.camera_id == camera_id)
    if location_id is not None:
        query = query.where(VehicleEvent.location_id == location_id)
    if vehicle_type is not None:
        query = query.where(VehicleEvent.vehicle_type == vehicle_type)
    return int(session.scalar(query) or 0)


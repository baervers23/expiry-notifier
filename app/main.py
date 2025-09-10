import os, sqlite3, requests, json, time, threading, base64, jwt, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# ---------------------------
# ENV
# ---------------------------
load_dotenv()
FETCH_INTERVAL=3600 # 3600s = 60min

JFA_USER=os.getenv("JFA_USER")
JFA_PASS=os.getenv("JFA_PASS")
JFA_API_URL=os.getenv("JFA_API_URL")
PATH_TOKEN=os.getenv("PATH_TOKEN")
PATH_USERS=os.getenv("PATH_USERS")

LOG_FILE = os.getenv("LOG_FILE", "/logs/app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "debug").lower()

DB_PATH = os.getenv("DB_PATH", "/data/members.db")
DB_CLEANSTART = os.getenv("DB_CLEANSTART", "false").lower() == "true"

# ---------------------------
# Logging & Debug
# ---------------------------
def setup_logging(log_file: str, log_level: str = "debug"):
    import logging
    from logging.handlers import RotatingFileHandler

    level = getattr(logging, log_level.upper(), logging.DEBUG)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[file_handler, console_handler])

def debug_env():
    logging.debug("===== ENV DEBUG START =====")
    for k, v in os.environ.items():
        if any(x in k.upper() for x in ["PASS", "SECRET", "TOKEN"]):
            logging.debug(f"{k} = ***HIDDEN***")
        else:
            logging.debug(f"{k} = {v}")
    logging.debug("===== ENV DEBUG END =====")
    logging.debug("")


setup_logging(LOG_FILE, LOG_LEVEL)



# ---------------------------
# Settings (Pydantic)
# ---------------------------
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import BaseModel

class Settings(BaseSettings):
    ORG_JWT_SECRET: Optional[str] = os.getenv("ORG_JWT_SECRET")
    VERIFY_ORG_JWT: bool = (os.getenv("VERIFY_ORG_JWT", "true").lower() == "true")
    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "*")

settings = Settings()

# ---------------------------
# App & CORS
# ---------------------------
app = FastAPI(title="Reminder API", version="1.0.0")
origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Models
# ---------------------------
class MembershipRequest(BaseModel):
    username: Optional[str] = None

class MembershipResponse(BaseModel):
    username: str
    expiry_epoch: int
    days_left: int
    expiry_date: str

# ---------------------------
# DB
# ---------------------------
def reset_database():

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS memberships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            disabled INTEGER DEFAULT 0,
            expiry_epoch INTEGER,
            expiry_iso TEXT,
            jellyfin_id TEXT,
            fetched_at TEXT,
            raw_json TEXT
        )
        """)
        conn.commit()

def init_db():
    logging.info("")
    
    if DB_CLEANSTART:
        reset_database()
        logging.info("🗑️ removed old db and recreated, ready.")
        
    logging.debug(f"🔄 init db: {DB_PATH}")
    dirn = os.path.dirname(DB_PATH)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    if not os.path.exists(DB_PATH):
        logging.debug(f"❌ DB file not found, creating new: {DB_PATH}")
        open(DB_PATH, "a").close()
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS memberships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            disabled INTEGER DEFAULT 0,
            expiry_epoch INTEGER,
            expiry_iso TEXT,
            jellyfin_id TEXT,
            fetched_at TEXT,
            raw_json TEXT
        )
        """)
        # Reset Autoincrement
        c.execute("DELETE FROM sqlite_sequence WHERE name='memberships'")
        conn.commit()
    logging.info("✅ done, DB ready.")
    logging.info("")

# ---------------------------
# Loop
# ---------------------------
def loop():
    init_db()
    if LOG_LEVEL == "debug":
        debug_env()
    while True:
        try:
            token = fetch_token()
            if token:
                users = fetch_users(token)
                if users:
                    store_users(users)
        except Exception as e:
            import traceback
            logging.error(f"❌ error in loop: {e}")
            logging.error(traceback.format_exc())
        time.sleep(FETCH_INTERVAL)

# ---------------------------
# Helpers
# ---------------------------
def now_epoch() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())

def at_least_zero_days_from_epoch(expiry_epoch: int) -> int:
    return max(0, (expiry_epoch - now_epoch() + 86399) // 86400)

def iso_from_epoch(expiry_epoch: int) -> str:
    return datetime.fromtimestamp(expiry_epoch, tz=timezone.utc).isoformat()

def decode_org_jwt(bearer: str) -> dict:
    if not bearer or not bearer.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing_token")
    token = bearer.split(" ", 1)[1].strip()
    try:
        if settings.VERIFY_ORG_JWT and settings.ORG_JWT_SECRET:
            return jwt.decode(token, settings.ORG_JWT_SECRET, algorithms=["HS256"])
        else:
            return jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="invalid_token")

# ---------------------------
# DB Operations
# ---------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def store_users(users):
    if isinstance(users, dict) and "users" in users:
        users = users["users"]
    if not isinstance(users, list):
        logging.warning("⚠️ Wrong format; expect list.")
        return

    conn = get_db()
    c = conn.cursor()
    api_usernames = [u.get("name") for u in users if isinstance(u, dict) and u.get("name")]
    db_usernames = [row[0] for row in c.execute("SELECT username FROM memberships").fetchall()]

    removed, inserted, updated, skipped = 0, 0, 0, 0

    for old_user in db_usernames:
        if old_user not in api_usernames:
            logging.info(f"🗑️ remove user: {old_user}, missing in API")
            c.execute("DELETE FROM memberships WHERE username=?", (old_user,))
            removed += 1

    upsert_logs = []

    for entry in users:
        if not isinstance(entry, dict):
            continue
        username = entry.get("name")
        if not username:
            logging.warning(f"⚠️ skipped user without username: {entry}")
            skipped += 1
            continue

        email = entry.get("email")
        jellyfin_id = entry.get("id")
        disabled = 1 if entry.get("disabled", False) else 0
        expiry_epoch = entry.get("expiry")
        if isinstance(expiry_epoch, str) and expiry_epoch.isdigit():
            expiry_epoch = int(expiry_epoch)
        elif not isinstance(expiry_epoch, (int, float)):
            expiry_epoch = None
        expiry_iso = iso_from_epoch(expiry_epoch) if expiry_epoch else None
        days_left = at_least_zero_days_from_epoch(expiry_epoch) if expiry_epoch else None
        raw_json = json.dumps(entry, ensure_ascii=False)

        try:
            exists = c.execute("SELECT 1 FROM memberships WHERE username=? LIMIT 1", (username,)).fetchone()
            c.execute(
                """
                INSERT INTO memberships (username, email, expiry_epoch, expiry_iso, disabled, jellyfin_id, fetched_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    email=excluded.email,
                    expiry_epoch=excluded.expiry_epoch,
                    expiry_iso=excluded.expiry_iso,
                    disabled=excluded.disabled,
                    jellyfin_id=excluded.jellyfin_id,
                    fetched_at=excluded.fetched_at,
                    raw_json=excluded.raw_json
                """,
                (
                    username, email,
                    int(expiry_epoch) if expiry_epoch is not None else None,
                    expiry_iso, disabled,
                    jellyfin_id, datetime.utcnow().isoformat(),
                    raw_json,
                ),
            )
            if exists:
                updated += 1
            else:
                inserted += 1

            upsert_logs.append((username, days_left))

        except Exception as e:
            logging.warning(f"❌ DB error for {username}: {e}")

    # Tabelle für Upsert-Logs erstellen
    if upsert_logs:
        def format_upsert_table(logs):
            header = f"                             | {'Username':20} | {'Days Left':9} |\n                             |{'-'*22}|{'-'*11}|"
            rows = [f"                             | {u:20} | {str(d) if d is not None else 'N/A':9} |" for u, d in logs]
            return "\n".join([header] + rows)

        table = format_upsert_table(upsert_logs)
        logging.debug("")
        logging.debug(f"💾 Upserted users:\n{table}")

    conn.commit()
    conn.close()
    logging.info("")
    logging.info(f"✅ refreshed DB: {len(api_usernames)} API users, {inserted} inserted, {updated} updated, {removed} removed, {skipped} skipped")


def store_usersx(users):
    if isinstance(users, dict) and "users" in users:
        users = users["users"]
    if not isinstance(users, list):
        logging.warning("⚠️ Wrong format; expect list.")
        return

    conn = get_db()
    c = conn.cursor()
    api_usernames = [u.get("name") for u in users if isinstance(u, dict) and u.get("name")]
    db_usernames = [row[0] for row in c.execute("SELECT username FROM memberships").fetchall()]

    removed, inserted, updated, skipped = 0, 0, 0, 0
    for old_user in db_usernames:
        if old_user not in api_usernames:
            logging.info(f"🗑️ remove user: {old_user}, missing in API")
            c.execute("DELETE FROM memberships WHERE username=?", (old_user,))
            removed += 1

    for entry in users:
        if not isinstance(entry, dict):
            continue
        username = entry.get("name")
        if not username:
            logging.warning(f"⚠️ skipped user without username: {entry}")
            skipped += 1
            continue

        email = entry.get("email")
        jellyfin_id = entry.get("id")
        disabled = 1 if entry.get("disabled", False) else 0
        expiry_epoch = entry.get("expiry")
        if isinstance(expiry_epoch, str) and expiry_epoch.isdigit():
            expiry_epoch = int(expiry_epoch)
        elif not isinstance(expiry_epoch, (int, float)):
            expiry_epoch = None
        expiry_iso = iso_from_epoch(expiry_epoch) if expiry_epoch else None
        days_left = at_least_zero_days_from_epoch(expiry_epoch) if expiry_epoch else None
        raw_json = json.dumps(entry, ensure_ascii=False)

        try:
            exists = c.execute("SELECT 1 FROM memberships WHERE username=? LIMIT 1", (username,)).fetchone()
            c.execute(
                """
                INSERT INTO memberships (username, email, expiry_epoch, expiry_iso, disabled, jellyfin_id, fetched_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    email=excluded.email,
                    expiry_epoch=excluded.expiry_epoch,
                    expiry_iso=excluded.expiry_iso,
                    disabled=excluded.disabled,
                    jellyfin_id=excluded.jellyfin_id,
                    fetched_at=excluded.fetched_at,
                    raw_json=excluded.raw_json
                """,
                (
                    username, email,
                    int(expiry_epoch) if expiry_epoch is not None else None,
                    expiry_iso, disabled,
                    jellyfin_id, datetime.utcnow().isoformat(),
                    raw_json,
                ),
            )
            if exists:
                updated += 1
            else:
                inserted += 1
            logging.debug("")
            logging.debug(f"💾 upsert user: {username}, days_left={days_left}")
        except Exception as e:
            logging.warning(f"❌ DB error for {username}: {e}")

    conn.commit()
    conn.close()
    logging.info("")
    logging.info(f"✅ refreshed DB: {len(api_usernames)} API users, {inserted} inserted, {updated} updated, {removed} removed, {skipped} skipped")

# ---------------------------
# API Calls
# ---------------------------
def fetch_token():
    AUTH = base64.b64encode(f"{JFA_USER}:{JFA_PASS}".encode()).decode()
    TOKEN_URL = f"{JFA_API_URL.rstrip('/')}{PATH_TOKEN}"
    try:
        logging.debug(f"📡 Get Token via BasicAuth -> {TOKEN_URL}")
        headers={"Authorization": f"Basic {AUTH}", 'Accept': 'application/json'}
        res = requests.get(TOKEN_URL, headers=headers, timeout=15)
        res.raise_for_status()
        return res.json().get("token")
    except Exception as e:
        logging.warning(f"❌ Error while fetching token: {e}")
        return None

def fetch_users(token):
    USERS_URL = f"{JFA_API_URL.rstrip('/')}{PATH_USERS}"
    try:
        logging.debug("")
        logging.debug(f"📡 Fetch User via Token -> {USERS_URL}")
        headers = {"Authorization": f"Bearer {token}"}
        res = requests.get(USERS_URL, headers=headers, timeout=15)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logging.warning(f"❌ Error while fetching users: {e}")
        return None
    logging.debug("")

# ---------------------------
# FastAPI Routes
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/membership", response_model=MembershipResponse)
def membership(body: MembershipRequest, authorization: str = Header(default=None, convert_underscores=False)):
    payload = decode_org_jwt(authorization)
    username  = body.username  or payload.get("username") or payload.get("name") or payload.get("sub")
    email     = payload.get("email")

    if not username and not email:
        raise HTTPException(status_code=400, detail="missing_user_identifier")

    with get_db() as conn:
        row = conn.execute(
            "SELECT id, username, email, expiry_epoch FROM memberships WHERE lower(username)=lower(?) LIMIT 1",
            (username,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="user_not_found")
        row = dict(row)

    expiry_epoch = int(row["expiry_epoch"])
    days_left    = at_least_zero_days_from_epoch(expiry_epoch)

    return MembershipResponse(
        username=row.get("username") or username or "",
        expiry_epoch=expiry_epoch,
        days_left=days_left,
        expiry_date=iso_from_epoch(expiry_epoch),
    )

# ---------------------------
# Startup Event → Loop
# ---------------------------
@app.on_event("startup")
def start_background_tasks():
    threading.Thread(target=loop, daemon=True).start()

# ---------------------------
# Main Loop
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_config=None)
    logging.info("FastAPI running on http://0.0.0.0:8000")

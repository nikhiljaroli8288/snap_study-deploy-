from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import certifi, uuid, os

DB_ERROR_MSG = "Database is unreachable. Please check your MongoDB Atlas IP whitelist and cluster status."

MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is not set")

client = MongoClient(
    MONGO_URI,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=10000,
)
db = client["SnapStudy"]
users_collection = db["users"]
history_collection = db["history"]
study_data_collection = db["study_data"]

_indexes_created = False

def _ensure_indexes():
    """Create indexes once, lazily on first DB operation."""
    global _indexes_created
    if _indexes_created:
        return
    try:
        users_collection.create_index("email", unique=True)
        history_collection.create_index([("email", 1), ("created_at", -1)])
        study_data_collection.create_index("key", unique=True)
    except Exception as e:
        print(f"[testdb] Warning: could not create indexes: {e}")
    _indexes_created = True


def create_user(name, email, password):
    """Register a new user. Returns (True, message) or (False, error)."""
    try:
        _ensure_indexes()
        if users_collection.find_one({"email": email}):
            return False, "An account with this email already exists."
        hashed = generate_password_hash(password)
        users_collection.insert_one({
            "name": name,
            "email": email,
            "password": hashed,
        })
        return True, "Account created successfully."
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        print(f"[testdb] DB connection error: {e}")
        return False, DB_ERROR_MSG


def verify_user(email, password):
    """Check credentials. Returns (True, user_doc) or (False, error)."""
    try:
        _ensure_indexes()
        user = users_collection.find_one({"email": email})
        if not user:
            return False, "No account found with this email."
        if not check_password_hash(user["password"], password):
            return False, "Incorrect password."
        return True, user
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        print(f"[testdb] DB connection error: {e}")
        return False, DB_ERROR_MSG


# ── History helpers ──────────────────────────────────────────────────────

def save_history(email, video_title, video_url, data_key):
    """Save a history entry for the user."""
    try:
        history_collection.insert_one({
            "email": email,
            "video_title": video_title,
            "video_url": video_url,
            "data_key": data_key,
            "created_at": datetime.utcnow(),
        })
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        print(f"[testdb] DB connection error in save_history: {e}")


def get_history(email, days=7):
    """Return user's history entries from the last N days, newest first."""
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        cursor = history_collection.find(
            {"email": email, "created_at": {"$gte": cutoff}},
            {"_id": 0, "video_title": 1, "video_url": 1, "data_key": 1, "created_at": 1},
        ).sort("created_at", -1)
        return list(cursor)
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        print(f"[testdb] DB connection error in get_history: {e}")
        return []


# ── Study data storage ───────────────────────────────────────────────────

def save_study_data(data, video_url):
    """Save generated study data to MongoDB. Returns a unique key."""
    try:
        key = str(uuid.uuid4())
        study_data_collection.insert_one({
            "key": key,
            "data": data,
            "video_url": video_url,
            "created_at": datetime.utcnow(),
        })
        return key
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        print(f"[testdb] DB connection error in save_study_data: {e}")
        raise ConnectionError(DB_ERROR_MSG) from e


def load_study_data(key):
    """Load study data from MongoDB by key. Returns (data, video_url) or (None, '')."""
    if not key:
        return None, ''
    try:
        doc = study_data_collection.find_one({"key": key}, {"_id": 0, "data": 1, "video_url": 1})
        if not doc:
            return None, ''
        return doc.get("data"), doc.get("video_url", '')
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        print(f"[testdb] DB connection error in load_study_data: {e}")
        return None, ''


def find_study_data_by_url(video_url):
    """Find existing study data for a video URL (cache). Returns (data, key) or (None, None)."""
    if not video_url:
        return None, None
    try:
        doc = study_data_collection.find_one(
            {"video_url": video_url},
            {"_id": 0, "data": 1, "key": 1},
            sort=[("created_at", -1)],
        )
        if doc:
            return doc.get("data"), doc.get("key")
        return None, None
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        print(f"[testdb] DB connection error in find_study_data_by_url: {e}")
        return None, None

# System constants
RECOGNITION_THRESHOLD = 0.6
UNLOCK_DURATION = 3  # seconds
MODE_CHECK_INTERVAL = 10  # seconds

# File paths
AUTHORIZED_FACES_FILE = "authorized_faces.pkl"
API_USERS_FILE = "api_users.pkl"

# API Configuration (can be overridden)
API_BASE_URL = None
API_HEADERS = {
    'Content-Type': 'application/json'
}
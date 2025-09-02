from pathlib import Path

# Home Directory
HOME_DIR = Path(Path(__file__).resolve()).parent.parent

# Subdirectories
ASSETS_DIR = Path(HOME_DIR, "assets")
CONFIG_DIR = Path(HOME_DIR, "config")
PAGES_DIR = Path(HOME_DIR, "pages")
CSS_DIR = Path(CONFIG_DIR, "css")

# Database and Data Paths
DATA_DIR = Path(HOME_DIR, "data")
CHROMADB_DIR = Path(DATA_DIR, "chromadb")
TEMP_DIR = Path(DATA_DIR, "temp")

APPS = [
    {"name": "NextGen OpsMate",
     "description": "The IT Services Incident Response Assistant is an AI-powered app that helps IT support teams quickly resolve incidents. By analyzing incident descriptions, it provides step-by-step instructions, escalation paths, and links to relevant documentation, significantly reducing resolution time.",
     "page": "opsmate.py",
     "page_icon": ":material/chat:",
     "image_icon": "opsmate.gif",
     "access_privilege_role": ["user"],
    },
    {"name": "Incident Analytics",
     "description": "Incident ticket analytics provide insights into the volume, types, resolution times, and trends of IT support issues. This data helps identify bottlenecks, improve processes, and optimize resource allocation for faster and more efficient problem resolution.",
     "page": "analytics.py",
     "page_icon": ":material/analytics:",
     "image_icon": "analytics.png",
     "access_privilege_role": ["analyst"],
    },
]

API_MODELS = [
"gemini-1.5-flash"
]

EMBEDDING_MODEL = "nomic-embed-text"

OLLAMA_MODEL_PROVIDER = "ollama"
API_MODEL_PROVIDER = "google_genai"


CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


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

API_MODELS = [
"gemini-1.5-flash"
]

EMBEDDING_MODEL = "nomic-embed-text"

OLLAMA_MODEL_PROVIDER = "ollama"
API_MODEL_PROVIDER = "google_genai"

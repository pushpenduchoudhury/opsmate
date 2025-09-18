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
    # {"name": "NextGen OpsMate",
    #  "description": "The IT Services Incident Response Assistant is an AI-powered app that helps IT support teams quickly resolve incidents. By analyzing incident descriptions, it provides step-by-step instructions, escalation paths, and links to relevant documentation, significantly reducing resolution time.",
    #  "page": "opsmate.py",
    #  "page_icon": ":material/chat:",
    #  "image_icon": "opsmate.gif",
    #  "access_privilege_role": ["user"],
    # },
    {"name": "NextGen OpsMate (Voice)",
     "description": "The IT Services Incident Response Assistant is an AI-powered app that helps IT support teams quickly resolve incidents. By analyzing incident descriptions, it provides step-by-step instructions, escalation paths, and links to relevant documentation, significantly reducing resolution time.",
     "page": "opsmate_voice.py",
     "page_icon": ":material/chat:",
     "image_icon": "opsmate.gif",
     "access_privilege_role": ["user"],
    },
    # {"name": "NextGen OpsMate (Minimalistic)",
    #  "description": "The IT Services Incident Response Assistant is an AI-powered app that helps IT support teams quickly resolve incidents. By analyzing incident descriptions, it provides step-by-step instructions, escalation paths, and links to relevant documentation, significantly reducing resolution time.",
    #  "page": "opsmate_minimalistic.py",
    #  "page_icon": ":material/chat:",
    #  "image_icon": "opsmate.gif",
    #  "access_privilege_role": ["user"],
    # },
    {"name": "Incident Analytics",
     "description": "Incident ticket analytics provide insights into the volume, types, resolution times, and trends of IT support issues. This data helps identify bottlenecks, improve processes, and optimize resource allocation for faster and more efficient problem resolution.",
     "page": "analytics.py",
     "page_icon": ":material/analytics:",
     "image_icon": "analytics.png",
     "access_privilege_role": ["analyst"],
    },
]

GEMINI_MODELS = [
    "gemini-1.5-flash"
]

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound",
    "groq/compound-mini",
    "qwen/qwen3-32b"
]

OPENAI_MODELS = []

EMBEDDING_MODEL = "nomic-embed-text"

OLLAMA_MODEL_PROVIDER = "ollama"
GOOGLE_MODEL_PROVIDER = "google_genai"
GROQ_MODEL_PROVIDER = "groq"
OPENAI_MODEL_PROVIDER = "openai"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

SPEECH_TO_TEXT_MODEL = "whisper-large-v3"
TEXT_TO_SPEECH_MODEL = "playai-tts"

TTS_ENGINES = {
    "native" : {"voices" : []},
    "groq" : {"voices" : ["Arista-PlayAI", "Atlas-PlayAI", "Basil-PlayAI", "Briggs-PlayAI", "Calum-PlayAI", "Celeste-PlayAI", "Cheyenne-PlayAI", "Chip-PlayAI", "Cillian-PlayAI", "Deedee-PlayAI", "Fritz-PlayAI", "Gail-PlayAI", "Indigo-PlayAI", "Mamaw-PlayAI", "Mason-PlayAI", "Mikail-PlayAI", "Mitch-PlayAI", "Quinn-PlayAI", "Thunder-PlayAI"]},
}

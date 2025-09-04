# Getting Started

### Clone repository using the following URL:
https://github.com/pushpenduchoudhury/opsmate.git

<br>


**1. Create virtual environment:**
```console
python -m venv .venv
```

**2. Activate virtual environment:**
```console
.\.venv\Scripts\activate
```

**3. Create a `.env` file with the required API keys for LLM access:**

`.env` file:
```console
GROQ_API_KEY = "<your-api-key>"
GOOGLE_API_KEY = "<your-api-key>"
OPENAI_API_KEY = "<your-api-key>"
```

**4. Install python dependencies:**
```console
pip install -r requirements.txt
```

**5. Run the application**
```console
streamlit run main.py
```
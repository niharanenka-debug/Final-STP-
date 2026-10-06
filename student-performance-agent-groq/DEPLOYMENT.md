# GitHub and Streamlit Cloud deployment

## Push to GitHub

From the extracted project folder:

```bash
git init
git add .
git commit -m "Initial student performance AI app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

The `.gitignore` excludes `.env`, virtual environments, SQLite databases, Python caches, and uploaded files. Do not commit a real Groq API key.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select **New app**.
4. Choose the repository and branch.
5. Set the main file to:

```text
app.py
```

6. Open **Advanced settings → Secrets**.
7. Add:

```toml
GROQ_API_KEY = "gsk_your_real_key_here"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TPM_LIMIT = "8000"
GROQ_CHUNK_INPUT_TOKENS = "4000"
GROQ_MAX_REQUEST_TOKENS = "6000"
GROQ_MAX_OUTPUT_TOKENS = "700"
GROQ_MAX_RETRIES = "4"
LOW_COST_MODE = "true"
RUN_QUALITY_CHECK = "false"
```

8. Deploy.

The code reads `GROQ_API_KEY` from Streamlit Secrets in the cloud and from `.env` when running locally.

## Local run

```bash
cp .env.example .env
# put GROQ_API_KEY in .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data note

SQLite storage is local to the running Streamlit instance. On hosted environments, local files may not be permanent across restarts. Use an external database later if persistent multi-user history is required.

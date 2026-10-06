# Quick start

```bash
cd student-performance-agent-groq
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env  # set GROQ_API_KEY
streamlit run app.py
```

Use `sample_marks.csv` for the first upload.

The API key belongs in the root `.env` file next to `app.py`:

```env
GROQ_API_KEY=gsk_your_real_key_here
```

Do not put the key in `app.py`, `agents.py`, or `llm_gateway.py`. Do not share the key in chat or commit `.env` to Git.

Run tests with:

```bash
pytest -q
```

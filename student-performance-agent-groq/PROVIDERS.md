# Provider variants and low-token mode

## Which provider should you use?

### Groq

Use Groq first when you want very fast responses and a simple direct API. It is the better default for this student dashboard when latency matters. Groq's free limits are organization/model dependent, so check the limits shown in your Groq console.

### OpenRouter

Use OpenRouter when you want to switch between models without changing application code, use a free model variant, or add routing/fallback options. Free models have request caps; OpenRouter's current pricing page lists 50 requests/day on the free plan, while the exact model and provider limits can vary.

## Low-token design

The application defaults to low-cost mode:

- Python performs parsing, validation, averages, grades, and trends.
- The Performance Agent and Personalized Recommendation Agent use **one combined JSON call** for an individual student.
- The Report Agent uses a second compact call.
- The Quality Agent is optional and disabled by default.
- Output arrays are capped and prompts use compact JSON without indentation.
- The API request includes `max_tokens` so the model cannot generate an unnecessarily long answer.

Therefore, one individual report normally uses **two API calls**. Enabling the optional quality check makes it three.

## Approximate call plan

| Action | Default calls | With quality check |
|---|---:|---:|
| Individual report | 2 | 3 |
| Whole-class report | 2 | 3 |
| One question | 1 | 1 |

## Setup

Each provider project contains a provider-specific `.env.example` file. Copy it to `.env`, put in only your API key, and run Streamlit.

```bash
cp .env.example .env
# edit .env and replace the placeholder key
streamlit run app.py
```

Never commit `.env` or expose student names and marks unnecessarily to a third-party model.

# Deploying Parallax for personal access (phone / other computer)

This guide gets Parallax onto a private, password-protected URL you can open
from any device. We use **Streamlit Community Cloud** (free, deploys straight
from your GitHub repo).

> Security note: a deployed URL is reachable by anyone who has it. The built-in
> **password gate** (see step 3) is what keeps it private. Do not skip it.

## Prerequisites
- Your code is pushed to GitHub (`fakjak33/parallax-quant-lab`, `main` branch). ✅
- A Streamlit Community Cloud account (free): https://share.streamlit.io

## Steps

### 1. Sign in to Streamlit Community Cloud
Go to https://share.streamlit.io and "Continue with GitHub". Authorize it to
read your repositories.

### 2. Create the app
- Click **New app** → **Deploy a public app from a repo**.
- Repository: `fakjak33/parallax-quant-lab`
- Branch: `main`
- Main file path: `app.py`
- Click **Deploy**. First build takes a few minutes (installs `requirements.txt`).

### 3. Set your password (REQUIRED)
- In the app's **Settings → Secrets**, paste:
  ```toml
  app_password = "your-long-random-passphrase"
  ```
- Save. The app restarts; now it prompts for that password before showing anything.

### 4. Use it anywhere
You'll get a URL like `https://parallax-quant-lab-xxxx.streamlit.app`.
Open it on your phone or another computer, enter the password, and you're in.
(Tip: "Add to Home Screen" on your phone makes it feel like a native app.)

## Notes & limits
- **Data:** yfinance is fetched live and cached per session; the free tier sleeps
  when idle and wakes on the next visit (first load after sleeping is slower).
- **Resources:** the free tier has ~1 GB RAM. Parallax caps tickers (12) and
  spectrum/heatmap grid sizes to stay within that. Very long date ranges ×
  many tickers can still be heavy.
- **Updates:** push to `main` and Streamlit auto-redeploys.
- **yfinance ToS:** fine for personal use. Don't redistribute the data publicly.

## Alternatives
- **Local network only:** run `streamlit run app.py` and open
  `http://<your-computer-LAN-IP>:8501` from a phone on the same Wi-Fi. No cloud,
  but only works at home and only while your computer is on.
- **Always-on with more power:** Render / Railway / Fly.io (~$5–7/mo) using the
  same repo; add the same `app_password` secret there.

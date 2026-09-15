# House Price Prediction — Vercel Web App

A web version of the original `linear_regression.py` script: a form where
you enter a house's features and get a predicted price back, powered by
the same scikit-learn Linear Regression model, served as a Vercel
serverless function.

## How it works

- `train_model.py` trains the model on `Housing.csv` (identical logic to
  the original script) and saves `api/model.pkl` + `api/columns.json`.
- `api/predict.py` is a Flask app that Vercel runs as a serverless
  function at `POST /api/predict`. It loads the saved model, builds the
  one-hot-encoded feature row from your form input, and returns a JSON
  prediction.
- `index.html` is a static form that calls `/api/predict` and displays
  the result. Vercel serves it automatically since it's at the project
  root.

## Deploy to Vercel

1. Push this folder to a GitHub repo.
2. Go to [vercel.com](https://vercel.com), sign in with GitHub, click
   **Add New → Project**, and import the repo.
3. Vercel auto-detects `vercel.json` and the `api/` folder — no build
   config needed. Click **Deploy**.
4. You'll get a live URL like `your-project.vercel.app`. Open it, fill
   in the form, and get a prediction.

## Run locally first (recommended)

```bash
pip install -r requirements.txt
python train_model.py          # regenerates api/model.pkl if you change Housing.csv
python api/predict.py          # starts a local Flask server on :5000
```

Then open `index.html` in a browser — note that when running locally
this way, the fetch target in `index.html` (`/api/predict`) expects to
be served from the same origin, so it works out of the box on Vercel
but for local-only testing you may prefer `curl` against
`http://127.0.0.1:5000/api/predict` directly, or use `vercel dev`
(Vercel's CLI) which replicates the deployed routing locally.

## Retraining

If you update `Housing.csv`, rerun `python train_model.py` before
redeploying — it overwrites `api/model.pkl` and `api/columns.json` to
match the new data.

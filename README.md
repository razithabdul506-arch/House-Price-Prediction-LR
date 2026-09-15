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
  root, and `api/predict.py` also serves it at `/` when you run the app
  locally, so the page and the API always share an origin.

## Deploy to Vercel

1. Push this folder to a GitHub repo.
2. Go to [vercel.com](https://vercel.com), sign in with GitHub, click
   **Add New → Project**, and import the repo.
3. Vercel auto-detects `index.html` at the root and `api/predict.py` as a
   Python serverless function — no `vercel.json` or build config needed.
   Click **Deploy**.
4. You'll get a live URL like `your-project.vercel.app`. Open it, fill
   in the form, and get a prediction.

## Run locally first (recommended)

```bash
pip install -r requirements.txt
python train_model.py          # regenerates api/model.pkl if you change Housing.csv
python api/predict.py          # serves the form AND the API on :5000
```

Then open **http://127.0.0.1:5000** in your browser.

Do not open `index.html` by double-clicking it. The page would load over
`file://`, where its `fetch('/api/predict')` resolves to
`file:///api/predict` — there is no server at that address, so the browser
blocks the request and the form reports **"Failed to fetch"**. Loading the
page from `http://127.0.0.1:5000` puts it on the same origin as the API,
which is what the form expects.

To test the API on its own:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"area":7420,"bedrooms":4,"bathrooms":2,"stories":3,"parking":2,
       "mainroad":"yes","guestroom":"no","basement":"no",
       "hotwaterheating":"no","airconditioning":"yes","prefarea":"yes",
       "furnishingstatus":"furnished"}'
```

## Retraining

If you update `Housing.csv`, rerun `python train_model.py` before
redeploying — it overwrites `api/model.pkl` and `api/columns.json` to
match the new data.

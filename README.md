# House Price Prediction — Linear Regression

A Linear Regression model trained on `Housing.csv`, with a web form that
predicts a house price from its features.

**Open `index.html` in a browser. That's it — no server, no install.**

## How it works

`train_model.py` trains the model with scikit-learn and writes three things:

| File | Used by | Purpose |
|---|---|---|
| `model.js` | `index.html` | The fitted coefficients + intercept, as JSON |
| `server/model.pkl` | `server/predict.py` | The pickled model, for the JSON API |
| `server/columns.json` | `server/predict.py` | The one-hot column order the model expects |

A linear regression is just `coefficients · features + intercept`, so the
browser can evaluate it directly from `model.js` — the arithmetic is a dot
product, not something that needs Python at runtime. `index.html` rebuilds
the same one-hot row that `pd.get_dummies(..., drop_first=True)` produced
during training and computes the prediction in JavaScript. Results are
numerically identical to scikit-learn's.

## Run it

Just open `index.html`. Nothing to install.

## Retraining

Only needed if you change `Housing.csv`:

```bash
pip install -r requirements.txt
python train_model.py
```

This overwrites `model.js`, `server/model.pkl` and `server/columns.json` so
the page and the API both reflect the new data.

## Deploy to Vercel

1. Push this repo to GitHub.
2. On [vercel.com](https://vercel.com), **Add New → Project**, import the repo.
3. Framework preset: **Other**. No build command, no output directory.
4. **Deploy**.

It deploys as a static site — `index.html` and `model.js` are served as-is,
with no serverless function, no Python runtime and no build step.

> `server/` is deliberately outside `api/` so Vercel does **not** try to build
> it as a serverless function. Bundling scikit-learn, pandas and numpy comes
> close to Vercel's 250 MB function size limit, and none of it is needed to
> serve the page.

## Optional: the model as a JSON API

`server/predict.py` exposes the same model over HTTP, if you want to call it
from something other than the browser.

```bash
pip install -r requirements.txt
python server/predict.py     # http://127.0.0.1:5000
```

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"area":7420,"bedrooms":4,"bathrooms":2,"stories":3,"parking":2,
       "mainroad":"yes","guestroom":"no","basement":"no",
       "hotwaterheating":"yes","airconditioning":"yes","prefarea":"yes",
       "furnishingstatus":"semi-furnished"}'
# {"predicted_price": 8526044.19}
```

## Model performance

On a 20% held-out test split (`random_state=42`):

| Metric | Value |
|---|---|
| R² | 0.653 |
| MAE | ₹970,043 |
| RMSE | ₹1,324,506 |

R² of 0.65 means the model explains about 65% of the variance in price —
reasonable for a plain linear fit on 13 features, and the honest number to
quote rather than rounding up.

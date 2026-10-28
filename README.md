# ShopSense — Hybrid Product Recommendation System

A Streamlit app that combines content-based recommendations (TF-IDF over product descriptions) with item-item collaborative filtering (cosine similarity over customer purchase histories).

## Dataset
- **UCI Online Retail**: https://archive.ics.uci.edu/dataset/352/online+retail
- Daqing Chen (2015), DOI: `10.24432/C5BW33`
- License: CC BY 4.0; retain attribution when redistributing.
- The app downloads the official UCI archive on first run.

## Run locally
Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Open https://share.streamlit.io and sign in.
3. Choose **Create app**, select your repository and branch, and set the main file to `app.py`.
4. Deploy. The first launch downloads and processes the dataset, so allow it some time.
5. If the app runs out of memory, reduce `MAX_PRODUCTS` and `MAX_CUSTOMERS` near the top of `app.py`.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## How the hybrid model works
1. **Content-based**: TF-IDF vectorizes product descriptions, then cosine similarity measures textual similarity.
2. **Collaborative**: a sparse customer-product matrix records whether each customer purchased each product. Product vectors are compared using cosine similarity, recommending products bought by overlapping customers.
3. **Hybrid**: each similarity vector is normalized and combined with a tunable content weight. The sidebar slider controls the trade-off.

## Evaluation
The Evaluation tab offers a lightweight hold-one-item-out diagnostic on up to 300 customers. It reports Precision@10, Recall@10, and Hit Rate@10. Since one item is held out per customer, Recall@10 and Hit Rate@10 are equivalent. This random holdout is only a basic diagnostic; for stronger evidence, use a chronological train/test split, tune on a validation set, and compare with a popularity baseline.

## Limitations
- Historical purchase data is not explicit ratings or direct evidence of satisfaction.
- New products and new customers have limited collaborative signals (cold-start problem).
- Product descriptions may be duplicated or noisy.
- This is an educational prototype, not a production commerce recommender.

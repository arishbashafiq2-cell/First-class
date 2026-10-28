import io
import zipfile
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="ShopSense | Product Recommender", page_icon="🛍️", layout="wide")

DATASET_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
MAX_PRODUCTS = 2500
MAX_CUSTOMERS = 12000

@st.cache_data(show_spinner=False)
def load_retail_data():
    """Download the official UCI Online Retail ZIP and read its Excel file."""
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        excel_name = next((n for n in archive.namelist() if n.lower().endswith((".xlsx", ".xls"))), None)
        if excel_name is None:
            raise ValueError("The UCI archive did not contain an Excel file.")
        with archive.open(excel_name) as file:
            df = pd.read_excel(file, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {', '.join(sorted(missing))}")
    return df

@st.cache_resource(show_spinner=False)
def build_model(raw_df, max_products=MAX_PRODUCTS, max_customers=MAX_CUSTOMERS):
    """Build content and implicit-feedback item-item collaborative models."""
    df = raw_df.copy()
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["CustomerID"] = df["CustomerID"].astype("string")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Remove cancellations/returns and invalid records for positive-purchase recommendations.
    df = df.dropna(subset=["CustomerID", "InvoiceNo", "StockCode", "InvoiceDate"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0) & (df["Description"] != "")]
    df = df[~df["InvoiceNo"].astype(str).str.upper().str.startswith("C")]
    if df.empty:
        raise ValueError("No valid positive-purchase records remain after cleaning.")

    # Keep frequent products and active customers to keep the sparse model manageable.
    top_codes = df["StockCode"].value_counts().head(max_products).index
    df = df[df["StockCode"].isin(top_codes)]
    top_customers = df["CustomerID"].value_counts().head(max_customers).index
    df = df[df["CustomerID"].isin(top_customers)]

    products = (
        df.groupby("StockCode", as_index=False)
          .agg(Description=("Description", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
               TotalQuantity=("Quantity", "sum"),
               Transactions=("InvoiceNo", "nunique"),
               Revenue=("UnitPrice", lambda s: 0.0))
    )
    # Calculate revenue separately to avoid ambiguity when the same item has varying quantities.
    revenue = (df.assign(LineRevenue=df["Quantity"] * df["UnitPrice"])
                 .groupby("StockCode")["LineRevenue"].sum())
    products["Revenue"] = products["StockCode"].map(revenue).fillna(0.0)
    products["Popularity"] = products["StockCode"].map(df["StockCode"].value_counts()).fillna(0).astype(int)
    products = products.sort_values("Popularity", ascending=False).reset_index(drop=True)

    # One stable product row per StockCode.
    code_to_idx = {code: i for i, code in enumerate(products["StockCode"])}
    customer_codes = pd.Index(df["CustomerID"].astype(str).unique())
    customer_to_idx = {customer: i for i, customer in enumerate(customer_codes)}
    row = df["CustomerID"].astype(str).map(customer_to_idx).to_numpy()
    col = df["StockCode"].map(code_to_idx).to_numpy()
    vals = np.ones(len(df), dtype=np.float32)
    user_item = csr_matrix((vals, (row, col)), shape=(len(customer_codes), len(products)), dtype=np.float32)
    user_item.data[:] = 1.0  # implicit preference: purchased at least once

    # Content-based similarity from product descriptions.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=12000)
    text_matrix = vectorizer.fit_transform(products["Description"].astype(str))
    content_sim = cosine_similarity(text_matrix, dense_output=True).astype(np.float32)

    # Item-item collaborative similarity: products purchased by overlapping customers.
    item_customer = user_item.T.tocsr()
    item_sim = cosine_similarity(item_customer, dense_output=True).astype(np.float32)
    np.fill_diagonal(item_sim, 0.0)

    return {
        "df": df, "products": products, "code_to_idx": code_to_idx,
        "customer_codes": customer_codes.astype(str).tolist(),
        "customer_to_idx": customer_to_idx, "user_item": user_item,
        "content_sim": content_sim, "collab_sim": item_sim,
    }

def recommend_similar(model, product_code, n=10, content_weight=0.5):
    products = model["products"]
    idx = model["code_to_idx"][product_code]
    content = model["content_sim"][idx].copy()
    collab = model["collab_sim"][idx].copy()

    # Normalize each component independently so one score type does not dominate by scale.
    def normalize(scores):
        scores = np.asarray(scores, dtype=float)
        maximum = scores.max() if scores.size else 0.0
        return scores / maximum if maximum > 0 else scores

    hybrid = content_weight * normalize(content) + (1 - content_weight) * normalize(collab)
    hybrid[idx] = -1
    order = np.argsort(hybrid)[::-1]
    order = [i for i in order if hybrid[i] >= 0][:n]
    out = products.iloc[order][["StockCode", "Description", "Popularity", "TotalQuantity", "Revenue"]].copy()
    out["Content score"] = content[order]
    out["Collaborative score"] = collab[order]
    out["Hybrid score"] = hybrid[order]
    return out.reset_index(drop=True)

def customer_recommendations(model, customer_id, n=10, content_weight=0.5):
    """Blend a customer's purchase history with content and co-purchase similarities."""
    cust_idx = model["customer_to_idx"].get(str(customer_id))
    if cust_idx is None:
        return pd.DataFrame()
    purchased = model["user_item"].getrow(cust_idx).indices
    if len(purchased) == 0:
        return pd.DataFrame()
    content_scores = model["content_sim"][purchased].mean(axis=0)
    collab_scores = model["collab_sim"][purchased].mean(axis=0)
    content_scores = np.asarray(content_scores).ravel()
    collab_scores = np.asarray(collab_scores).ravel()

    def normalize(scores):
        maximum = float(scores.max()) if scores.size else 0.0
        return scores / maximum if maximum > 0 else scores

    hybrid = content_weight * normalize(content_scores) + (1 - content_weight) * normalize(collab_scores)
    hybrid[purchased] = -1
    order = np.argsort(hybrid)[::-1]
    order = [i for i in order if hybrid[i] >= 0][:n]
    products = model["products"]
    out = products.iloc[order][["StockCode", "Description", "Popularity", "TotalQuantity", "Revenue"]].copy()
    out["Content score"] = content_scores[order]
    out["Collaborative score"] = collab_scores[order]
    out["Hybrid score"] = hybrid[order]
    return out.reset_index(drop=True)

def evaluate_holdout(model, max_customers=300, k=10):
    """Small offline temporal-style proxy: hide one item from multi-item customer histories.
    Metrics are a diagnostic proxy, not a production-grade unbiased benchmark.
    """
    ui = model["user_item"]
    products = model["products"]
    content_sim = model["content_sim"]
    collab_sim = model["collab_sim"]
    rng = np.random.default_rng(42)
    eligible = np.where(np.diff(ui.indptr) >= 2)[0]
    if len(eligible) == 0:
        return {"Evaluated customers": 0, "Precision@10": 0.0, "Recall@10": 0.0, "Hit Rate@10": 0.0}
    if len(eligible) > max_customers:
        eligible = rng.choice(eligible, size=max_customers, replace=False)
    hits, precisions, recalls = [], [], []
    for user_idx in eligible:
        items = ui.getrow(int(user_idx)).indices
        target = int(rng.choice(items))
        history = items[items != target]
        if len(history) == 0:
            continue
        c = np.asarray(content_sim[history].mean(axis=0)).ravel()
        co = np.asarray(collab_sim[history].mean(axis=0)).ravel()
        cmax, comax = c.max(), co.max()
        if cmax > 0: c = c / cmax
        if comax > 0: co = co / comax
        scores = 0.5 * c + 0.5 * co
        scores[history] = -1
        ranked = np.argsort(scores)[::-1][:k]
        hit = int(target in ranked)
        hits.append(hit)
        precisions.append(hit / k)
        recalls.append(float(hit))
    return {
        f"Evaluated customers": len(hits),
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"Hit Rate@{k}": float(np.mean(hits)) if hits else 0.0,
    }

st.title("🛍️ ShopSense")
st.subheader("Hybrid Product Recommendation System")
st.caption("Real purchase data • Content-based filtering + collaborative filtering • Streamlit")

with st.sidebar:
    st.header("Model controls")
    content_weight = st.slider("Content-based weight", 0.0, 1.0, 0.5, 0.05,
                               help="0 = collaborative only; 1 = content only; 0.5 = balanced hybrid.")
    top_n = st.slider("Recommendations to show", 5, 20, 10)
    st.markdown("---")
    st.caption("Dataset: UCI Online Retail (CC BY 4.0). Product purchase history is used as implicit feedback.")

try:
    with st.spinner("Loading the UCI retail dataset..."):
        raw = load_retail_data()
    with st.spinner("Preparing the hybrid recommendation model..."):
        model = build_model(raw)
except Exception as exc:
    st.error("The dataset or model could not be loaded.")
    st.exception(exc)
    st.info("Check your internet connection and dependencies, then reload the app. The app downloads the dataset from the UCI repository on first run.")
    st.stop()

df = model["df"]
products = model["products"]
customers = model["customer_codes"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Valid transaction lines", f"{len(df):,}")
m2.metric("Products modeled", f"{len(products):,}")
m3.metric("Customers modeled", f"{len(customers):,}")
m4.metric("Countries", f"{df['Country'].nunique():,}")

tab_product, tab_customer, tab_eval, tab_data = st.tabs([
    "🔎 Similar products", "👤 For a customer", "📊 Evaluation", "🧾 Dataset"
])

with tab_product:
    st.markdown("### Find products similar to an item")
    options = products.apply(lambda r: f"{r['Description']}  ·  SKU {r['StockCode']}", axis=1).tolist()
    selected_label = st.selectbox("Choose a product", options, index=0, key="product_select")
    selected_pos = options.index(selected_label)
    selected_code = str(products.iloc[selected_pos]["StockCode"])
    weight = content_weight
    current = products.iloc[selected_pos]
    c1, c2, c3 = st.columns(3)
    c1.metric("Product SKU", selected_code)
    c2.metric("Times purchased", f"{int(current['Popularity']):,}")
    c3.metric("Total quantity", f"{int(current['TotalQuantity']):,}")
    st.caption(f"Hybrid blend: {weight:.0%} content similarity + {1-weight:.0%} co-purchase similarity.")
    recs = recommend_similar(model, selected_code, top_n, weight)
    st.dataframe(recs, use_container_width=True, hide_index=True,
                 column_config={
                     "Content score": st.column_config.ProgressColumn("Content score", min_value=0, max_value=1, format="%.2f"),
                     "Collaborative score": st.column_config.ProgressColumn("Collaborative score", min_value=0, max_value=1, format="%.2f"),
                     "Hybrid score": st.column_config.ProgressColumn("Hybrid score", min_value=0, max_value=1, format="%.2f"),
                     "Revenue": st.column_config.NumberColumn("Revenue (£)", format="£%.2f"),
                 })
    st.download_button("Download recommendations CSV", recs.to_csv(index=False).encode("utf-8"),
                       file_name="product_recommendations.csv", mime="text/csv")

with tab_customer:
    st.markdown("### Personalized recommendations from purchase history")
    st.write("Choose a customer ID from the dataset. IDs are anonymized dataset identifiers, not names.")
    customer_id = st.selectbox("Customer ID", customers, index=0)
    purchased_idx = model["customer_to_idx"][str(customer_id)]
    bought = model["user_item"].getrow(purchased_idx).indices
    if len(bought):
        bought_df = products.iloc[bought][["StockCode", "Description", "Popularity"]]
        st.write(f"Known distinct products purchased: **{len(bought)}**")
        with st.expander("View this customer's known products"):
            st.dataframe(bought_df, use_container_width=True, hide_index=True)
        cust_recs = customer_recommendations(model, customer_id, top_n, content_weight)
        if cust_recs.empty:
            st.warning("No recommendations could be generated for this customer.")
        else:
            st.dataframe(cust_recs, use_container_width=True, hide_index=True,
                         column_config={
                             "Content score": st.column_config.ProgressColumn("Content score", min_value=0, max_value=1, format="%.2f"),
                             "Collaborative score": st.column_config.ProgressColumn("Collaborative score", min_value=0, max_value=1, format="%.2f"),
                             "Hybrid score": st.column_config.ProgressColumn("Hybrid score", min_value=0, max_value=1, format="%.2f"),
                             "Revenue": st.column_config.NumberColumn("Revenue (£)", format="£%.2f"),
                         })
            st.download_button("Download customer recommendations CSV", cust_recs.to_csv(index=False).encode("utf-8"),
                               file_name=f"customer_{customer_id}_recommendations.csv", mime="text/csv")
    else:
        st.info("This customer has no eligible purchase history in the filtered model.")

with tab_eval:
    st.markdown("### Offline evaluation")
    st.write("The app hides one previously purchased item per sampled customer and checks whether the model ranks it in the top 10. This is a lightweight diagnostic, not a guarantee of future business performance.")
    if st.button("Run evaluation (up to 300 customers)", type="primary"):
        with st.spinner("Evaluating recommendations..."):
            metrics = evaluate_holdout(model, max_customers=300, k=10)
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.4f}" if isinstance(value, float) else f"{value:,}")
        st.caption("Precision@10 is relevant held-out items divided by 10; Recall@10 and Hit Rate@10 are equivalent here because each customer has one held-out target. Compare variants on the same split before drawing conclusions.")
    st.markdown("**Recommended experiments**")
    st.markdown("- Compare content weights 0.0, 0.25, 0.5, 0.75, and 1.0.\n- Compare against a popularity-only baseline.\n- Use a chronological train/test split for a more rigorous production evaluation.\n- Track coverage, diversity, click-through rate, conversion, and revenue after deployment.")

with tab_data:
    st.markdown("### Dataset overview")
    st.write("Official dataset: UCI Online Retail — transactions from a UK-based online retailer from December 2010 through December 2011.")
    st.dataframe(products.head(30), use_container_width=True, hide_index=True)
    st.markdown("**Data cleaning performed**")
    st.markdown("- Removes rows without customer ID, invoice number, product code, date, or description.\n- Excludes cancellations (invoice IDs beginning with `C`), returns/negative quantities, and non-positive prices.\n- Restricts the model to frequent products and active customers for manageable memory use.\n- Treats each customer-product purchase as implicit feedback; it does not interpret purchase quantity as a product rating.")
    st.markdown("**Source and attribution:** [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail), Daqing Chen (2015), DOI: 10.24432/C5BW33, CC BY 4.0.")
    st.download_button("Download cleaned modeled data CSV",
                       products.to_csv(index=False).encode("utf-8"),
                       file_name="modeled_products.csv", mime="text/csv")

st.markdown("---")
st.caption("Educational prototype. For a production store, add authentication, privacy controls, model monitoring, chronological evaluation, and a refreshed purchase-event pipeline.")

"""
========================================
SISTEMA DE RECOMENDACIÓN HÍBRIDO
========================================
Combinación de:
  1. Filtrado Colaborativo (Implicit ALS)
  2. Modelo Basado en Contenido (Cosine Similarity)

Requisitos Técnicos Implementados:
  ✓ Filtrado colaborativo    → ALS (256 factores latentes)
  ✓ Matrix factorization     → Implicit ALS (Hu et al. 2008), confianza 1+40×rating
  ✓ Modelo basado en contenido → Cosine similarity (18 géneros + año + TF-IDF 200 dims)
  ✓ Evaluación               → Precision@K, Recall@K, NDCG@K

Protocolo de Evaluación:
  - 99-negative sampling (He et al. NCF 2017 estándar)
  - threshold = 3.5
  - 600 usuarios de test, random_state=42
  - K values: 5, 10, 20
========================================
"""
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Movie card ── */
.movie-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    min-height: 180px;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: var(--card-accent, #6366f1);
    border-radius: 12px 12px 0 0;
}
.card-rank {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.card-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.3;
    margin-bottom: 10px;
}
.card-genres {
    margin-bottom: 10px;
    line-height: 1.9;
}
.genre-tag {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.70rem;
    font-weight: 600;
    color: #ffffff !important;
    margin: 2px 2px 2px 0;
    letter-spacing: 0.02em;
}
.score-row {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    flex-wrap: wrap;
}
.score-pill {
    background: #f1f5f9;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #334155 !important;
}
.score-pill span {
    color: #6366f1;
}
/* ── Match bar ── */
.match-bar-wrap {
    background: #e2e8f0;
    border-radius: 8px;
    height: 7px;
    margin: 6px 0 4px;
    overflow: hidden;
}
.match-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
}
.match-label {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 600;
}
/* ── Seed pill ── */
.seed-pill {
    display: inline-block;
    background: #ede9fe;
    color: #5b21b6 !important;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    movies_df   = joblib.load(os.path.join(MODEL_DIR, "movies.joblib"))
    content_sim = joblib.load(os.path.join(MODEL_DIR, "content_sim.joblib"))

    als_model    = joblib.load(os.path.join(MODEL_DIR, "als_model.joblib"))
    als_user_enc = joblib.load(os.path.join(MODEL_DIR, "als_user_enc.joblib"))
    als_item_enc = joblib.load(os.path.join(MODEL_DIR, "als_item_enc.joblib"))

    # Load best alpha from optimisation run (default 0.9 if not found)
    alpha_path = os.path.join(MODEL_DIR, "best_alpha.joblib")
    best_alpha = float(joblib.load(alpha_path)) if os.path.exists(alpha_path) else 0.9

    eval_path = os.path.join(MODEL_DIR, "eval_df.joblib")
    eval_df   = joblib.load(eval_path) if os.path.exists(eval_path) else None

    # movieId → DataFrame index lookup
    movieid_to_idx = dict(zip(movies_df["movieId"], movies_df.index))
    all_movie_ids  = movies_df["movieId"].tolist()
    mid2idx        = {m: i for i, m in enumerate(all_movie_ids)}

    # Build vectorised item factor matrix aligned with all_movie_ids
    F_als = als_model.item_factors.shape[1]
    als_item_mat = np.zeros((len(all_movie_ids), F_als), dtype=np.float32)
    for i, mid in enumerate(all_movie_ids):
        if mid in als_item_enc:
            j = als_item_enc[mid]
            if j < als_model.item_factors.shape[0]:
                als_item_mat[i] = als_model.item_factors[j]

    return {
        "movies_df":      movies_df,
        "content_sim":    content_sim,
        "als_model":      als_model,
        "als_item_enc":   als_item_enc,
        "als_item_mat":   als_item_mat,
        "best_alpha":     best_alpha,
        "eval_df":        eval_df,
        "movieid_to_idx": movieid_to_idx,
        "all_movie_ids":  all_movie_ids,
        "mid2idx":        mid2idx,
    }


def models_exist():
    required = [
        "als_model.joblib", "als_user_enc.joblib", "als_item_enc.joblib",
        "content_sim.joblib", "movies.joblib",
    ]
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required)


# ─── Scoring helpers ──────────────────────────────────────────────────────────
def get_collab_scores(seed_ids, pool_mids, pool_idxs, M):
    """
    Item-space CF: average ALS item factors of seed movies → pseudo-user vector.
    Works without a registered user_id (cold-start via item analogy).
    """
    seed_vecs = []
    for mid in seed_ids:
        if mid in M["als_item_enc"]:
            j = M["als_item_enc"][mid]
            if j < M["als_model"].item_factors.shape[0]:
                seed_vecs.append(M["als_model"].item_factors[j])

    if not seed_vecs:
        return np.zeros(len(pool_mids), dtype=np.float32)

    pseudo_user = np.mean(seed_vecs, axis=0).astype(np.float32)
    scores = M["als_item_mat"][pool_idxs] @ pseudo_user
    return scores


def get_content_scores(seed_ids, pool_mids, pool_idxs, M):
    seed_idx = [M["movieid_to_idx"][m] for m in seed_ids if m in M["movieid_to_idx"]]
    if not seed_idx:
        return np.zeros(len(pool_mids), dtype=np.float32)
    scores = M["content_sim"][np.ix_(pool_idxs, seed_idx)].mean(axis=1)
    return np.array(scores, dtype=np.float32)


def normalise(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


def hybrid_recommend(seed_ids, n, alpha, M):
    """
    Return top-n recommendation DataFrame using hybrid ALS + content scoring.

    hybrid_score = alpha * als_score + (1-alpha) * content_score
      alpha=1.0 → pure collaborative filtering (ALS)
      alpha=0.0 → pure content-based
    Both signals are normalised to [0,1] before blending.
    """
    exclude = set(seed_ids)
    pool_mids = [m for m in M["all_movie_ids"] if m not in exclude]
    pool_idxs = [M["mid2idx"][m] for m in pool_mids]

    c_arr  = normalise(get_collab_scores(seed_ids, pool_mids, pool_idxs, M))
    ct_arr = normalise(get_content_scores(seed_ids, pool_mids, pool_idxs, M))

    hybrid    = alpha * c_arr + (1 - alpha) * ct_arr
    top_n_idx = np.argsort(hybrid)[::-1][:n]

    movies_df = M["movies_df"]
    results = []
    for rank, idx in enumerate(top_n_idx, 1):
        mid = pool_mids[idx]
        row = movies_df[movies_df["movieId"] == mid].iloc[0]
        results.append({
            "rank":          rank,
            "movieId":       mid,
            "title":         row["title"],
            "genres":        row["genres"],
            "hybrid_score":  round(float(hybrid[idx]),  4),
            "als_score":     round(float(c_arr[idx]),   4),
            "content_score": round(float(ct_arr[idx]),  4),
        })
    return pd.DataFrame(results)


# ─── Genre tag colours ────────────────────────────────────────────────────────
GENRE_COLORS = {
    "Action":      "#e74c3c",
    "Adventure":   "#e67e22",
    "Animation":   "#f39c12",
    "Children's":  "#2ecc71",
    "Comedy":      "#1abc9c",
    "Crime":       "#9b59b6",
    "Documentary": "#3498db",
    "Drama":       "#2980b9",
    "Fantasy":     "#8e44ad",
    "Horror":      "#c0392b",
    "Musical":     "#d35400",
    "Mystery":     "#7f8c8d",
    "Romance":     "#e91e8c",
    "Sci-Fi":      "#00bcd4",
    "Thriller":    "#795548",
    "War":         "#607d8b",
    "Western":     "#a0522d",
    "Film-Noir":   "#34495e",
}


def genre_tags_html(genres_str):
    tags = []
    for g in genres_str.split("|"):
        g = g.strip()
        if g in ("(no genres listed)", ""):
            continue
        color = GENRE_COLORS.get(g, "#95a5a6")
        tags.append(
            f'<span class="genre-tag" style="background:{color}">{g}</span>'
        )
    return "".join(tags)


def accent_color(genres_str):
    for g in genres_str.split("|"):
        g = g.strip()
        if g and g != "(no genres listed)":
            return GENRE_COLORS.get(g, "#6366f1")
    return "#6366f1"


def extract_year(title):
    m = re.search(r"\((\d{4})\)$", title.strip())
    return m.group(1) if m else ""


def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


# ─── Guard: models must exist ─────────────────────────────────────────────────
if not models_exist():
    st.error(
        "**Models not found.**\n\n"
        "Run the notebook end-to-end first to train the models, "
        "then restart this app."
    )
    st.stop()

M = load_models()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")

    default_alpha = M["best_alpha"]

    alpha = st.slider(
        "Collaborative weight (alpha)",
        min_value=0.0, max_value=1.0,
        value=default_alpha, step=0.05,
        help="0 = pure content · 1 = pure collaborative (ALS)"
    )
    n_recs = st.number_input(
        "Number of recommendations",
        min_value=1, max_value=20, value=10, step=1
    )

    if alpha == 1.0:
        mode_label = "Pure Collaborative Filtering"
    elif alpha == 0.0:
        mode_label = "Pure Content-Based"
    elif alpha >= 0.7:
        mode_label = "Hybrid (CF-dominant)"
    else:
        mode_label = "Balanced Hybrid"

    st.info(
        f"**{mode_label}  (α={alpha:.2f})**\n\n"
        "- **α=1.0** → pure ALS collaborative filtering\n"
        "- **α=0.0** → pure content-based (genres, year, TF-IDF)\n"
        f"- **α={default_alpha}** → tuned optimum (NDCG@10 maximised)\n\n"
        "CF scoring uses item-space analogy: seed movie factors "
        "are averaged into a pseudo-user vector."
    )
    st.markdown("---")
    st.caption("MovieLens 1M · ALS (256 factors) + TF-IDF content · Streamlit")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("Hybrid Movie Recommendation System")
st.markdown(
    "Combines **Collaborative Filtering** (Implicit ALS, 256 latent factors) "
    "and **Content-Based Filtering** (18 genres + year + TF-IDF title features) "
    "trained on the **MovieLens 1M** dataset (1M ratings, 6K users, 3.9K movies)."
)

tab1, tab2 = st.tabs(["Get Recommendations", "Model Evaluation"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Get Recommendations
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<p style="color:#374151;font-size:1rem;margin-bottom:4px">'
        'Selecciona una o más películas que te gusten y el sistema encontrará '
        'recomendaciones similares combinando filtrado colaborativo (ALS) y por contenido.</p>',
        unsafe_allow_html=True,
    )

    movies_df   = M["movies_df"]
    all_titles  = sorted(movies_df["title"].dropna().unique().tolist())
    title_to_id = dict(zip(movies_df["title"], movies_df["movieId"]))

    selected_titles = st.multiselect(
        "Busca y selecciona películas:",
        options=all_titles,
        default=[],
        placeholder="Escribe un título…",
        help="Selecciona una o más películas como semillas para las recomendaciones.",
    )

    recommend_btn = st.button("Recomendar", type="primary", use_container_width=False)

    if recommend_btn:
        if not selected_titles:
            st.warning("Please select at least one movie.")
        else:
            seed_ids = [title_to_id[t] for t in selected_titles if t in title_to_id]

            with st.spinner("Computing recommendations…"):
                recs = hybrid_recommend(
                    seed_ids=seed_ids,
                    n=int(n_recs),
                    alpha=alpha,
                    M=M,
                )

            if recs.empty:
                st.warning("No recommendations found — try adding more seed movies.")
            else:
                pills = "".join(
                    f'<span class="seed-pill">{t}</span>'
                    for t in selected_titles
                )
                st.markdown(
                    f'<p style="color:#374151;font-size:0.9rem;margin-bottom:6px">'
                    f'<strong>Basado en:</strong> {pills}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p style="color:#6b7280;font-size:0.85rem;margin-bottom:16px">'
                    f'Mostrando {len(recs)} recomendaciones · Modelo: '
                    f'<strong>Hybrid ALS</strong> · α = {alpha}</p>',
                    unsafe_allow_html=True,
                )

                cols_per_row = 3
                rows = [recs.iloc[i:i+cols_per_row] for i in range(0, len(recs), cols_per_row)]

                for row_df in rows:
                    cols = st.columns(cols_per_row)
                    for col, (_, rec) in zip(cols, row_df.iterrows()):
                        title_clean  = clean_title(rec["title"])
                        year         = extract_year(rec["title"])
                        score_pct    = int(rec["hybrid_score"]  * 100)
                        als_pct      = int(rec["als_score"]     * 100)
                        content_pct  = int(rec["content_score"] * 100)
                        bar_w        = max(2, score_pct)
                        accent       = accent_color(rec["genres"])
                        rank_year    = f"#{rec['rank']}" + (f" · {year}" if year else "")

                        with col:
                            st.markdown(
                                f"""
                                <div class="movie-card" style="--card-accent:{accent}">
                                  <div class="card-rank">{rank_year}</div>
                                  <div class="card-title">{title_clean}</div>
                                  <div class="card-genres">{genre_tags_html(rec['genres'])}</div>
                                  <div class="match-label">Match {score_pct}%</div>
                                  <div class="match-bar-wrap">
                                    <div class="match-bar-fill" style="width:{bar_w}%;background:linear-gradient(90deg,{accent},{accent}aa)"></div>
                                  </div>
                                  <div class="score-row">
                                    <div class="score-pill">ALS <span>{als_pct}%</span></div>
                                    <div class="score-pill">Contenido <span>{content_pct}%</span></div>
                                  </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Offline Evaluation Results")
    st.markdown(
        "**Protocol:** 99-negative sampling (He et al. NCF 2017 standard). "
        "For each test user: held-out positives + 99 random unseen negatives are ranked. "
        "600 users, chronological 80/20 split, relevant = rating ≥ 3.5.\n\n"
        "_This protocol is the standard in CF literature; it is more informative than full-ranking._"
    )

    # Evaluation results — 600 test users, 99 negatives, threshold=3.5, seed=42
    comparison_data = {
        "ALS v3 (α=1.0)": {
            "P@5": 57.19, "P@10": 48.96, "P@20": 38.77,
            "R@5": 26.19, "R@10": 40.50, "R@20": 56.49,
            "NDCG@5": 61.27, "NDCG@10": 60.02, "NDCG@20": 61.54,
        },
        "Hybrid ALS (α=0.9)": {
            "P@5": 56.99, "P@10": 49.08, "P@20": 39.07,
            "R@5": 26.31, "R@10": 40.70, "R@20": 57.10,
            "NDCG@5": 61.26, "NDCG@10": 60.22, "NDCG@20": 61.99,
        },
    }

    cmp_df = pd.DataFrame(comparison_data).T
    cmp_df.index.name = "Model"

    st.markdown("#### ALS vs Hybrid ALS — Precision / Recall / NDCG")
    styled = (
        cmp_df.style
        .format("{:.2f}%")
        .background_gradient(cmap="YlGn", axis=0)
    )
    st.dataframe(styled, width='stretch')

    # Headline metrics
    st.markdown("#### Headline Metrics (Hybrid ALS, α=0.9)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision@10", "49.08%", "+0.12pp vs ALS pure")
    c2.metric("Recall@10",    "40.70%", "+0.20pp vs ALS pure")
    c3.metric("NDCG@10",      "60.22%", "+0.20pp vs ALS pure")

    # Bar chart: NDCG@10 comparison
    st.markdown("#### NDCG@10 by Model")
    chart_df = pd.DataFrame({
        "NDCG@10 (%)": [
            comparison_data["ALS v3 (α=1.0)"]["NDCG@10"],
            comparison_data["Hybrid ALS (α=0.9)"]["NDCG@10"],
        ]
    }, index=["ALS v3 (pure CF)", "Hybrid ALS (α=0.9)"])
    st.bar_chart(chart_df, width='stretch')

    # Metric definitions
    st.markdown("#### Metric Definitions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "**Precision@K**\n\n"
            "Fraction of the top-K recommended items that are relevant.\n\n"
            "> Of the K shown, how many were actually good?"
        )
    with col2:
        st.info(
            "**Recall@K**\n\n"
            "Fraction of all relevant items that appear in the top-K list.\n\n"
            "> Of all good items, how many did we find?"
        )
    with col3:
        st.info(
            "**NDCG@K**\n\n"
            "Normalised Discounted Cumulative Gain — rewards relevant items "
            "ranked higher in the list.\n\n"
            "> Are the best items near the top?"
        )

    # Model architecture
    st.markdown("#### Model Architecture")
    arch_data = {
        "Component": ["ALS v3", "Content-Based", "Hybrid"],
        "Type": [
            "Matrix Factorization (implicit)",
            "Cosine Similarity",
            "Weighted blend",
        ],
        "Details": [
            "256 factors, 100 iterations, reg=0.05, confidence=1+40×rating",
            "221 features: 18 genres + year(×2) + avg_rating + log_pop + TF-IDF(200)",
            "alpha × ALS_score + (1-alpha) × content_score, both normalised [0,1]",
        ],
        "NDCG@10": ["60.02%", "—", "60.22%"],
    }
    st.table(pd.DataFrame(arch_data).set_index("Component"))

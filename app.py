"""
app.py
------
ToxiShield AI – Streamlit Dashboard
Run with:  streamlit run app.py
"""

import os
import sys
import io

import pandas as pd
import streamlit as st

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="ToxiShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS – dark professional theme ─────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #C9D1D9;
}
[data-testid="stAppViewContainer"] {
    background: #0D1117;
}
[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #21262D;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 0.85rem; }
[data-testid="stMetricValue"] { color: #4ECDC4 !important; font-size: 1.6rem; font-weight: 700; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4ECDC4, #2A9D8F);
    color: #0D1117;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 0.55rem 1.4rem;
    font-size: 0.95rem;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #21262D; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #161B22; border-radius: 8px 8px 0 0;
    color: #8B949E; font-weight: 600; padding: 8px 18px;
}
.stTabs [aria-selected="true"] { background: #21262D; color: #4ECDC4 !important; }

/* ── Info / success / warning boxes ── */
.info-card {
    background: #161B22; border: 1px solid #21262D;
    border-left: 4px solid #4ECDC4; border-radius: 8px;
    padding: 16px 20px; margin: 8px 0;
}
.risk-critical { border-left-color: #FF4C4C !important; }
.risk-high     { border-left-color: #FF8C42 !important; }
.risk-medium   { border-left-color: #FFD166 !important; }
.risk-low      { border-left-color: #06D6A0 !important; }

/* ── Text area / inputs ── */
.stTextArea textarea, .stTextInput input {
    background: #161B22 !important;
    color: #C9D1D9 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    font-size: 0.95rem;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: #161B22 !important;
    border: 1px solid #30363D !important;
}

/* ── DataFrames ── */
.stDataFrame { border: 1px solid #21262D; border-radius: 8px; overflow: hidden; }

/* ── Dividers ── */
hr { border-color: #21262D; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (keep startup fast) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_predict():
    from predict import predict_single, predict_batch, CLASS_NAMES, RISK_LEVELS
    return predict_single, predict_batch, CLASS_NAMES, RISK_LEVELS


@st.cache_resource(show_spinner=False)
def _load_meta():
    from evaluation import load_meta
    return load_meta()


def _eval_charts():
    from evaluation import (
        plot_model_comparison,
        plot_confusion_matrix,
        plot_prediction_probabilities,
        metrics_dataframe,
    )
    return plot_model_comparison, plot_confusion_matrix, plot_prediction_probabilities, metrics_dataframe


def _utils():
    import utils as u
    return u


# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = [
    "🏠 Home",
    "📊 Dataset Analytics",
    "🧹 Text Preprocessing Demo",
    "🤖 Toxic Comment Prediction",
    "📈 Model Performance",
    "☁ WordCloud",
    "📁 Batch CSV Prediction",
    "⬇ Download Results",
    "ℹ About",
]

with st.sidebar:
    st.markdown("## 🛡️ ToxiShield AI")
    st.markdown("*Intelligent Content Moderation*")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<small style='color:#8B949E'>Model: TF-IDF + Logistic Regression<br>"
        "Dataset: Jigsaw Toxic Comments<br>"
        "Version: 1.0.0</small>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(
        "<h1 style='color:#4ECDC4;font-size:2.6rem;font-weight:800;'>🛡️ ToxiShield AI</h1>"
        "<p style='color:#8B949E;font-size:1.15rem;margin-top:-10px;'>"
        "Intelligent Toxic Comment Detection &amp; Content Moderation System</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Categories Detected", "7")
    c2.metric("ML Models Trained", "2")
    c3.metric("Vectoriser", "TF-IDF")
    c4.metric("Framework", "Scikit-learn")

    st.markdown("### 🚀 What ToxiShield AI Does")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div class='info-card'>
<b>🔍 Real-Time Detection</b><br>
Classify any comment into 7 toxicity categories with confidence scores and risk levels.
</div>
<div class='info-card'>
<b>📁 Batch Moderation</b><br>
Upload a CSV of thousands of comments and get predictions for every row instantly.
</div>
<div class='info-card'>
<b>📊 Analytics Dashboard</b><br>
Explore dataset statistics, class distributions, word frequencies, and more.
</div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div class='info-card'>
<b>☁ WordCloud Visualisation</b><br>
Visualise the most frequent words in toxic vs non-toxic comments.
</div>
<div class='info-card'>
<b>📈 Model Evaluation</b><br>
Compare Logistic Regression vs Naïve Bayes on accuracy, F1, confusion matrix, and CV score.
</div>
<div class='info-card'>
<b>⬇ Downloadable Reports</b><br>
Export predictions as CSV or a fully styled Excel moderation report.
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🏷️ Toxicity Categories")
    cats = {
        "Non-Toxic":     ("✅", "#06D6A0"),
        "Toxic":         ("⚡", "#FF8C42"),
        "Severe Toxic":  ("💀", "#FF4C4C"),
        "Obscene":       ("🔞", "#FF6B6B"),
        "Threat":        ("⚠️", "#FFD166"),
        "Insult":        ("😠", "#F4A261"),
        "Identity Hate": ("🚫", "#E76F51"),
    }
    cols = st.columns(7)
    for col, (name, (icon, color)) in zip(cols, cats.items()):
        col.markdown(
            f"<div style='text-align:center;background:#161B22;border:1px solid #21262D;"
            f"border-top:3px solid {color};border-radius:8px;padding:12px 6px;'>"
            f"<div style='font-size:1.6rem'>{icon}</div>"
            f"<div style='font-size:0.78rem;margin-top:6px;color:{color};font-weight:700'>{name}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.info("💡 **Quick Start:** Use the sidebar to navigate. Head to **🤖 Toxic Comment Prediction** to try the model, or **📁 Batch CSV Prediction** to process a file.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Analytics":
    st.markdown("<h2 style='color:#4ECDC4'>📊 Dataset Analytics</h2>", unsafe_allow_html=True)
    st.markdown("Upload your `train.csv` (Jigsaw dataset) to explore statistics and visualisations.")

    uploaded = st.file_uploader("Upload train.csv", type=["csv"])
    u = _utils()

    @st.cache_data(show_spinner="Parsing dataset …")
    def _load_df(data: bytes) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(data))

    if uploaded:
        df = _load_df(uploaded.read())
        st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns.")

        # Stats
        st.markdown("#### 📋 Dataset Overview")
        stats = u.dataset_stats(df)
        cols  = st.columns(len(stats))
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k, f"{v:,}" if isinstance(v, int) else v)

        st.markdown("#### 🔎 Column Description")
        desc = pd.DataFrame({
            "Column":      df.columns.tolist(),
            "Dtype":       df.dtypes.astype(str).tolist(),
            "Non-Null":    df.notnull().sum().tolist(),
            "Null Count":  df.isnull().sum().tolist(),
            "Sample":      [str(df[c].iloc[0])[:60] for c in df.columns],
        })
        st.dataframe(desc, use_container_width=True)

        st.markdown("#### 📊 Class Distribution")
        label_cols = [c for c in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"] if c in df.columns]
        if label_cols:
            st.pyplot(u.plot_class_distribution(df))

        st.markdown("#### 📏 Comment Length Distribution")
        st.pyplot(u.plot_comment_length_distribution(df))

        st.markdown("#### 📝 Word Frequency (raw text)")
        top_n = st.slider("Top N words", 10, 50, 20, 5)
        st.pyplot(u.plot_word_frequency(df, top_n=top_n))

        st.markdown("#### 🔍 Sample Data")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.markdown("""
<div class='info-card'>
<b>📁 Dataset not uploaded yet.</b><br><br>
Download the Jigsaw Toxic Comment Classification dataset from Kaggle:<br>
<a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"
   style="color:#4ECDC4">kaggle.com → Jigsaw Toxic Comment Classification</a><br><br>
Then upload <code>train.csv</code> using the file uploader above.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TEXT PREPROCESSING DEMO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧹 Text Preprocessing Demo":
    st.markdown("<h2 style='color:#4ECDC4'>🧹 Text Preprocessing Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("Enter any text below to see every preprocessing step applied in sequence.")

    default = "Hello!! <b>Visit https://example.com</b> for FREE 💰 😡 discount-123 offers!!!"
    text    = st.text_area("Raw Input Text", value=default, height=120)

    if st.button("▶ Run Preprocessing Pipeline"):
        from preprocessing import preprocess_text
        steps = preprocess_text(text, return_steps=True)
        st.markdown("---")
        for step_name, result in steps.items():
            icon = "✅" if "Final" in step_name else "🔹"
            bg   = "#1a2a1a" if "Final" in step_name else "#161B22"
            border = "#06D6A0" if "Final" in step_name else "#21262D"
            st.markdown(
                f"<div style='background:{bg};border:1px solid {border};"
                f"border-left:4px solid {border};border-radius:8px;padding:12px 16px;margin:6px 0;'>"
                f"<b style='color:#4ECDC4'>{icon} {step_name}</b><br>"
                f"<code style='color:#E6EDF3;font-size:0.88rem;word-break:break-word'>{result}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TOXIC COMMENT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Toxic Comment Prediction":
    st.markdown("<h2 style='color:#4ECDC4'>🤖 Toxic Comment Prediction</h2>", unsafe_allow_html=True)
    st.markdown("Enter a comment and the model will classify it in real time.")

    EXAMPLES = [
        "Select an example …",
        "I love this community! Everyone is so helpful.",
        "You are a complete idiot and should shut up.",
        "I will find you and make you regret this.",
        "People like you make me sick, go away.",
        "Thanks for sharing! Great tutorial.",
        "Stop spreading hate, people of your type are disgusting.",
    ]

    example = st.selectbox("📋 Try an example comment", EXAMPLES)
    comment = st.text_area(
        "💬 Your Comment",
        value="" if example == EXAMPLES[0] else example,
        height=140,
        placeholder="Type or paste a comment here …",
    )

    if st.button("🔍 Analyse Comment") and comment.strip():
        try:
            predict_single, *_ = _load_predict()
            with st.spinner("Analysing …"):
                result = predict_single(comment)

            risk    = result["risk_level"].lower()
            r_class = f"risk-{risk}"
            icons   = {"Critical": "💀", "High": "🚨", "Medium": "😠", "Low": "✅"}
            r_icon  = icons.get(result["risk_level"], "❓")
            r_color = {"Critical":"#FF4C4C","High":"#FF8C42","Medium":"#FFD166","Low":"#06D6A0"}.get(result["risk_level"],"#C9D1D9")

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("🏷️ Predicted Category", result["label"])
            col2.metric("📊 Confidence", f"{result['confidence']:.1f}%")
            col3.metric("⚡ Risk Level", result["risk_level"])

            st.markdown(
                f"<div class='info-card {r_class}'>"
                f"<b style='font-size:1.1rem'>{r_icon} {result['risk_level']} Risk – {result['label']}</b><br><br>"
                f"<b>🛡️ Suggested Action:</b> {result['suggested_action']}<br><br>"
                f"<b>🧹 Cleaned Text:</b> <code style='font-size:0.85rem'>{result['cleaned_text'][:300]}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### 📊 Prediction Probability Distribution")
            from evaluation import plot_prediction_probabilities
            st.pyplot(plot_prediction_probabilities(result["probabilities"]))

        except FileNotFoundError:
            st.error("❌ Model not found. Please train the model first using `python train_model.py --data data/train.csv`.")
    elif st.button("🔍 Analyse Comment"):
        st.warning("Please enter a comment first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown("<h2 style='color:#4ECDC4'>📈 Model Performance</h2>", unsafe_allow_html=True)

    meta = _load_meta()
    if not meta:
        st.warning("⚠️ No trained model found. Run `python train_model.py --data data/train.csv` first.")
        st.stop()

    plot_model_comparison, plot_confusion_matrix, _, metrics_dataframe = _eval_charts()

    st.markdown(f"#### ✅ Best Model: **{meta.get('best', 'N/A')}**")
    st.markdown("---")

    st.markdown("#### 📋 Metrics Summary")
    df_metrics = metrics_dataframe(meta)
    st.dataframe(df_metrics.set_index("Model"), use_container_width=True)

    st.markdown("#### 📊 Model Comparison Chart")
    st.pyplot(plot_model_comparison(meta))

    st.markdown("#### 🟦 Confusion Matrix")
    tabs = st.tabs(["Logistic Regression", "Multinomial Naïve Bayes"])
    with tabs[0]:
        st.pyplot(plot_confusion_matrix(meta, "lr"))
    with tabs[1]:
        st.pyplot(plot_confusion_matrix(meta, "nb"))

    st.markdown("#### 📝 Classification Report")
    for key, label in [("lr", "Logistic Regression"), ("nb", "Multinomial Naïve Bayes")]:
        with st.expander(f"📄 {label}"):
            report = meta.get(key, {}).get("classification_report", "Not available.")
            st.code(report, language=None)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: WORDCLOUD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "☁ WordCloud":
    st.markdown("<h2 style='color:#4ECDC4'>☁ WordCloud Visualisation</h2>", unsafe_allow_html=True)
    st.markdown("Upload `train.csv` to generate WordClouds for toxic and non-toxic comments.")

    uploaded = st.file_uploader("Upload train.csv", type=["csv"], key="wc_upload")
    u = _utils()

    if uploaded:
        @st.cache_data(show_spinner="Generating WordClouds …")
        def _gen_wc(data: bytes):
            df = pd.read_csv(io.BytesIO(data))
            return u.toxic_wordcloud(df), u.nontoxic_wordcloud(df)

        fig_toxic, fig_clean = _gen_wc(uploaded.read())
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_toxic)
        with col2:
            st.pyplot(fig_clean)
    else:
        st.info("Upload `train.csv` to see the WordClouds.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH CSV PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch CSV Prediction":
    st.markdown("<h2 style='color:#4ECDC4'>📁 Batch CSV Prediction</h2>", unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV with a **`comment_text`** column. "
        "ToxiShield AI will classify every row and let you download the results."
    )

    u = _utils()

    with st.expander("📥 Download a sample CSV to test"):
        sample_df  = u.generate_sample_df(10)
        sample_csv = sample_df.to_csv(index=False).encode()
        st.download_button("⬇ Download sample_comments.csv", sample_csv,
                           file_name="sample_comments.csv", mime="text/csv")
        st.dataframe(sample_df, use_container_width=True)

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"], key="batch_upload")

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        if "comment_text" not in raw_df.columns:
            st.error("❌ The CSV must contain a `comment_text` column.")
        else:
            st.success(f"✅ {len(raw_df):,} comments loaded.")
            st.dataframe(raw_df.head(10), use_container_width=True)

            if st.button("🚀 Run Batch Prediction"):
                try:
                    _, predict_batch, *_ = _load_predict()
                    with st.spinner(f"Classifying {len(raw_df):,} comments …"):
                        result_df = predict_batch(raw_df)

                    st.success("✅ Batch prediction complete!")
                    st.session_state["batch_results"] = result_df

                    st.markdown("#### 🔍 Preview Results")
                    st.dataframe(result_df, use_container_width=True)

                    # Summary
                    st.markdown("#### 📊 Label Distribution")
                    dist = result_df["predicted_label"].value_counts().reset_index()
                    dist.columns = ["Category", "Count"]
                    col1, col2 = st.columns(2)
                    col1.dataframe(dist, use_container_width=True)
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5, 4))
                    fig.patch.set_facecolor("#0E1117")
                    ax.set_facecolor("#161B22")
                    ax.pie(dist["Count"], labels=dist["Category"], autopct="%1.1f%%",
                           textprops={"color":"#C9D1D9","fontsize":8},
                           wedgeprops={"edgecolor":"#0E1117"})
                    col2.pyplot(fig)

                except FileNotFoundError:
                    st.error("❌ Model not found. Train the model first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DOWNLOAD RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⬇ Download Results":
    st.markdown("<h2 style='color:#4ECDC4'>⬇ Download Results</h2>", unsafe_allow_html=True)

    u = _utils()
    results = st.session_state.get("batch_results", None)

    if results is None:
        st.warning("⚠️ No batch predictions available yet. Run a **📁 Batch CSV Prediction** first.")
    else:
        st.success(f"✅ {len(results):,} predictions ready for download.")
        st.dataframe(results.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇ Download predictions.csv",
                data=u.predictions_to_csv(results),
                file_name="predictions.csv",
                mime="text/csv",
            )
        with col2:
            st.download_button(
                "⬇ Download moderation_report.xlsx",
                data=u.predictions_to_xlsx(results),
                file_name="moderation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.markdown("#### 📋 File Contents")
        st.markdown("""
<div class='info-card'>
<b>predictions.csv</b><br>
Raw CSV with columns: comment_text, cleaned_text, predicted_label, confidence, risk_level, suggested_action.
</div>
<div class='info-card'>
<b>moderation_report.xlsx</b><br>
Styled Excel workbook with two sheets:
<ul>
<li><b>Predictions</b> – colour-coded by risk level (Critical / High / Medium / Low)</li>
<li><b>Summary</b> – category count breakdown</li>
</ul>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ About":
    st.markdown("<h2 style='color:#4ECDC4'>ℹ About ToxiShield AI</h2>", unsafe_allow_html=True)
    st.markdown("""
<div class='info-card'>
<h4>🛡️ Project Overview</h4>
ToxiShield AI is a production-ready NLP system that automatically detects and classifies toxic
comments into seven categories using machine learning. Built for real-world content moderation
pipelines, it combines TF-IDF text vectorisation with Logistic Regression and Multinomial
Naïve Bayes classifiers, evaluated side-by-side to select the best performer.
</div>

<div class='info-card'>
<h4>🧠 Tech Stack</h4>
<ul>
<li><b>Language:</b> Python 3.11+</li>
<li><b>ML:</b> Scikit-learn (Logistic Regression, Multinomial NB), TF-IDF Vectoriser</li>
<li><b>NLP:</b> NLTK (tokenisation, stopwords, lemmatisation)</li>
<li><b>UI:</b> Streamlit</li>
<li><b>Visualisation:</b> Matplotlib, Seaborn, WordCloud</li>
<li><b>Export:</b> Pandas, OpenPyXL</li>
</ul>
</div>

<div class='info-card'>
<h4>📊 Dataset</h4>
Jigsaw Toxic Comment Classification Challenge (Kaggle) — ~159 k Wikipedia talk-page comments
labelled across six toxicity categories.
</div>

<div class='info-card'>
<h4>🚀 Quick Start</h4>
<pre><code>git clone https://github.com/yourusername/toxishield-ai
cd toxishield-ai
pip install -r requirements.txt

# Download Jigsaw dataset from Kaggle → place train.csv in data/
python train_model.py --data data/train.csv

streamlit run app.py</code></pre>
</div>

<div class='info-card'>
<h4>🔮 Future Improvements</h4>
<ul>
<li>Fine-tuned BERT / DistilBERT transformer model</li>
<li>Multilingual toxicity detection</li>
<li>REST API endpoint (FastAPI)</li>
<li>Docker containerisation</li>
<li>Real-time WebSocket moderation feed</li>
<li>Active learning feedback loop</li>
</ul>
</div>

<div class='info-card'>
<h4>📄 License</h4>
MIT License – free for personal and commercial use.
</div>
""", unsafe_allow_html=True)

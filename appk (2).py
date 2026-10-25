import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="CardioCluster AI",
    page_icon="🫀",
    layout="wide"
)

# Attractive GUI Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #142744);
    color: white;
}
[data-testid="stSidebar"] {
    background: #0b1628;
}
.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}
.hero {
    padding: 28px;
    border: 1px solid #31577c;
    border-radius: 22px;
    background: linear-gradient(120deg, #163f63, #1b2e50);
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 42px;
}
.hero p {
    color: #c4d7ee;
    font-size: 17px;
}
.metric {
    padding: 20px;
    border: 1px solid #2b4568;
    background: #11213a;
    border-radius: 16px;
}
.stButton button, .stFormSubmitButton button {
    background: linear-gradient(90deg, #00bfa6, #3b82f6);
    color: white;
    border-radius: 12px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🫀 CardioCluster AI</h1>
    <p>Smart Heart Dataset Analytics Dashboard</p>
    <p>Explore data, visualize clusters and predict patient groups reliably.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Workspace")

    k = st.slider(
        "Number of Clusters (K)",
        min_value=2,
        max_value=10,
        value=5
    )

    try:
        df_raw=pd.read_excel("heart_disease_uci.xls")

except FileNotFoundError:
    st.error(
        "heart_disease_uci.xls ko app.py ke same "
        "folder mein rakho ya sidebar se upload karo."
    )
    st.stop()

except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

if df_raw.empty:
    st.error("Dataset empty hai.")
    st.stop()

# Clean raw dataframe columns
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
df_raw.columns = [str(c).strip() for c in df_raw.columns]
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()

# Keep a clean working copy for ML by dropping id, num, and unnamed columns
df_model = df_raw.drop(
    columns=[c for c in df_raw.columns if c.lower() in ["id", "num", "unnamed: 0"]],
    errors="ignore"
).copy()

# Missing values handling
for col in df_model.columns:
    if pd.api.types.is_numeric_dtype(df_model[col]):
        median_value = df_model[col].median()
        if pd.isna(median_value):
            median_value = 0
        df_model[col] = df_model[col].fillna(median_value)
    else:
        mode_value = df_model[col].mode(dropna=True)
        if len(mode_value) > 0:
            fill_value = mode_value.iloc[0]
        else:
            fill_value = "Unknown"
        df_model[col] = df_model[col].fillna(fill_value)

# Encode categorical columns for training
encoded_df = pd.get_dummies(df_model, dtype=float)
encoded_df = encoded_df.loc[:, ~encoded_df.columns.duplicated()].copy()

if len(df_model) <= k:
    st.error("Dataset mein clusters ke liye aur records chahiye.")
    st.stop()

if encoded_df.shape[1] == 0:
    st.error("Dataset mein usable columns nahi hain.")
    st.stop()

# Train K-Means and save artifacts
scaler = StandardScaler()
X = scaler.fit_transform(encoded_df)

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
labels = model.fit_predict(X)

# Save model state for strict feature consistency
model_data = {
    "model": model,
    "scaler": scaler,
    "feature_columns": encoded_df.columns.tolist()
}
joblib.dump(model_data, "heart_kmeans_model.joblib")

# Silhouette score
try:
    score = silhouette_score(X, labels)
except Exception:
    score = np.nan

# Tabs setup
tab1, tab2, tab3 = st.tabs([
    "✨ Dashboard",
    "🎯 Predict Cluster",
    "📊 Visualizations"
])

# DASHBOARD TAB
with tab1:
    st.subheader("📈 Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, len(df_model), "Total Records"),
        (c2, len(df_model.columns), "Features"),
        (c3, k, "Clusters"),
        (c4, f"{score:.3f}" if np.isfinite(score) else "N/A", "Silhouette Score")
    ]

    for col, value, label in metrics:
        with col:
            st.markdown(
                f"""
                <div class="metric">
                    <p>{label}</p>
                    <h2>{value}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

    left, right = st.columns(2)

    with left:
        st.subheader("Cluster Distribution")
        counts = pd.Series(labels).value_counts().sort_index()
        chart_df = pd.DataFrame({
            "Cluster": [f"Cluster {i}" for i in counts.index],
            "Records": counts.values
        })

        fig = px.bar(
            chart_df,
            x="Cluster",
            y="Records",
            color="Cluster",
            text="Records",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Dataset Preview")
        st.dataframe(df_model.head(10), use_container_width=True)

# PREDICTION TAB
with tab2:
    st.subheader("🎯 Predict a New Profile")
    st.info(
        "Values choose karo aur Predict Cluster button dabao. "
        "Aap niche direct dataset row index se bhi test kar sakte hain!"
    )

    # Quick test helper using actual dataset rows
    with st.expander("🧪 Quick Test with Actual Dataset Row"):
        test_idx = st.number_input("Enter Row Index (0 to len(df)-1)", min_value=0, max_value=len(df_model)-1, value=0)
        if st.button("Test This Dataset Row"):
            row_sample = df_model.iloc[[test_idx]]
            row_encoded = pd.get_dummies(row_sample, dtype=float)
            row_encoded = row_encoded.reindex(columns=encoded_df.columns, fill_value=0)
            row_scaled = scaler.transform(row_encoded)
            pred_cluster = int(model.predict(row_scaled)[0])
            st.success(f"Dataset Row {test_idx} belongs to ---> **Cluster {pred_cluster}**")
            st.dataframe(row_sample, use_container_width=True)

    with st.form("prediction_form"):
        values = {}
        input_cols = st.columns(3)

        for i, col in enumerate(df_model.columns):
            series = df_model[col]
            with input_cols[i % 3]:
                if pd.api.types.is_numeric_dtype(series):
                    minimum = float(series.min())
                    maximum = float(series.max())
                    default = float(series.median())

                    if not np.isfinite(minimum): minimum = 0.0
                    if not np.isfinite(maximum): maximum = minimum
                    if not np.isfinite(default): default = minimum

                    if minimum == maximum:
                        values[col] = st.number_input(col, value=default)
                    else:
                        values[col] = st.number_input(
                            col,
                            min_value=minimum,
                            max_value=maximum,
                            value=min(max(default, minimum), maximum)
                        )
                else:
                    options = series.astype(str).value_counts().index.tolist()
                    if not options:
                        options = ["Unknown"]
                    values[col] = st.selectbox(col, options)

        submitted = st.form_submit_button(
            "🚀 Predict Cluster",
            use_container_width=True
        )

    if submitted:
        new_row = pd.DataFrame([values], columns=df_model.columns)
        
        saved_data = joblib.load("heart_kmeans_model.joblib")
        loaded_model = saved_data["model"]
        loaded_scaler = saved_data["scaler"]
        train_cols = saved_data["feature_columns"]

        new_encoded = pd.get_dummies(new_row, dtype=float)
        new_encoded = new_encoded.loc[:, ~new_encoded.columns.duplicated()].copy()
        new_encoded = new_encoded.reindex(columns=train_cols, fill_value=0)

        new_scaled = loaded_scaler.transform(new_encoded)
        prediction = int(loaded_model.predict(new_scaled)[0])

        st.success("Prediction completed successfully!")

        st.markdown(
            f"""
            <div class="metric">
                <p>Predicted Group</p>
                <h1>🫀 Cluster {prediction}</h1>
                <p>Profile successfully evaluated using consistent cluster centroids.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Entered Profile")
        st.dataframe(new_row, use_container_width=True)

# VISUALIZATIONS TAB
with tab3:
    st.subheader("📊 PCA Cluster Visualization")

    if X.shape[1] >= 2 and len(df_model) >= 2:
        points = PCA(n_components=2, random_state=42).fit_transform(X)
        plot_df = pd.DataFrame({
            "PCA 1": points[:, 0],
            "PCA 2": points[:, 1],
            "Cluster": [f"Cluster {x}" for x in labels]
        })

        fig = px.scatter(
            plot_df,
            x="PCA 1",
            y="PCA 2",
            color="Cluster",
            title="Patient Groups Using PCA",
            template="plotly_dark",
            opacity=0.85,
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster-wise Feature Averages")
    numeric_df = df_model.select_dtypes(include=np.number).copy()

    if not numeric_df.empty:
        numeric_df["Cluster"] = labels
        summary = numeric_df.groupby("Cluster").mean(numeric_only=True).round(2)
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("No numeric columns available.")

st.markdown("---")
st.caption(
    "Built with Streamlit, Pandas, Plotly and Scikit-learn. "
    "Educational clustering project only; not for medical diagnosis."
)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score

# st.set_page_config(
#     page_title="CardioCluster AI",
#     page_icon="🫀",
#     layout="wide"
# )

# # Attractive GUI
# st.markdown("""
# <style>
# .stApp {
#     background: linear-gradient(135deg, #07111f, #142744);
#     color: white;
# }
# [data-testid="stSidebar"] {
#     background: #0b1628;
# }
# .block-container {
#     padding-top: 2rem;
#     max-width: 1450px;
# }
# .hero {
#     padding: 28px;
#     border: 1px solid #31577c;
#     border-radius: 22px;
#     background: linear-gradient(120deg, #163f63, #1b2e50);
#     margin-bottom: 24px;
# }
# .hero h1 {
#     color: white;
#     font-size: 42px;
# }
# .hero p {
#     color: #c4d7ee;
#     font-size: 17px;
# }
# .metric {
#     padding: 20px;
#     border: 1px solid #2b4568;
#     background: #11213a;
#     border-radius: 16px;
# }
# .stButton button, .stFormSubmitButton button {
#     background: linear-gradient(90deg, #00bfa6, #3b82f6);
#     color: white;
#     border-radius: 12px;
#     font-weight: bold;
# }
# </style>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div class="hero">
#     <h1>🫀 CardioCluster AI</h1>
#     <p>Smart Heart Dataset Analytics Dashboard</p>
#     <p>Explore data, visualize clusters and predict a new profile's group.</p>
# </div>
# """, unsafe_allow_html=True)

# # Sidebar
# with st.sidebar:
#     st.title("⚙️ Workspace")

#     k = st.slider(
#         "Number of Clusters (K)",
#         min_value=2,
#         max_value=10,
#         value=5
#     )

#     uploaded = st.file_uploader(
#         "Upload Heart Disease CSV",
#         type=["csv"]
#     )

# # Load dataset
# try:
#     if uploaded is not None:
#         df = pd.read_csv(uploaded)
#     else:
#         df = pd.read_csv("heart_disease_uci.csv")

# except FileNotFoundError:
#     st.error(
#         "heart_disease_uci.csv ko appk.py ke same "
#         "folder mein rakho ya sidebar se upload karo."
#     )
#     st.stop()

# except Exception as e:
#     st.error(f"Dataset loading error: {e}")
#     st.stop()

# if df.empty:
#     st.error("Dataset empty hai.")
#     st.stop()

# # Fix duplicate column names
# df = df.loc[:, ~df.columns.duplicated()].copy()
# df.columns = [str(c).strip() for c in df.columns]
# df = df.loc[:, ~df.columns.duplicated()].copy()

# # Remove empty index column if present
# df = df.drop(
#     columns=[
#         c for c in df.columns
#         if c.lower() == "unnamed: 0"
#     ],
#     errors="ignore"
# )

# # Missing values handling
# for col in df.columns:

#     if pd.api.types.is_numeric_dtype(df[col]):
#         median_value = df[col].median()

#         if pd.isna(median_value):
#             median_value = 0

#         df[col] = df[col].fillna(median_value)

#     else:
#         mode_value = df[col].mode(dropna=True)

#         if len(mode_value) > 0:
#             fill_value = mode_value.iloc[0]
#         else:
#             fill_value = "Unknown"

#         df[col] = df[col].fillna(fill_value)

# # Encode categorical columns
# df = df.drop(columns=["id", "num"], errors="ignore")
# encoded_df = pd.get_dummies(df, dtype=float)

# # Ensure unique encoded column names
# encoded_df = encoded_df.loc[
#     :, ~encoded_df.columns.duplicated()
# ].copy()

# if len(df) <= k:
#     st.error("Dataset mein clusters ke liye aur records chahiye.")
#     st.stop()

# if encoded_df.shape[1] == 0:
#     st.error("Dataset mein usable columns nahi hain.")
#     st.stop()

# # Train K-Means
# scaler = StandardScaler()
# X = scaler.fit_transform(encoded_df)

# model = KMeans(
#     n_clusters=k,
#     random_state=42,
#     n_init=10
# )

# labels = model.fit_predict(X)

# # Silhouette score
# try:
#     score = silhouette_score(X, labels)
# except Exception:
#     score = np.nan

# # Tabs
# tab1, tab2, tab3 = st.tabs([
#     "✨ Dashboard",
#     "🎯 Predict Cluster",
#     "📊 Visualizations"
# ])

# # DASHBOARD
# with tab1:

#     st.subheader("📈 Project Overview")

#     c1, c2, c3, c4 = st.columns(4)

#     metrics = [
#         (c1, len(df), "Total Records"),
#         (c2, len(df.columns), "Features"),
#         (c3, k, "Clusters"),
#         (
#             c4,
#             f"{score:.3f}" if np.isfinite(score) else "N/A",
#             "Silhouette Score"
#         )
#     ]

#     for col, value, label in metrics:
#         with col:
#             st.markdown(
#                 f"""
#                 <div class="metric">
#                     <p>{label}</p>
#                     <h2>{value}</h2>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )

#     left, right = st.columns(2)

#     with left:
#         st.subheader("Cluster Distribution")

#         counts = pd.Series(labels).value_counts().sort_index()

#         chart_df = pd.DataFrame({
#             "Cluster": [
#                 f"Cluster {i}" for i in counts.index
#             ],
#             "Records": counts.values
#         })

#         fig = px.bar(
#             chart_df,
#             x="Cluster",
#             y="Records",
#             color="Cluster",
#             text="Records",
#             template="plotly_dark",
#             color_discrete_sequence=px.colors.qualitative.Vivid
#         )

#         fig.update_layout(
#             showlegend=False,
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(0,0,0,0)"
#         )

#         st.plotly_chart(fig, use_container_width=True)

#     with right:
#         st.subheader("Dataset Preview")
#         st.dataframe(df.head(10), use_container_width=True)

# # PREDICTION
# with tab2:

#     st.subheader("🎯 Predict a New Profile")

#     st.info(
#         "Values choose karo aur Predict Cluster button dabao. "
#         "Result similar records ka group batata hai, medical diagnosis nahi."
#     )

#     with st.form("prediction_form"):

#         values = {}
#         input_cols = st.columns(3)

#         for i, col in enumerate(df.columns):

#             series = df[col]

#             with input_cols[i % 3]:

#                 if pd.api.types.is_numeric_dtype(series):

#                     minimum = float(series.min())
#                     maximum = float(series.max())
#                     default = float(series.median())

#                     if not np.isfinite(minimum):
#                         minimum = 0.0

#                     if not np.isfinite(maximum):
#                         maximum = minimum

#                     if not np.isfinite(default):
#                         default = minimum

#                     if minimum == maximum:

#                         values[col] = st.number_input(
#                             col,
#                             value=default
#                         )

#                     else:

#                         values[col] = st.number_input(
#                             col,
#                             min_value=minimum,
#                             max_value=maximum,
#                             value=min(
#                                 max(default, minimum),
#                                 maximum
#                             )
#                         )

#                 else:

#                     options = (
#                         series.astype(str)
#                         .value_counts()
#                         .index.tolist()
#                     )

#                     if not options:
#                         options = ["Unknown"]

#                     values[col] = st.selectbox(
#                         col,
#                         options
#                     )

#         submitted = st.form_submit_button(
#             "🚀 Predict Cluster",
#             use_container_width=True
#         )

#     if submitted:

#         new_row = pd.DataFrame(
#             [values],
#             columns=df.columns
#         )

#         # Encode new profile separately
#         new_encoded = pd.get_dummies(
#             new_row,
#             dtype=float
#         )

#         # Remove duplicate encoded labels
#         new_encoded = new_encoded.loc[
#             :, ~new_encoded.columns.duplicated()
#         ].copy()

#         # Match training columns exactly
#         new_encoded = new_encoded.reindex(
#             columns=encoded_df.columns,
#             fill_value=0
#         )

#         # Predict using trained model
#         new_scaled = scaler.transform(new_encoded)

#         prediction = int(
#             model.predict(new_scaled)[0]
#         )

#         st.success("Prediction completed!")

#         st.markdown(
#             f"""
#             <div class="metric">
#                 <p>Predicted Group</p>
#                 <h1>🫀 Cluster {prediction}</h1>
#                 <p>Profile grouped with similar dataset records.</p>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#         st.subheader("Entered Profile")
#         st.dataframe(new_row, use_container_width=True)

# # VISUALIZATIONS
# with tab3:

#     st.subheader("📊 PCA Cluster Visualization")

#     if X.shape[1] >= 2 and len(df) >= 2:

#         points = PCA(
#             n_components=2,
#             random_state=42
#         ).fit_transform(X)

#         plot_df = pd.DataFrame({
#             "PCA 1": points[:, 0],
#             "PCA 2": points[:, 1],
#             "Cluster": [
#                 f"Cluster {x}" for x in labels
#             ]
#         })

#         fig = px.scatter(
#             plot_df,
#             x="PCA 1",
#             y="PCA 2",
#             color="Cluster",
#             title="Patient Groups Using PCA",
#             template="plotly_dark",
#             opacity=0.85,
#             color_discrete_sequence=px.colors.qualitative.Vivid
#         )

#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(0,0,0,0)"
#         )

#         st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Cluster-wise Feature Averages")

#     numeric_df = df.select_dtypes(
#         include=np.number
#     ).copy()

#     if not numeric_df.empty:

#         numeric_df["Cluster"] = labels

#         summary = (
#             numeric_df.groupby("Cluster")
#             .mean(numeric_only=True)
#             .round(2)
#         )

#         st.dataframe(
#             summary,
#             use_container_width=True
#         )

#     else:
#         st.info("No numeric columns available.")

# st.markdown("---")

# st.caption(
#     "Built with Streamlit, Pandas, Plotly and Scikit-learn. "
#     "Educational clustering project only; not for medical diagnosis."
# )

# import streamlit as st
# import pandas as pd
# import joblib

# st.set_page_config(page_title="Heart Cluster AI", page_icon="❤️", layout="wide")

# st.title("❤️ Heart Disease Clustering")
# st.write("K-Means Clustering Dashboard")

# # Load the trained model
# saved = joblib.load("heart_kmeans_model.joblib")

# model = saved["model"]
# scaler = saved["scaler"]
# feature_columns = saved["feature_columns"]

# st.subheader("Enter Patient Details")

# age = st.number_input("Age", 1, 120, 50)
# sex = st.selectbox("Sex", ["Male", "Female"])
# cp = st.selectbox(
#     "Chest Pain Type",
#     ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
# )
# trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
# chol = st.number_input("Cholesterol", 0, 700, 200)
# fbs = st.selectbox("Fasting Blood Sugar > 120", [False, True])
# restecg = st.selectbox(
#     "Resting ECG",
#     ["normal", "st-t abnormality", "lv hypertrophy"]
# )
# thalch = st.number_input("Maximum Heart Rate", 40, 250, 150)
# exang = st.selectbox("Exercise-Induced Angina", [False, True])
# oldpeak = st.number_input("ST Depression", -5.0, 10.0, 1.0)
# slope = st.selectbox("ST Slope", ["upsloping", "flat", "downsloping"])
# ca = st.number_input("Major Vessels", 0, 4, 0)
# thal = st.selectbox(
#     "Thalassemia",
#     ["normal", "fixed defect", "reversible defect"]
# )
# dataset = st.selectbox(
#     "Dataset Source",
#     ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
# )

# if st.button("Find Patient Cluster"):

#     patient = pd.DataFrame([{
#         "age": age,
#         "sex": sex,
#         "dataset": dataset,
#         "cp": cp,
#         "trestbps": trestbps,
#         "chol": chol,
#         "fbs": fbs,
#         "restecg": restecg,
#         "thalch": thalch,
#         "exang": exang,
#         "oldpeak": oldpeak,
#         "slope": slope,
#         "ca": ca,
#         "thal": thal
#     }])

#     # Match the training data's encoded columns
#     patient_encoded = pd.get_dummies(patient, drop_first=False)
#     patient_encoded = patient_encoded.reindex(
#         columns=feature_columns, fill_value=0
#     )

#     # Scale using the already-trained scaler
#     patient_scaled = scaler.transform(patient_encoded)

#     # Predict the cluster
#     cluster = model.predict(patient_scaled)[0]

#     st.success(f"Patient belongs to Cluster {cluster}")

#     st.subheader("Entered Patient Details")
#     st.dataframe(patient, use_container_width=True)

#     st.warning(
#         "This is an educational clustering result, not a medical diagnosis."
#     )

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
DEFAULT_DATA_PATH = r"c:/Users/karti/Downloads/travel (1).csv"
FEATURE_COLS = [
    "total_search_events",
    "total_bookings",
    "booking_rate",
    "avg_distance",
    "mobile_ratio",
    "package_ratio",
    "avg_adults",
    "avg_children",
    "avg_rooms",
    "avg_stay_length",
    "avg_days_to_checkin",
    "family_traveler",
    "solo_traveler",
]


@dataclass
class PipelineConfig:
    min_events: int = 3
    n_clusters: int = 4
    stay_clip_low: int = 1
    stay_clip_high: int = 30
    checkin_clip_low: int = 0
    checkin_clip_high: int = 365


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, use_batchnorm: bool = False):
        super().__init__()

        def block(in_dim, out_dim, bn=False):
            layers = [nn.Linear(in_dim, out_dim)]
            if bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            return layers

        self.encoder = nn.Sequential(
            *block(input_dim, 32, bn=use_batchnorm),
            *block(32, 16, bn=use_batchnorm),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            *block(latent_dim, 16, bn=use_batchnorm),
            *block(16, 32, bn=use_batchnorm),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataframe(uploaded_file, path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(path)


def load_and_normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    rename_map = {
        "orig_destination_distance": "distance",
        "srch_adults_cnt": "adults",
        "srch_children_cnt": "children",
        "srch_rm_cnt": "rooms",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)

    for col in ["date_time", "srch_ci", "srch_co"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "stay_length" not in df.columns and {"srch_ci", "srch_co"}.issubset(df.columns):
        df["stay_length"] = (df["srch_co"] - df["srch_ci"]).dt.days
    if "days_to_checkin" not in df.columns and {"date_time", "srch_ci"}.issubset(df.columns):
        df["days_to_checkin"] = (df["srch_ci"] - df["date_time"]).dt.days

    required = [
        "user_id",
        "is_booking",
        "is_mobile",
        "is_package",
        "distance",
        "adults",
        "children",
        "rooms",
        "stay_length",
        "days_to_checkin",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required canonical columns: {missing}")

    keep = required + [c for c in ["date_time", "srch_ci", "srch_co"] if c in df.columns]
    df = df[keep].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_and_clip(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    out["stay_length"] = out["stay_length"].clip(cfg.stay_clip_low, cfg.stay_clip_high)
    out["days_to_checkin"] = out["days_to_checkin"].clip(cfg.checkin_clip_low, cfg.checkin_clip_high)
    out.loc[out["distance"] < 0, "distance"] = np.nan

    numeric_cols = ["distance", "adults", "children", "rooms", "stay_length", "days_to_checkin"]
    binary_cols = ["is_mobile", "is_package", "is_booking"]

    for col in numeric_cols:
        out[col] = out[col].fillna(out[col].median())
    for col in binary_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def build_user_features(df: pd.DataFrame, min_events: int = 3) -> pd.DataFrame:
    grouped = df.groupby("user_id", as_index=False).agg(
        total_search_events=("user_id", "size"),
        total_bookings=("is_booking", "sum"),
        avg_distance=("distance", "mean"),
        mobile_ratio=("is_mobile", "mean"),
        package_ratio=("is_package", "mean"),
        avg_adults=("adults", "mean"),
        avg_children=("children", "mean"),
        avg_rooms=("rooms", "mean"),
        avg_stay_length=("stay_length", "mean"),
        avg_days_to_checkin=("days_to_checkin", "mean"),
    )
    grouped["booking_rate"] = grouped["total_bookings"] / grouped["total_search_events"]
    grouped["family_traveler"] = (grouped["avg_children"] > 0).astype(int)
    grouped["solo_traveler"] = (
        (grouped["avg_adults"] <= 1) & (grouped["avg_children"] == 0) & (grouped["avg_rooms"] <= 1)
    ).astype(int)
    return grouped[grouped["total_search_events"] >= min_events].reset_index(drop=True)


def build_user_features_all(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("user_id", as_index=False).agg(
        total_search_events=("user_id", "size"),
        total_bookings=("is_booking", "sum"),
        avg_distance=("distance", "mean"),
        mobile_ratio=("is_mobile", "mean"),
        package_ratio=("is_package", "mean"),
        avg_adults=("adults", "mean"),
        avg_children=("children", "mean"),
        avg_rooms=("rooms", "mean"),
        avg_stay_length=("stay_length", "mean"),
        avg_days_to_checkin=("days_to_checkin", "mean"),
    )
    grouped["booking_rate"] = grouped["total_bookings"] / grouped["total_search_events"]
    grouped["family_traveler"] = (grouped["avg_children"] > 0).astype(int)
    grouped["solo_traveler"] = (
        (grouped["avg_adults"] <= 1) & (grouped["avg_children"] == 0) & (grouped["avg_rooms"] <= 1)
    ).astype(int)
    return grouped


def train_autoencoder(
    X_scaled: np.ndarray,
    latent_dim: int,
    denoising: bool = False,
    noise_std: float = 0.0,
    use_batchnorm: bool = False,
    weight_decay: float = 0.0,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> Tuple[nn.Module, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True, drop_last=False)

    model = Autoencoder(input_dim=X_scaled.shape[1], latent_dim=latent_dim, use_batchnorm=use_batchnorm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_in = x_batch + torch.randn_like(x_batch) * noise_std if denoising else x_batch
            x_hat = model(x_in)
            loss = criterion(x_hat, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encoder(X_tensor.to(device)).cpu().numpy()
    return model, z


def assign_cluster_names(profile_df: pd.DataFrame) -> Dict[int, str]:
    names = {}
    for cid, row in profile_df.iterrows():
        tags = []
        if row["booking_rate"] >= profile_df["booking_rate"].quantile(0.75):
            tags.append("High-Booking")
        elif row["booking_rate"] <= profile_df["booking_rate"].quantile(0.25):
            tags.append("Low-Booking")
        if row["mobile_ratio"] >= profile_df["mobile_ratio"].quantile(0.75):
            tags.append("Mobile-First")
        if row["package_ratio"] >= profile_df["package_ratio"].quantile(0.75):
            tags.append("Package-Oriented")
        if row["family_traveler"] >= profile_df["family_traveler"].quantile(0.75):
            tags.append("Family")
        if row["solo_traveler"] >= profile_df["solo_traveler"].quantile(0.75):
            tags.append("Solo")
        names[cid] = " / ".join(tags) if tags else "Balanced"
    return names


def compute_retention_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Day-N retention based on days since each user's first observed event."""
    if "date_time" not in df.columns:
        return pd.DataFrame(columns=["days_since_first", "active_users", "retention_rate"])

    events = df[["user_id", "date_time"]].dropna().copy()
    if events.empty:
        return pd.DataFrame(columns=["days_since_first", "active_users", "retention_rate"])

    events["event_date"] = pd.to_datetime(events["date_time"], errors="coerce").dt.floor("D")
    events = events.dropna(subset=["event_date"])
    if events.empty:
        return pd.DataFrame(columns=["days_since_first", "active_users", "retention_rate"])

    first_dates = events.groupby("user_id")["event_date"].min().rename("first_date")
    events = events.join(first_dates, on="user_id")
    events["days_since_first"] = (events["event_date"] - events["first_date"]).dt.days.astype(int)

    curve = (
        events.groupby("days_since_first")["user_id"]
        .nunique()
        .reset_index(name="active_users")
        .sort_values("days_since_first")
    )
    cohort_size = first_dates.shape[0]
    curve["retention_rate"] = curve["active_users"] / cohort_size
    return curve


@st.cache_data(show_spinner=False)
def run_pipeline(raw_df: pd.DataFrame, cfg: PipelineConfig):
    set_seed(SEED)
    df_norm = load_and_normalize(raw_df)
    df_clean = clean_and_clip(df_norm, cfg)

    user_df = build_user_features(df_clean, min_events=cfg.min_events)
    X = user_df[FEATURE_COLS].copy()
    if X.columns.tolist() != FEATURE_COLS:
        raise ValueError("Feature mismatch before scaling.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_stage1 = PCA(n_components=0.90, random_state=SEED)
    X_pca = pca_stage1.fit_transform(X_scaled)
    kmeans1 = KMeans(n_clusters=cfg.n_clusters, random_state=SEED, n_init=20)
    labels1 = kmeans1.fit_predict(X_pca)
    sil1 = silhouette_score(X_pca, labels1)
    stage1_vis = PCA(n_components=2, random_state=SEED).fit_transform(X_scaled)

    ae_model, Z2 = train_autoencoder(X_scaled, latent_dim=5, epochs=100)
    kmeans2 = KMeans(n_clusters=cfg.n_clusters, random_state=SEED, n_init=20)
    labels2 = kmeans2.fit_predict(Z2)
    sil2 = silhouette_score(Z2, labels2)
    stage2_vis = PCA(n_components=2, random_state=SEED).fit_transform(Z2)

    dae_model, Z3 = train_autoencoder(
        X_scaled,
        latent_dim=3,
        denoising=True,
        noise_std=0.1,
        use_batchnorm=True,
        weight_decay=1e-5,
        epochs=140,
    )
    kmeans3 = KMeans(n_clusters=cfg.n_clusters, random_state=SEED, n_init=20)
    labels3 = kmeans3.fit_predict(Z3)
    sil3 = silhouette_score(Z3, labels3)
    user_df["final_cluster"] = labels3

    cluster_profile = user_df.groupby("final_cluster")[FEATURE_COLS].mean().round(3)
    cluster_counts = user_df["final_cluster"].value_counts().sort_index().rename("user_count")
    cluster_profile = cluster_profile.join(cluster_counts)
    cluster_name_map = assign_cluster_names(cluster_profile)
    user_df["cluster_name"] = user_df["final_cluster"].map(cluster_name_map)

    all_user_df = build_user_features_all(df_clean)
    X_all = all_user_df[FEATURE_COLS].copy()
    X_all_scaled = scaler.transform(X_all)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dae_model.eval()
    with torch.no_grad():
        Z_all = dae_model.encoder(torch.tensor(X_all_scaled, dtype=torch.float32, device=device)).cpu().numpy()
    all_labels = kmeans3.predict(Z_all)
    all_user_df["predicted_cluster"] = all_labels
    all_user_df["cluster_name"] = all_user_df["predicted_cluster"].map(cluster_name_map)
    all_user_df["seen_in_training"] = all_user_df["total_search_events"] >= cfg.min_events
    all_user_df["avg_group_size"] = all_user_df["avg_adults"] + all_user_df["avg_children"]
    all_user_df["latent_x"] = Z_all[:, 0]
    all_user_df["latent_y"] = Z_all[:, 1]

    retention_curve = compute_retention_curve(df_clean)

    comparison_df = pd.DataFrame(
        {
            "stage": ["Stage 1: PCA + KMeans", "Stage 2: AE + KMeans", "Stage 3: DAE + KMeans"],
            "representation_dim": [X_pca.shape[1], Z2.shape[1], Z3.shape[1]],
            "silhouette_score": [sil1, sil2, sil3],
        }
    ).sort_values("silhouette_score", ascending=False)

    return {
        "raw_shape": raw_df.shape,
        "user_df_train": user_df,
        "all_user_df": all_user_df,
        "cluster_profile": cluster_profile,
        "cluster_name_map": cluster_name_map,
        "comparison_df": comparison_df,
        "silhouettes": (sil1, sil2, sil3),
        "stage1_vis": stage1_vis,
        "stage1_labels": labels1,
        "stage2_vis": stage2_vis,
        "stage2_labels": labels2,
        "stage3_vis": Z3[:, :2],
        "stage3_labels": labels3,
        "retention_curve": retention_curve,
    }


def scatter_plot(points: np.ndarray, labels: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=labels, palette="tab10", s=35, alpha=0.9, ax=ax, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Traveler Segmentation", layout="wide")
    st.title("Traveler Behavior Segmentation")
    st.caption("Single-page app: PCA+KMeans, AE+KMeans, DAE+KMeans, cluster naming, and full-user labeling.")

    with st.sidebar:
        st.header("Inputs")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        data_path = st.text_input("Or CSV path", value=DEFAULT_DATA_PATH)
        st.divider()
        min_events = st.number_input("Min search events for training users", min_value=1, max_value=20, value=3)
        n_clusters = st.number_input("Number of clusters", min_value=2, max_value=10, value=4)
        stay_low, stay_high = st.slider("Stay length clip (days)", min_value=0, max_value=60, value=(1, 30))
        chk_low, chk_high = st.slider("Planning window clip (days)", min_value=0, max_value=730, value=(0, 365))
        run_btn = st.button("Run Segmentation", type="primary")

    if "results" not in st.session_state:
        st.session_state.results = None

    if run_btn:
        try:
            raw_df = load_dataframe(uploaded_file, data_path)
            cfg = PipelineConfig(
                min_events=min_events,
                n_clusters=n_clusters,
                stay_clip_low=stay_low,
                stay_clip_high=stay_high,
                checkin_clip_low=chk_low,
                checkin_clip_high=chk_high,
            )
            st.session_state.results = run_pipeline(raw_df, cfg)
            st.success("Pipeline completed.")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    results = st.session_state.results
    if results is None:
        st.info("Set inputs in the sidebar and click 'Run Segmentation'.")
        return

    sil1, sil2, sil3 = results["silhouettes"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows", f"{results['raw_shape'][0]:,}")
    c2.metric("Stage 1 Silhouette", f"{sil1:.4f}")
    c3.metric("Stage 2 Silhouette", f"{sil2:.4f}")
    c4.metric("Stage 3 Silhouette", f"{sil3:.4f}")

    st.subheader("Model Comparison")
    st.dataframe(results["comparison_df"], use_container_width=True)

    comp_fig, comp_ax = plt.subplots(figsize=(7, 4))
    comp_plot = results["comparison_df"].sort_values("stage")
    sns.barplot(data=comp_plot, x="stage", y="silhouette_score", palette="viridis", ax=comp_ax)
    for i, v in enumerate(comp_plot["silhouette_score"]):
        comp_ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    comp_ax.set_ylim(0, min(1.0, comp_plot["silhouette_score"].max() + 0.15))
    comp_ax.set_title("Silhouette Score by Approach")
    comp_ax.set_xlabel("")
    comp_fig.tight_layout()
    st.pyplot(comp_fig)

    st.subheader("Cluster Visuals")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.pyplot(scatter_plot(results["stage1_vis"], results["stage1_labels"], "Stage 1: PCA + KMeans"))
    with col_b:
        st.pyplot(scatter_plot(results["stage2_vis"], results["stage2_labels"], "Stage 2: AE + KMeans"))
    with col_c:
        st.pyplot(scatter_plot(results["stage3_vis"], results["stage3_labels"], "Stage 3: DAE + KMeans"))

    train_df = results["user_df_train"]
    all_df = results["all_user_df"]
    retention_df = results["retention_curve"]
    cluster_profile = results["cluster_profile"].copy().reset_index()
    cluster_profile["cluster_name"] = cluster_profile["final_cluster"].map(results["cluster_name_map"])

    one_time_users = int((all_df["total_search_events"] == 1).sum())
    total_users = int(len(all_df))
    st.caption(
        f"All labeled users: {total_users:,} | Single-use users (1 event): {one_time_users:,} "
        f"({(100 * one_time_users / max(total_users, 1)):.2f}%)"
    )

    st.subheader("Retention Curve")
    if retention_df.empty:
        st.info("Retention curve unavailable because valid `date_time` values are missing.")
    else:
        max_day = int(retention_df["days_since_first"].max())
        day_limit = st.slider("Retention horizon (days)", min_value=1, max_value=max(1, max_day), value=min(90, max(1, max_day)))
        retention_view = retention_df[retention_df["days_since_first"] <= day_limit].copy()
        retention_view["retention_pct"] = 100 * retention_view["retention_rate"]

        ret_chart = (
            alt.Chart(retention_view)
            .mark_line(point=True)
            .encode(
                x=alt.X("days_since_first:Q", title="Days Since First Event"),
                y=alt.Y("retention_pct:Q", title="Retention (%)"),
                tooltip=[
                    alt.Tooltip("days_since_first:Q", title="Day"),
                    alt.Tooltip("active_users:Q", title="Active Users"),
                    alt.Tooltip("retention_pct:Q", title="Retention %", format=".2f"),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(ret_chart, use_container_width=True)

    st.subheader("Stage 3 Cluster Overview")
    left, right = st.columns(2)
    with left:
        counts = train_df["final_cluster"].value_counts().sort_index()
        fig_cnt, ax_cnt = plt.subplots(figsize=(6, 4))
        sns.barplot(x=counts.index.astype(str), y=counts.values, palette="tab10", ax=ax_cnt)
        ax_cnt.set_title("Stage 3 Cluster Size (Training Users)")
        ax_cnt.set_xlabel("Cluster")
        ax_cnt.set_ylabel("Users")
        fig_cnt.tight_layout()
        st.pyplot(fig_cnt)
    with right:
        heat_cols = [
            "booking_rate",
            "mobile_ratio",
            "package_ratio",
            "avg_distance",
            "avg_adults",
            "avg_children",
            "avg_rooms",
            "avg_stay_length",
            "avg_days_to_checkin",
        ]
        heat = train_df.groupby("final_cluster")[heat_cols].mean()
        heat_z = (heat - heat.mean()) / heat.std(ddof=0).replace(0, 1)
        fig_h, ax_h = plt.subplots(figsize=(8, 4))
        sns.heatmap(heat_z, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax_h)
        ax_h.set_title("Stage 3 Profiles (Z-Score)")
        fig_h.tight_layout()
        st.pyplot(fig_h)

    st.subheader("Cluster Names")
    st.dataframe(
        cluster_profile[
            [
                "final_cluster",
                "cluster_name",
                "user_count",
                "booking_rate",
                "mobile_ratio",
                "package_ratio",
                "family_traveler",
                "solo_traveler",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Filter and Search Labeled Users")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        selected_clusters = st.multiselect(
            "Cluster Name",
            options=sorted(all_df["cluster_name"].dropna().unique().tolist()),
            default=sorted(all_df["cluster_name"].dropna().unique().tolist()),
        )
    with f2:
        min_search = int(all_df["total_search_events"].min())
        max_search = int(all_df["total_search_events"].max())
        search_range = st.slider("Search Events", min_value=min_search, max_value=max_search, value=(min_search, max_search))
    with f3:
        min_people = float(all_df["avg_group_size"].min())
        max_people = float(all_df["avg_group_size"].max())
        people_range = st.slider("Avg People (Adults+Children)", min_value=min_people, max_value=max_people, value=(min_people, max_people))
    with f4:
        user_query = st.text_input("Search by user_id")

    include_single_use = st.checkbox("Include single-use users (1 event)", value=True)

    filtered = all_df.copy()
    filtered = filtered[filtered["cluster_name"].isin(selected_clusters)]
    filtered = filtered[
        (filtered["total_search_events"] >= search_range[0]) & (filtered["total_search_events"] <= search_range[1])
    ]
    filtered = filtered[(filtered["avg_group_size"] >= people_range[0]) & (filtered["avg_group_size"] <= people_range[1])]
    if not include_single_use:
        filtered = filtered[filtered["total_search_events"] > 1]
    if user_query.strip():
        filtered = filtered[filtered["user_id"].astype(str).str.contains(user_query.strip(), case=False, regex=False)]

    st.subheader("Interactive User Map (Stage 3 Latent Space)")
    hover_cols = [
        "user_id",
        "cluster_name",
        "predicted_cluster",
        "total_search_events",
        "total_bookings",
        "booking_rate",
        "avg_adults",
        "avg_children",
        "avg_group_size",
        "avg_rooms",
        "avg_days_to_checkin",
        "avg_stay_length",
        "seen_in_training",
    ]
    scatter_df = filtered.copy()
    scatter_df["seen_in_training_label"] = np.where(scatter_df["seen_in_training"], "Train (>=min events)", "Scored-only (<min events)")
    interactive_scatter = (
        alt.Chart(scatter_df)
        .mark_circle(size=65, opacity=0.75)
        .encode(
            x=alt.X("latent_x:Q", title="Latent X"),
            y=alt.Y("latent_y:Q", title="Latent Y"),
            color=alt.Color("cluster_name:N", title="Cluster Label"),
            shape=alt.Shape("seen_in_training_label:N", title="Population"),
            tooltip=[alt.Tooltip(c, title=c.replace("_", " ").title()) for c in hover_cols],
        )
        .properties(height=430)
        .interactive()
    )
    st.altair_chart(interactive_scatter, use_container_width=True)

    st.subheader("Single-Use Users by Cluster")
    single_df = all_df[all_df["total_search_events"] == 1].copy()
    if single_df.empty:
        st.info("No single-use users found.")
    else:
        single_summary = (
            single_df.groupby("cluster_name", as_index=False)["user_id"]
            .count()
            .rename(columns={"user_id": "single_use_users"})
            .sort_values("single_use_users", ascending=False)
        )
        single_chart = (
            alt.Chart(single_summary)
            .mark_bar()
            .encode(
                x=alt.X("cluster_name:N", title="Cluster Label", sort="-y"),
                y=alt.Y("single_use_users:Q", title="Single-Use Users"),
                tooltip=[
                    alt.Tooltip("cluster_name:N", title="Cluster Label"),
                    alt.Tooltip("single_use_users:Q", title="Single-Use Users"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(single_chart, use_container_width=True)

    st.caption(
        f"Showing {len(filtered):,} users | Training users: {int(filtered['seen_in_training'].sum()):,} | "
        f"Additional labeled users: {int((~filtered['seen_in_training']).sum()):,}"
    )
    st.dataframe(
        filtered[
            [
                "user_id",
                "predicted_cluster",
                "cluster_name",
                "seen_in_training",
                "total_search_events",
                "total_bookings",
                "booking_rate",
                "avg_group_size",
                "avg_adults",
                "avg_children",
                "avg_rooms",
                "avg_days_to_checkin",
                "avg_stay_length",
            ]
        ].sort_values(["predicted_cluster", "total_search_events"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()

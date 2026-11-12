"""Six functional exploratory pages backed by actual synthetic computation."""
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from medreconlab.pipeline import experiment
from medreconlab.methods import METHODS

st.set_page_config(page_title="MedReconLab", layout="wide")
st.title("MedReconLab")
st.caption("Academic synthetic reconstruction only — NOT for diagnosis")
page = st.sidebar.radio("Page", ["Overview", "Reconstruction", "Comparison", "Metrics",
                                  "Experiments", "Reports"])
modality = st.sidebar.selectbox("Modality", list(METHODS))
size = st.sidebar.select_slider("Image size", options=[32, 64, 96, 128], value=64)
seed = st.sidebar.number_input("Random seed", value=0, min_value=0)
projections = st.sidebar.slider("CT projections", 10, 180, 60)
acceleration = st.sidebar.selectbox("MRI acceleration", [2, 4, 8], index=1)
noise = st.sidebar.slider("Measurement noise", 0.0, 0.1, 0.01, 0.005)
use_unet = st.sidebar.checkbox("Use locally trained CT U-Net", value=False,
                               disabled=modality != "ct")
checkpoint_path = None
if use_unet:
    st.sidebar.caption("Trusted local checkpoints only. No external uploads.")
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint inside outputs/", value="outputs/ct_unet/best.pt")



@st.cache_data(show_spinner=False)
def compute(modality, size, seed, projections, acceleration, noise):
    return experiment(modality=modality, size=size, seed=seed, projections=projections,
                      acceleration=acceleration, noise=noise)


if page == "Overview":
    st.markdown("""### Real synthetic reconstruction
    CT: filtered backprojection and SART. MRI: zero-filled FFT and iterative k-space
    projection. X-ray: Wiener and Richardson-Lucy restoration.
    Physics-informed PyTorch optimizers are optional library functions.
    Advanced learned models are **not implemented**. No real patient data is loaded.
    """)
    st.json({"available_methods": METHODS})
elif page == "Experiments":
    st.markdown("### Recorded CT experiments")
    st.caption("Read-only local SQLite MLflow data. No remote tracking servers or artifacts.")
    database = st.text_input("Tracking database inside project", value="mlflow.db")
    if st.button("Load MLflow runs"):
        try:
            from medreconlab.experiment_view import read_experiment_history

            summaries, curves = read_experiment_history(database)
            st.session_state["mlflow_summaries"] = summaries
            st.session_state["mlflow_curves"] = curves
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            st.error(str(exc))
            st.session_state.pop("mlflow_summaries", None)
            st.session_state.pop("mlflow_curves", None)
    summaries = st.session_state.get("mlflow_summaries", [])
    curves = st.session_state.get("mlflow_curves", {})
    if summaries:
        st.dataframe(pd.DataFrame(summaries), use_container_width=True)
        selected_id = st.selectbox("Inspect run", [r["run_id"] for r in summaries],
                                  format_func=lambda run_id: next(
                                      (f'{r["name"]} | {run_id[:10]}' for r in summaries
                                       if r["run_id"] == run_id), run_id))
        history = pd.DataFrame(curves.get(selected_id, []))
        if not history.empty:
            st.line_chart(history.pivot_table(index="step", columns="metric",
                                             values="value", aggfunc="last"))
        else:
            st.info("No per-epoch training metrics for this run.")
        st.download_button("Export run summaries CSV",
                           pd.DataFrame(summaries).to_csv(index=False),
                           "mlflow_run_summaries.csv", "text/csv")
    else:
        st.info("Run CT training with --mlflow-uri sqlite:///... and load the database.")
else:
    with st.spinner("Running actual reconstruction..."):
        outcome = compute(modality, size, int(seed), projections, acceleration, noise)
    if modality == "ct" and use_unet and checkpoint_path:
        from pathlib import Path
        from medreconlab.ct_inference import infer_ct_checkpoint

        output_root = Path("outputs").resolve()
        selected = Path(checkpoint_path).resolve()
        if not selected.is_relative_to(output_root) or not selected.is_file():
            st.warning("Choose an existing trusted checkpoint within the outputs/ folder.")
        else:
            try:
                from skimage.transform import iradon
                fbp = iradon(outcome["measured"], theta=outcome["settings"]["theta"],
                             filter_name="ramp", circle=True, output_size=size)
                outcome["reconstructions"]["unet"] = infer_ct_checkpoint(
                    selected, fbp, outcome["measured"], outcome["settings"]["theta"],
                    reference=outcome["reference"])
                st.caption("U-Net inference from a locally trained checkpoint. "
                           "Training and current synthetic acquisition must match.")
            except (RuntimeError, ValueError, ImportError, OSError, KeyError) as exc:
                st.error(f"Checkpoint inference unavailable: {exc}")
    rows = [{"method": m, **v["metrics"], "runtime_s": v["runtime_s"],
             "measurement_residual": v["measurement_residual"]}
            for m, v in outcome["reconstructions"].items()]
    metrics = pd.DataFrame(rows)
    if page in ("Reconstruction", "Comparison"):
        images = {"Reference": outcome["reference"], "Degraded input": outcome["degraded"],
                  **{name: r["image"] for name, r in outcome["reconstructions"].items()}}
        cols = st.columns(len(images))
        for col, (name, img) in zip(cols, images.items()):
            col.markdown(f"**{name}**")
            col.image(img, clamp=True, width="stretch")
            if page == "Comparison":
                col.caption("Absolute error")
                col.image(abs(img - outcome["reference"]), clamp=True, width="stretch")
    if page == "Metrics":
        st.dataframe(metrics, use_container_width=True)
        st.bar_chart(metrics.set_index("method")[["psnr", "ssim"]])
        st.bar_chart(metrics.set_index("method")[["runtime_s", "measurement_residual"]])
    if page == "Reports":
        st.dataframe(metrics)
        st.download_button("Export metrics CSV", metrics.to_csv(index=False),
                           "metrics.csv", "text/csv")
        st.download_button("Export metrics JSON", metrics.to_json(orient="records"),
                           "metrics.json", "application/json")
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(outcome["reference"], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Reference")
        first = next(iter(outcome["reconstructions"].values()))["image"]
        axes[1].imshow(first, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Reconstruction")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=180)
        plt.close(fig)
        st.download_button("Export comparison PNG", output.getvalue(),
                           "comparison.png", "image/png")

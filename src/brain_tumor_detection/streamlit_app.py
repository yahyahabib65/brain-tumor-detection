from brain_tumor_detection.config import CSV_PATH, FEATURE_COLUMNS
from brain_tumor_detection.predict import (
    load_mri_model,
    load_tabular_artifacts,
    predict_from_features,
    predict_from_mri,
)
import pandas as pd
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon=":material/neurology:",
    layout="wide",
)


@st.cache_resource
def get_tabular_resources():
  return load_tabular_artifacts()


@st.cache_resource
def get_mri_resource():
  return load_mri_model()


@st.cache_data
def get_feature_defaults() -> dict[str, float]:
  """Loads median feature values from Brain Tumor.csv if available."""
  if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
    df = pd.read_csv(CSV_PATH)
    return {col: float(df[col].median()) for col in FEATURE_COLUMNS}
  return {col: 0.0 for col in FEATURE_COLUMNS}


def render_prediction_banner(has_tumor: bool, prob: float) -> None:
  """Displays the diagnosis result and confidence metric."""
  confidence = prob if has_tumor else (1.0 - prob)
  if has_tumor:
    st.error(f"### 🚨 Diagnosis: **Tumor Detected**")
  else:
    st.success(f"### ✅ Diagnosis: **No Tumor Detected**")

  col1, col2 = st.columns(2)
  col1.metric("Prediction Confidence", f"{confidence:.2%}")
  col2.metric("Raw Tumor Probability", f"{prob:.2%}")
  st.progress(float(prob))


def main() -> None:
  st.title(":material/neurology: Brain Tumor Detection System")
  st.markdown(
      "Detect brain tumors using either **raw MRI scan images (CNN)** "
      "or **extracted radiomic feature values (Tabular MLP)**."
  )

  tab_mri, tab_features = st.tabs(
      [":material/image: Predict from MRI Scan", ":material/edit: Predict from Feature Values"]
  )

  # --- TAB 1: MRI SCAN UPLOAD ---
  with tab_mri:
    st.subheader("Upload Brain MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI image (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
      image = Image.open(uploaded_file)
      col_img, col_res = st.columns([1, 1.5])

      with col_img:
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

      with col_res:
        if st.button(
            "🔍 Analyze MRI Scan", type="primary", use_container_width=True
        ):
          try:
            with st.spinner("Running CNN inference..."):
              mri_model = get_mri_resource()
              has_tumor, prob = predict_from_mri(image, model=mri_model)
            render_prediction_banner(has_tumor, prob)
          except FileNotFoundError as e:
            st.warning(str(e))

#   # --- TAB 2: TABULAR FEATURE VALUES ---
#   with tab_features:
#     st.subheader("Enter Extracted MRI Feature Values")
#     defaults = get_feature_defaults()

#     if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
#       df_samples = pd.read_csv(CSV_PATH)
#       with st.expander("🎲 Load a random sample row from dataset"):
#         if st.button("Load Random Sample"):
#           sample_row = df_samples.sample(1).iloc[0]
#           for col in FEATURE_COLUMNS:
#             st.session_state[f"feat_{col}"] = float(sample_row[col])
#           st.info(
#               f"Loaded **{sample_row['Image']}** "
#               f"(Actual Class: **{'Tumor (1)' if sample_row['Class'] == 1 else 'No Tumor (0)'}**)"
#           )

#     feature_inputs: dict[str, float] = {}
#     cols = st.columns(3)
#     for idx, col_name in enumerate(FEATURE_COLUMNS):
#       with cols[idx % 3]:
#         default_val = st.session_state.get(
#             f"feat_{col_name}", defaults[col_name]
#         )
#         feature_inputs[col_name] = st.number_input(
#             label=col_name,
#             value=float(default_val),
#             format="%.5f",
#             key=f"input_{col_name}",
#         )

#     if st.button(
#         "🧮 Predict from Features", type="primary", use_container_width=True
#     ):
#       try:
#         with st.spinner("Running Tabular MLP inference..."):
#           tab_model, scaler = get_tabular_resources()
#           has_tumor, prob = predict_from_features(
#               feature_inputs, model=tab_model, scaler=scaler
#           )
#         render_prediction_banner(has_tumor, prob)
#       except FileNotFoundError as e:
#         st.warning(str(e))


#   # --- TAB 2: TABULAR FEATURE VALUES ---
#   with tab_features:
#     st.subheader("Enter Extracted MRI Feature Values")
#     defaults = get_feature_defaults()

#     # Initialize the DataFrame in session_state once
#     if "features_df" not in st.session_state:
#       st.session_state.features_df = pd.DataFrame({
#           "Feature": FEATURE_COLUMNS,
#           "Value": [float(defaults[col]) for col in FEATURE_COLUMNS],
#       })

#     if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
#       df_samples = pd.read_csv(CSV_PATH)
#       if st.button("Load Random Sample Row", icon=":material/casino:"):
#         sample_row = df_samples.sample(1).iloc[0]
#         st.session_state.features_df = pd.DataFrame({
#             "Feature": FEATURE_COLUMNS,
#             "Value": [float(sample_row[col]) for col in FEATURE_COLUMNS],
#         })
#         st.session_state.sample_info = (
#             f"Loaded **{sample_row['Image']}** "
#             f"(Actual Class: **{'Tumor (1)' if sample_row['Class'] == 1 else 'No Tumor (0)'}**)"
#         )

#       if "sample_info" in st.session_state:
#         st.info(st.session_state.sample_info, icon=":material/info:")

#     # Editable DataFrame: Feature column is locked, rows are fixed, only Value is editable
#     edited_df = st.data_editor(
#         st.session_state.features_df,
#         disabled=["Feature"],
#         num_rows="fixed",
#         hide_index=True,
#         width="stretch",
#         column_config={
#             "Feature": st.column_config.TextColumn(
#                 "Feature Name", width="medium"
#             ),
#             "Value": st.column_config.NumberColumn(
#                 "Value",
#                 format="%.5f",
#                 required=True,
#             ),
#         },
#     )

#     if st.button(
#         "Predict from Features",
#         icon=":material/calculate:",
#         type="primary",
#         width="stretch",
#     ):
#       try:
#         feature_inputs = dict(zip(edited_df["Feature"], edited_df["Value"]))
#         with st.spinner("Running Tabular MLP inference..."):
#           tab_model, scaler = get_tabular_resources()
#           has_tumor, prob = predict_from_features(
#               feature_inputs, model=tab_model, scaler=scaler
#           )
#         render_prediction_banner(has_tumor, prob)
#       except FileNotFoundError as e:
#         st.warning(str(e))

  # --- TAB 2: TABULAR FEATURE VALUES (LEFT: DATAFRAME | RIGHT: REPORT IMAGE) ---
  with tab_features:
    defaults = get_feature_defaults()

    if "features_df" not in st.session_state:
      st.session_state.features_df = pd.DataFrame({
          "Feature": FEATURE_COLUMNS,
          "Value": [float(defaults[col]) for col in FEATURE_COLUMNS],
      })

    col_left, col_right = st.columns([1, 1], gap="large")

    # --- RIGHT SIDE: UPLOAD REPORT IMAGE & EXTRACT ---
    with col_right:
      st.subheader(":material/document_scanner: Extract from Report Image")
      report_file = st.file_uploader(
          "Upload a feature report image (.jpg, .jpeg, .png)",
          type=["jpg", "jpeg", "png"],
          key="report_uploader",
      )

      if report_file is not None:
        report_img = Image.open(report_file)
        st.image(
            report_img,
            caption="Uploaded Report",
            width="stretch",
            alt="Uploaded report image",
        )

        if st.button(
            "Extract Values & Predict",
            icon=":material/auto_awesome:",
            type="primary",
            width="stretch",
        ):
          try:
            with st.spinner("Extracting feature values from report image..."):
              extracted = extract_features_from_report_image(report_img)

            if not extracted:
              st.warning(
                  "Could not detect matching feature names in the image. "
                  "Make sure the report clearly shows labels like 'Mean', 'Variance', etc.",
                  icon=":material/warning:",
              )
            else:
              # Update DataFrame on the left with extracted values (fallback to current value if missing)
              current_map = dict(
                  zip(
                      st.session_state.features_df["Feature"],
                      st.session_state.features_df["Value"],
                  )
              )
              updated_values = [
                  float(extracted.get(col, current_map[col]))
                  for col in FEATURE_COLUMNS
              ]
              st.session_state.features_df = pd.DataFrame({
                  "Feature": FEATURE_COLUMNS,
                  "Value": updated_values,
              })
              st.success(
                  f"Extracted {len(extracted)}/{len(FEATURE_COLUMNS)} features from report!",
                  icon=":material/check_circle:",
              )

              # Run prediction immediately
              feature_inputs = dict(zip(FEATURE_COLUMNS, updated_values))
              tab_model, scaler = get_tabular_resources()
              has_tumor, prob = predict_from_features(
                  feature_inputs, model=tab_model, scaler=scaler
              )
              render_prediction_banner(has_tumor, prob)
          except FileNotFoundError as e:
            st.warning(str(e))

    # --- LEFT SIDE: FIXED-KEY DATAFRAME EDITOR ---
    with col_left:
      st.subheader(":material/table_chart: Feature Values Table")

      if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        df_samples = pd.read_csv(CSV_PATH)
        if st.button(
            "Load Random Sample Row", icon=":material/casino:", width="stretch"
        ):
          sample_row = df_samples.sample(1).iloc[0]
          st.session_state.features_df = pd.DataFrame({
              "Feature": FEATURE_COLUMNS,
              "Value": [float(sample_row[col]) for col in FEATURE_COLUMNS],
          })
          st.session_state.sample_info = (
              f"Loaded **{sample_row['Image']}** "
              f"(Actual Class: **{'Tumor (1)' if sample_row['Class'] == 1 else 'No Tumor (0)'}**)"
          )

        if "sample_info" in st.session_state:
          st.info(st.session_state.sample_info, icon=":material/info:")

      edited_df = st.data_editor(
          st.session_state.features_df,
          disabled=["Feature"],
          num_rows="fixed",
          hide_index=True,
          width="stretch",
          column_config={
              "Feature": st.column_config.TextColumn(
                  "Feature Name", width="medium"
              ),
              "Value": st.column_config.NumberColumn(
                  "Value",
                  format="%.5f",
                  required=True,
              ),
          },
      )

      if st.button(
          "Predict from Table Values",
          icon=":material/calculate:",
          type="primary",
          width="stretch",
      ):
        try:
          feature_inputs = dict(zip(edited_df["Feature"], edited_df["Value"]))
          with st.spinner("Running Tabular MLP inference..."):
            tab_model, scaler = get_tabular_resources()
            has_tumor, prob = predict_from_features(
                feature_inputs, model=tab_model, scaler=scaler
            )
          render_prediction_banner(has_tumor, prob)
        except FileNotFoundError as e:
          st.warning(str(e))

if __name__ == "__main__":
  main()
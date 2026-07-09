import os
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image


st.set_page_config(
    page_title="Machine Predictive Maintenance",
    layout="wide",
    page_icon="🔧"
)


palette = ['#FF5733', '#FFC300', '#DAF7A6', '#C70039', '#900C3F']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)


TYPE_MAPPING = {'L': 0, 'M': 1, 'H': 2}
TYPE_LABEL_FROM_CODE = {0: 'L', 1: 'M', 2: 'H'}
DEFAULT_FEATURES = [
    'Type',
    'Air_temperature_K',
    'Process_temperature_K',
    'Rotational_speed_rpm',
    'Torque_Nm',
    'Tool_wear_min',
    'Temp_diff_K',
    'Power_proxy',
    'Wear_speed_ratio',
    'Torque_wear_interaction',
    'Temp_torque_interaction',
    'Wear_temp_interaction'
]


@st.cache_data
def load_data():
    data_path = 'data/ai4i2020.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    return pd.read_csv(data_path)


@st.cache_resource
def load_model():
    model_path = 'models/xgb_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


@st.cache_resource
def load_model_meta():
    meta_path = 'models/model_meta.joblib'
    if os.path.exists(meta_path):
        return joblib.load(meta_path)
    return {
        "features": DEFAULT_FEATURES,
        "best_threshold": 0.50,
        "best_params": {},
        "cv_average_precision": None,
        "test_average_precision": None,
        "test_roc_auc": None
    }


@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)


def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    df.columns = df.columns.str.replace("[", "", regex=False)
    df.columns = df.columns.str.replace("]", "", regex=False)
    df.columns = df.columns.str.replace("/", "_", regex=False)
    df = df.drop(['UDI', 'Product_ID'], axis=1)
    df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

    df['Temp_diff_K'] = df['Process_temperature_K'] - df['Air_temperature_K']
    df['Power_proxy'] = df['Torque_Nm'] * df['Rotational_speed_rpm']
    df['Wear_speed_ratio'] = df['Tool_wear_min'] / (df['Rotational_speed_rpm'] + 1)
    df['Torque_wear_interaction'] = df['Torque_Nm'] * df['Tool_wear_min']
    df['Temp_torque_interaction'] = df['Temp_diff_K'] * df['Torque_Nm']
    df['Wear_temp_interaction'] = df['Tool_wear_min'] * df['Temp_diff_K']
    return df


def build_feature_row(selected_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear):
    row = pd.DataFrame([{
        'Type': TYPE_MAPPING[selected_type],
        'Air_temperature_K': air_temperature,
        'Process_temperature_K': process_temperature,
        'Rotational_speed_rpm': rotational_speed,
        'Torque_Nm': torque,
        'Tool_wear_min': tool_wear
    }])

    row['Temp_diff_K'] = row['Process_temperature_K'] - row['Air_temperature_K']
    row['Power_proxy'] = row['Torque_Nm'] * row['Rotational_speed_rpm']
    row['Wear_speed_ratio'] = row['Tool_wear_min'] / (row['Rotational_speed_rpm'] + 1)
    row['Torque_wear_interaction'] = row['Torque_Nm'] * row['Tool_wear_min']
    row['Temp_torque_interaction'] = row['Temp_diff_K'] * row['Torque_Nm']
    row['Wear_temp_interaction'] = row['Tool_wear_min'] * row['Temp_diff_K']
    return row


try:
    df_raw = load_data()
    df = preprocess(df_raw)
    model = load_model()
    model_meta = load_model_meta()
except Exception as e:
    st.error(f"App could not load required files: {e}")
    st.stop()


MAIN_FEATURES = model_meta.get("features", DEFAULT_FEATURES)
DEFAULT_THRESHOLD = float(model_meta.get("best_threshold", 0.50))

missing_features = [col for col in MAIN_FEATURES if col not in df.columns]
if missing_features:
    st.error(f"Missing required model features: {missing_features}")
    st.stop()

X = df[MAIN_FEATURES]


if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = pd.DataFrame(columns=[
        'Type',
        'Air_temperature_K',
        'Process_temperature_K',
        'Rotational_speed_rpm',
        'Torque_Nm',
        'Tool_wear_min',
        'Threshold',
        'Prediction',
        'Probability'
    ])


if 'telemetry' not in st.session_state:
    rng = np.random.default_rng(42)
    t = np.arange(60)
    st.session_state['telemetry'] = pd.DataFrame({
        'time': t,
        'Air_temperature_K': 298 + np.sin(t / 6) * 0.6 + rng.normal(0, 0.08, 60),
        'Process_temperature_K': 308 + np.cos(t / 7) * 0.5 + rng.normal(0, 0.08, 60),
        'Rotational_speed_rpm': 2000 + np.sin(t / 8) * 120 + rng.normal(0, 15, 60)
    })


img_path = 'factory_image.jpg'
if os.path.exists(img_path):
    img = Image.open(img_path)

    w, h = img.size
    left = int(w * 0.05)
    top = int(h * 0.12)
    right = int(w * 0.95)
    bottom = int(h * 0.38)

    img_cropped = img.crop((left, top, right, bottom))
    st.image(img_cropped, use_container_width=True)


st.title('Predictive Maintenance Dashboard')
st.markdown('This dashboard predicts machine failure using the tuned XGBoost model and the saved threshold from training.')


top1, top2, top3 = st.columns(3)
with top1:
    st.metric("Default model threshold", f"{DEFAULT_THRESHOLD:.2f}")
with top2:
    ap = model_meta.get("test_average_precision")
    st.metric("Test PR-AUC", f"{ap:.4f}" if ap is not None else "N/A")
with top3:
    roc = model_meta.get("test_roc_auc")
    st.metric("Test ROC-AUC", f"{roc:.4f}" if roc is not None else "N/A")


tab1, tab2, tab3 = st.tabs([' Prediction', ' Data Exploration', ' SHAP & Telemetry'])


with tab1:
    st.markdown('## Predictive Maintenance Prediction')
    col1, col2 = st.columns([0.34, 0.66])

    with col1:
        st.header('Input Features')

        selected_type = st.selectbox('Type', list(TYPE_MAPPING.keys()), key='pred_type')
        air_temperature = st.slider('Air Temp [K]', 290.0, 310.0, 299.5, 0.1, key='pred_air_temp')
        process_temperature = st.slider('Proc Temp [K]', 300.0, 320.0, 308.7, 0.1, key='pred_proc_temp')
        rotational_speed = st.slider('Speed [rpm]', 1000, 3000, 2134, key='pred_speed')
        torque = st.slider('Torque [Nm]', 3.5, 77.0, 45.3, 0.1, key='pred_torque')
        tool_wear = st.slider('Tool Wear [min]', 0, 250, 127, key='pred_tool_wear')

        threshold = st.slider(
            'Failure threshold',
            0.0,
            1.0,
            float(DEFAULT_THRESHOLD),
            0.01,
            key='pred_threshold'
        )
        st.caption(f'Failure is shown when probability >= {threshold:.2f}. Tuned default = {DEFAULT_THRESHOLD:.2f}')

        predict_clicked = st.button(
            'Predict Failure',
            use_container_width=True,
            key='predict_button'
        )

        if predict_clicked:
            x_input = build_feature_row(
                selected_type,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear
            )[MAIN_FEATURES]

            prob = float(model.predict_proba(x_input)[0][1])
            pred = 1 if prob >= threshold else 0

            if pred == 1:
                st.error(f'Prediction: Failure | Probability: {prob:.3f}')
                pred_label = 'Failure'
            else:
                st.success(f'Prediction: No Failure | Probability: {prob:.3f}')
                pred_label = 'No Failure'

            st.progress(min(max(prob, 0.0), 1.0))

            risk_band = (
                "High risk" if prob >= max(threshold, 0.70)
                else "Moderate risk" if prob >= threshold
                else "Low risk"
            )
            st.info(f'Risk band: {risk_band}')

            if risk_band == "High risk":
                st.warning(
                    "Recommended action: Perform immediate inspection, check tool wear, torque, and temperature conditions, and schedule urgent maintenance."
                )
            elif risk_band == "Moderate risk":
                st.info(
                    "Recommended action: Schedule preventive maintenance, review recent sensor trends, and monitor the machine more frequently."
                )
            else:
                st.success(
                    "Recommended action: No immediate maintenance required; continue routine monitoring and standard inspection."
                )

            new_entry = pd.DataFrame({
                'Type': [selected_type],
                'Air_temperature_K': [air_temperature],
                'Process_temperature_K': [process_temperature],
                'Rotational_speed_rpm': [rotational_speed],
                'Torque_Nm': [torque],
                'Tool_wear_min': [tool_wear],
                'Threshold': [round(threshold, 2)],
                'Prediction': [pred_label],
                'Probability': [round(prob, 3)]
            })

            st.session_state['prediction_history'] = pd.concat(
                [st.session_state['prediction_history'], new_entry],
                ignore_index=True
            )

    with col2:
        st.subheader('Prediction History')
        st.dataframe(st.session_state['prediction_history'], use_container_width=True)

        st.subheader('Current Input')
        st.json({
            'Type': selected_type,
            'Air Temperature [K]': air_temperature,
            'Process Temperature [K]': process_temperature,
            'Rotational Speed [rpm]': rotational_speed,
            'Torque [Nm]': torque,
            'Tool Wear [min]': tool_wear,
            'Applied Threshold': round(threshold, 2)
        })

        if len(st.session_state['prediction_history']) > 0:
            hist_fig = px.line(
                st.session_state['prediction_history'].reset_index(),
                x='index',
                y='Probability',
                color='Prediction',
                markers=True,
                title='Prediction probability over time'
            )
            hist_fig.update_layout(
                xaxis_title='Prediction number',
                yaxis_title='Failure probability',
                margin=dict(l=20, r=20, t=50, b=20),
                height=360
            )
            st.plotly_chart(hist_fig, use_container_width=True)


with tab2:
    st.markdown('## Data Exploration')

    c1, c2 = st.columns(2)
    with c1:
        type_f = st.selectbox('Filter Type', ['All'] + list(TYPE_MAPPING.keys()), key='filter_type')
        air_rng = st.slider('Air Range [K]', 290.0, 310.0, (290.0, 310.0), 0.1, key='filter_air')
        proc_rng = st.slider('Proc Range [K]', 300.0, 320.0, (300.0, 320.0), 0.1, key='filter_proc')

    with c2:
        speed_rng = st.slider('Speed Range [rpm]', 1000, 3000, (1000, 3000), key='filter_speed')
        torque_rng = st.slider('Torque Range [Nm]', 3.5, 77.0, (3.5, 77.0), 0.1, key='filter_torque')
        wear_rng = st.slider('Wear Range [min]', 0, 250, (0, 250), key='filter_wear')

    filtered = df.copy()
    if type_f != 'All':
        filtered = filtered[filtered['Type'] == TYPE_MAPPING[type_f]]

    filtered = filtered[
        (filtered['Air_temperature_K'].between(*air_rng)) &
        (filtered['Process_temperature_K'].between(*proc_rng)) &
        (filtered['Rotational_speed_rpm'].between(*speed_rng)) &
        (filtered['Torque_Nm'].between(*torque_rng)) &
        (filtered['Tool_wear_min'].between(*wear_rng))
    ]

    st.write('### Filtered sample')
    display_df = filtered.copy()
    if 'Type' in display_df.columns:
        display_df['Type'] = display_df['Type'].map(TYPE_LABEL_FROM_CODE)
    st.dataframe(display_df.head(20), use_container_width=True)

    a1, a2 = st.columns(2)

    with a1:
        st.write('### Machine Failure Count')
        counts = filtered['Machine_failure'].value_counts().reset_index()
        counts.columns = ['Machine_failure', 'count']
        counts['Machine_failure'] = counts['Machine_failure'].map({0: 'No Failure', 1: 'Failure'})
        fig = px.bar(
            counts,
            x='Machine_failure',
            y='count',
            labels={'Machine_failure': 'Machine status', 'count': 'Count'},
            color='Machine_failure',
            height=360
        )
        st.plotly_chart(fig, use_container_width=True)

    with a2:
        st.write('### Correlation Heatmap')
        corr = filtered.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, cmap='Greens', ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.write('### Feature Distributions')
    b1, b2, b3 = st.columns(3)

    with b1:
        fig = px.box(
            filtered,
            x='Machine_failure',
            y='Air_temperature_K',
            labels={'Machine_failure': 'Failure', 'Air_temperature_K': 'Air Temp'},
            height=340
        )
        st.plotly_chart(fig, use_container_width=True)

    with b2:
        fig = px.box(
            filtered,
            x='Machine_failure',
            y='Process_temperature_K',
            labels={'Machine_failure': 'Failure', 'Process_temperature_K': 'Proc Temp'},
            height=340
        )
        st.plotly_chart(fig, use_container_width=True)

    with b3:
        fig = px.box(
            filtered,
            x='Machine_failure',
            y='Tool_wear_min',
            labels={'Machine_failure': 'Failure', 'Tool_wear_min': 'Tool Wear'},
            height=340
        )
        st.plotly_chart(fig, use_container_width=True)

    d1, d2 = st.columns(2)

    with d1:
        fig = px.histogram(
            filtered,
            x='Rotational_speed_rpm',
            color='Machine_failure',
            barmode='overlay',
            nbins=30,
            height=360
        )
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        fig = px.histogram(
            filtered,
            x='Torque_Nm',
            color='Machine_failure',
            barmode='overlay',
            nbins=30,
            height=360
        )
        st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.markdown('## SHAP & Telemetry')

    x_sample = X.sample(min(200, len(X)), random_state=42)

    try:
        explainer = load_shap_explainer(model)
        shap_values_raw = explainer.shap_values(x_sample, check_additivity=False)

        if isinstance(shap_values_raw, list):
            shap_values_raw = shap_values_raw[1]

        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

        shap_values = shap.Explanation(
            values=shap_values_raw,
            base_values=np.repeat(expected_value, len(x_sample)),
            data=x_sample.values,
            feature_names=x_sample.columns.tolist()
        )

        s1, s2 = st.columns(2, vertical_alignment="top")

        with s1:
            st.markdown('### SHAP Summary')
            fig_summary = plt.figure(figsize=(8, 5.2))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig_summary, use_container_width=True)
            plt.close(fig_summary)

        with s2:
            header_left, header_right = st.columns([0.62, 0.38], vertical_alignment="bottom")
            with header_left:
                st.markdown('### SHAP for one sample')
            with header_right:
                idx = st.number_input(
                    'Sample index',
                    min_value=0,
                    max_value=len(x_sample) - 1,
                    value=0,
                    step=1,
                    label_visibility="collapsed",
                    key='shap_sample_index'
                )

            fig_waterfall = plt.figure(figsize=(8, 5.2))
            shap.plots.waterfall(shap_values[int(idx)], show=False)
            st.pyplot(fig_waterfall, use_container_width=True)
            plt.close(fig_waterfall)

            with st.expander("Generate SHAP text explanation"):
                sample_idx = int(idx)
                sample_values = shap_values.values[sample_idx]
                sample_features = x_sample.iloc[sample_idx]

                contrib_df = pd.DataFrame({
                    "feature": x_sample.columns,
                    "value": sample_features.values,
                    "shap_value": sample_values
                })

                pretty_names = {
                    "Air_temperature_K": "Air temperature",
                    "Process_temperature_K": "Process temperature",
                    "Rotational_speed_rpm": "Rotational speed",
                    "Torque_Nm": "Torque",
                    "Tool_wear_min": "Tool wear",
                    "Type": "Machine type",
                    "Temp_diff_K": "Temperature difference",
                    "Power_proxy": "Power proxy",
                    "Wear_speed_ratio": "Wear-speed ratio",
                    "Torque_wear_interaction": "Torque-wear interaction",
                    "Temp_torque_interaction": "Temperature-torque interaction",
                    "Wear_temp_interaction": "Wear-temperature interaction"
                }

                contrib_df["feature_label"] = contrib_df["feature"].map(pretty_names).fillna(contrib_df["feature"])
                top_positive = contrib_df.sort_values("shap_value", ascending=False).head(3)
                top_negative = contrib_df.sort_values("shap_value", ascending=True).head(3)

                positive_text = ", ".join([
                    f"{row.feature_label} ({row.value:.2f}) increased failure risk"
                    for row in top_positive.itertuples()
                ])
                negative_text = ", ".join([
                    f"{row.feature_label} ({row.value:.2f}) reduced failure risk"
                    for row in top_negative.itertuples()
                ])

                st.write(f"Top factors increasing risk: {positive_text}.")
                st.write(f"Top factors reducing risk: {negative_text}.")

    except Exception as e:
        st.warning(f'SHAP visualization could not be generated: {e}')

    st.write('### Live Telemetry')

    if st.button('Simulate New Telemetry Point', use_container_width=True, key='telemetry_button'):
        last = st.session_state['telemetry'].iloc[-1]
        new_row = pd.DataFrame({
            'time': [int(last['time']) + 1],
            'Air_temperature_K': [float(last['Air_temperature_K']) + np.random.normal(0, 0.12)],
            'Process_temperature_K': [float(last['Process_temperature_K']) + np.random.normal(0, 0.12)],
            'Rotational_speed_rpm': [float(last['Rotational_speed_rpm']) + np.random.normal(0, 18)]
        })
        st.session_state['telemetry'] = pd.concat(
            [st.session_state['telemetry'], new_row],
            ignore_index=True
        ).tail(60).reset_index(drop=True)

    tel = st.session_state['telemetry']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tel['time'], y=tel['Air_temperature_K'], mode='lines', name='Air Temp'))
    fig.add_trace(go.Scatter(x=tel['time'], y=tel['Process_temperature_K'], mode='lines', name='Proc Temp'))
    fig.add_trace(go.Scatter(x=tel['time'], y=tel['Rotational_speed_rpm'], mode='lines', name='Speed'))
    fig.update_layout(
        title='Telemetry trends (last 60 points)',
        xaxis_title='Time',
        yaxis_title='Value',
        margin=dict(l=20, r=20, t=50, b=20),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)
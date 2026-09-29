from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


CLASS_NAMES = [
    "T-shirt / top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

st.set_page_config(
    page_title="Fashion MNIST | FCNN Lab",
    page_icon="◩",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
        --ink: #182521;
        --muted: #66736c;
        --paper: #f5f5ef;
        --line: #d9ded5;
        --green: #c7e76b;
        --coral: #e76f51;
    }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #e9eee4; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] * { color: var(--ink); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
    .eyebrow { font: 500 0.72rem 'DM Mono', monospace; letter-spacing: 0; text-transform: uppercase; color: #52705f; }
    .page-title { font-size: 2.65rem; font-weight: 800; line-height: 1.05; margin: 0.35rem 0 0.4rem; }
    .subtle { color: var(--muted); font-size: 0.94rem; }
    .rule { border-top: 1px solid var(--line); margin: 1.35rem 0 1.6rem; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 0.9rem 1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.55rem; }
    div[data-testid="stButton"] button[kind="primary"] { background: var(--ink); border-color: var(--ink); color: white; }
    div[data-testid="stButton"] button[kind="primary"]:hover { background: #315344; border-color: #315344; }
    [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] * { color: white; }
    .section-label { font: 500 0.72rem 'DM Mono', monospace; text-transform: uppercase; color: #52705f; margin-bottom: -0.3rem; }
    @media (max-width: 760px) {
        .page-title { font-size: 1.85rem; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset():
    (images_train, labels_train), (images_test, labels_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    images_train = images_train.astype("float32") / 255.0
    images_test = images_test.astype("float32") / 255.0
    labels_train_one_hot = tf.keras.utils.to_categorical(labels_train, 10)
    labels_test_one_hot = tf.keras.utils.to_categorical(labels_test, 10)
    return (
        images_train,
        labels_train,
        labels_train_one_hot,
        images_test,
        labels_test,
        labels_test_one_hot,
    )


def build_model(learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


with st.sidebar:
    st.markdown('<p class="eyebrow">Training controls</p>', unsafe_allow_html=True)
    st.markdown("### Fully connected network")
    epochs = st.slider("Epochs", min_value=1, max_value=30, value=8)
    batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256], value=128)
    learning_rate = st.select_slider(
        "SGD learning rate",
        options=[0.001, 0.003, 0.01, 0.03, 0.1],
        value=0.01,
        format_func=lambda value: f"{value:g}",
    )
    st.caption("784 inputs · 2 hidden layers · 10 classes · dropout regularization")
    train_clicked = st.button("Train model", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown('<p class="eyebrow">Objective</p>', unsafe_allow_html=True)
    st.caption("Categorical cross-entropy · Stochastic gradient descent · 20% validation split")

st.markdown('<p class="eyebrow">Computer vision / classification</p>', unsafe_allow_html=True)
st.markdown('<h1 class="page-title">Fashion MNIST <span style="color:#e76f51">/</span> FCNN lab</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtle">Explore the dataset, train a dense neural network, and inspect how well it recognizes unseen apparel.</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

with st.spinner("Loading Fashion-MNIST dataset…"):
    x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot = load_dataset()

if train_clicked:
    model = build_model(learning_rate)
    progress = st.progress(0, text="Preparing training run…")
    status = st.empty()

    class TrainingProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            progress.progress(
                (epoch + 1) / epochs,
                text=f"Epoch {epoch + 1}/{epochs} · validation accuracy {logs.get('val_accuracy', 0):.1%}",
            )
            status.caption(
                f"Training accuracy {logs.get('accuracy', 0):.1%}  ·  "
                f"Validation loss {logs.get('val_loss', 0):.3f}"
            )

    with st.spinner("Training the FCNN…"):
        history = model.fit(
            x_train,
            y_train_one_hot,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.20,
            callbacks=[TrainingProgress()],
            verbose=0,
        )
        test_loss, test_accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
        predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (y_test, predictions), 1)

    st.session_state["model"] = model
    st.session_state["history"] = history.history
    st.session_state["test_loss"] = float(test_loss)
    st.session_state["test_accuracy"] = float(test_accuracy)
    st.session_state["confusion"] = confusion
    st.session_state["run_config"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    progress.empty()
    status.success("Training complete. The test set was evaluated separately from training and validation.")

distribution = pd.DataFrame(
    {
        "Fashion item": CLASS_NAMES,
        "Training images": np.bincount(y_train, minlength=10),
        "Test images": np.bincount(y_test, minlength=10),
    }
)
model_ready = "history" in st.session_state

metric_columns = st.columns(4)
metric_columns[0].metric("Training images", f"{len(x_train):,}")
metric_columns[1].metric("Test images", f"{len(x_test):,}")
metric_columns[2].metric("Image dimensions", "28 × 28")
metric_columns[3].metric("Model status", "Trained" if model_ready else "Not trained")

overview_tab, training_tab, predictions_tab = st.tabs(
    ["Dataset", "Training & evaluation", "Predictions"]
)

with overview_tab:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown('<p class="section-label">01 / Class balance</p>', unsafe_allow_html=True)
        st.subheader("Images by category")
        st.bar_chart(distribution.set_index("Fashion item"), height=330)
        st.caption("The official split contains 60,000 training and 10,000 test images.")
    with right:
        st.markdown('<p class="section-label">02 / Training examples</p>', unsafe_allow_html=True)
        st.subheader("A sample from each class")
        sample_columns = st.columns(5)
        for class_id, column in enumerate(sample_columns * 2):
            image_index = int(np.flatnonzero(y_train == class_id)[0])
            column.image(
                x_train[image_index],
                caption=CLASS_NAMES[class_id],
                clamp=True,
                use_container_width=True,
            )
        st.caption("Pixel values are scaled from [0, 255] to [0, 1]; labels are one-hot encoded for training.")

with training_tab:
    if not model_ready:
        st.info("Choose the training settings in the sidebar and select **Train model** to generate learning curves and test results.")
    else:
        history_frame = pd.DataFrame(st.session_state["history"])
        history_frame.insert(0, "Epoch", np.arange(1, len(history_frame) + 1))
        config = st.session_state["run_config"]
        st.markdown(
            f'<p class="section-label">Latest run / {config["epochs"]} epochs / batch {config["batch_size"]} / SGD {config["learning_rate"]:g}</p>',
            unsafe_allow_html=True,
        )
        result_columns = st.columns(3)
        result_columns[0].metric("Test accuracy", f"{st.session_state['test_accuracy']:.1%}")
        result_columns[1].metric("Test loss", f"{st.session_state['test_loss']:.3f}")
        result_columns[2].metric("Val. accuracy", f"{history_frame['val_accuracy'].iloc[-1]:.1%}")
        loss_column, accuracy_column = st.columns(2, gap="large")
        with loss_column:
            st.markdown('<p class="section-label">Optimization</p>', unsafe_allow_html=True)
            st.subheader("Cross-entropy loss")
            st.line_chart(history_frame, x="Epoch", y=["loss", "val_loss"], height=280)
        with accuracy_column:
            st.markdown('<p class="section-label">Generalization</p>', unsafe_allow_html=True)
            st.subheader("Classification accuracy")
            accuracy_frame = history_frame.copy()
            accuracy_frame[["accuracy", "val_accuracy"]] *= 100
            st.line_chart(
                accuracy_frame,
                x="Epoch",
                y=["accuracy", "val_accuracy"],
                height=280,
            )
            st.caption("Training and validation curves help reveal underfitting or a widening overfit gap.")
        st.markdown('<p class="section-label">Test set / actual rows · predicted columns</p>', unsafe_allow_html=True)
        st.subheader("Confusion matrix")
        confusion_frame = pd.DataFrame(st.session_state["confusion"], index=CLASS_NAMES, columns=CLASS_NAMES)
        st.dataframe(confusion_frame, use_container_width=True)

with predictions_tab:
    if not model_ready:
        st.info("Train the model first to explore predictions on held-out test images.")
    else:
        image_index = st.slider("Test image index", min_value=0, max_value=len(x_test) - 1, value=0)
        prediction = st.session_state["model"].predict(x_test[image_index : image_index + 1], verbose=0)[0]
        predicted_class = int(np.argmax(prediction))
        actual_class = int(y_test[image_index])
        image_column, detail_column = st.columns([0.7, 1.3], gap="large")
        with image_column:
            st.image(
                x_test[image_index],
                caption=f"Test sample {image_index:,}",
                clamp=True,
                width=180,
            )
        with detail_column:
            st.markdown('<p class="section-label">Held-out sample</p>', unsafe_allow_html=True)
            st.subheader(f"Prediction: {CLASS_NAMES[predicted_class]}")
            st.metric("Actual label", CLASS_NAMES[actual_class])
            st.metric("Confidence", f"{prediction[predicted_class]:.1%}")
            confidence_frame = pd.DataFrame(
                {"Confidence": prediction}, index=CLASS_NAMES
            ).sort_values("Confidence", ascending=False)
            st.bar_chart(confidence_frame, height=300)
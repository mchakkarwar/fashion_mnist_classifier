# Fashion MNIST FCNN Lab

A Streamlit app for exploring Fashion-MNIST and training a fully connected neural network to classify its ten apparel categories.

## Live app

[Open Fashion MNIST FCNN Lab](https://fashion-mnist-classification.streamlit.app/)

## Run locally

Use Python 3.9 or newer. From the project directory, install the dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The Keras dataset loader downloads Fashion-MNIST the first time the app runs. Streamlit caches the dataset for subsequent reruns.

## Workflow

- Inspect training and test class counts and example images.
- Normalize image pixels to `[0, 1]` and one-hot encode the labels.
- Configure epochs, batch size, and SGD learning rate in the sidebar.
- Train a dense network with two ReLU hidden layers and dropout regularization, using categorical cross-entropy and a 20% validation split.
- Compare training and validation loss/accuracy, evaluate on the separate test split, inspect a confusion matrix, and explore predictions on individual test images.

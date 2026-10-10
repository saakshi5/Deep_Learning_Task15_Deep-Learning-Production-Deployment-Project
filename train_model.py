
import os
import json
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("artifacts", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Small illustrative dataset. Expand with reviewed, real examples
# before considering use in a real IT support environment.
DATA = {
    "Network": [
        "wifi is disconnected and internet is not working",
        "cannot connect to the office network",
        "vpn connection keeps dropping",
        "network is very slow today",
        "router is not responding",
        "unable to access websites on company network",
        "ethernet connection has no internet",
        "dns lookup failed on internal portal",
        "video calls freeze due to network issues",
        "network cable is unplugged",
        "cannot reach the server over network",
        "wireless signal keeps disconnecting",
        "internet connection times out",
        "vpn fails to connect",
        "network latency is high",
        "shared network drive is unreachable",
        "packet loss during remote meeting",
        "office wifi has no internet access",
        "ip address conflict on my laptop",
        "proxy connection is failing",
    ],
    "Hardware": [
        "laptop will not power on",
        "keyboard keys are not working",
        "monitor screen is flickering",
        "printer has a paper jam",
        "computer fan is making loud noise",
        "mouse is not responding",
        "laptop battery is not charging",
        "external monitor is not detected",
        "desktop keeps overheating",
        "headset microphone is broken",
        "printer is not printing pages",
        "laptop screen is cracked",
        "usb port does not detect my device",
        "computer randomly shuts down",
        "webcam is not detected",
        "docking station is not working",
        "hard drive makes clicking sounds",
        "keyboard is typing wrong keys",
        "monitor has no display",
        "printer reports a hardware error",
    ],
    "Software": [
        "application crashes when I open it",
        "software installation failed",
        "program freezes after login",
        "spreadsheet application will not start",
        "browser keeps closing unexpectedly",
        "cannot update desktop application",
        "application shows an unexpected error",
        "software is running very slowly",
        "email client crashes on startup",
        "database application fails to launch",
        "app update is stuck",
        "program displays a blank screen",
        "document editor cannot save files",
        "application is not responding",
        "plugin causes browser to crash",
        "software license activation error",
        "application cannot open a file",
        "failed to install the latest patch",
        "screen sharing application crashes",
        "program has corrupted settings",
    ],
    "Security": [
        "suspicious email asks me to click a link",
        "possible phishing email received",
        "antivirus detected malware on my computer",
        "I clicked a suspicious attachment",
        "account may have been compromised",
        "unusual login alert from another country",
        "possible ransomware warning on workstation",
        "browser redirected to a suspicious website",
        "unexpected password reset email received",
        "security software is disabled unexpectedly",
        "someone is asking for my verification code",
        "suspicious file appeared on desktop",
        "unauthorized access alert received",
        "email attachment looks malicious",
        "computer may be infected with a virus",
        "unknown login activity on my account",
        "possible data leak in shared folder",
        "phishing page requested my credentials",
        "unexpected multi factor authentication prompts",
        "report a suspected security incident",
    ],
    "Access": [
        "please reset my account password",
        "locked out of my company account",
        "request access to the finance folder",
        "cannot sign in to employee portal",
        "permission denied when opening shared documents",
        "new employee needs an account",
        "my account is locked after failed logins",
        "need access to project management tool",
        "multi factor authentication setup help",
        "forgot my login password",
        "request administrator permission for software",
        "access denied to internal dashboard",
        "employee account creation request",
        "need to change my account permissions",
        "cannot log in after password change",
        "request access to a shared mailbox",
        "my user account has been disabled",
        "need VPN account access approval",
        "unable to authenticate to the portal",
        "please unlock my user account",
    ],
}

# Convert category dictionary to a DataFrame.
df = pd.DataFrame([
    {"text": text, "label": label}
    for label, examples in DATA.items()
    for text in examples
])

df.to_csv("artifacts/demo_incidents.csv", index=False)

# Separate training and testing examples.
X_train_text, X_test_text, y_train_text, y_test_text = (
    train_test_split(
        df["text"].values,
        df["label"].values,
        test_size=0.25,
        random_state=SEED,
        stratify=df["label"].values,
    )
)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train_text)
y_test = encoder.transform(y_test_text)

MAX_WORDS = 3000
MAX_LEN = 30

# Fit the tokenizer ONLY on training text.
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>",
)
tokenizer.fit_on_texts(X_train_text)

X_train = pad_sequences(
    tokenizer.texts_to_sequences(X_train_text),
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)
X_test = pad_sequences(
    tokenizer.texts_to_sequences(X_test_text),
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)

# Define the neural network.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LEN,)),
    tf.keras.layers.Embedding(
        input_dim=MAX_WORDS,
        output_dim=64,
        mask_zero=True,
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, dropout=0.2)
    ),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(
        len(encoder.classes_),
        activation="softmax",
    ),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Keep the best model based on validation loss.
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "artifacts/incident_model.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
]

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=8,
    callbacks=callbacks,
    verbose=1,
)

training_seconds = round(time.time() - start_time, 2)

# Load the best checkpoint and evaluate on held-out test data.
model = tf.keras.models.load_model(
    "artifacts/incident_model.keras"
)

probabilities = model.predict(X_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)

accuracy = float(accuracy_score(y_test, predictions))

report = classification_report(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0,
)

# Save preprocessing artifacts.
with open("artifacts/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

with open("artifacts/labels.json", "w", encoding="utf-8") as f:
    json.dump(encoder.classes_.tolist(), f, indent=2)

# Save actual metrics from this run.
metrics = {
    "test_accuracy": accuracy,
    "test_samples": int(len(y_test)),
    "training_seconds": training_seconds,
    "epochs_completed": len(history.history["loss"]),
    "categories": encoder.classes_.tolist(),
    "classification_report": report,
    "warning": (
        "Small illustrative dataset. Results do not establish "
        "production-level performance."
    ),
}

with open("reports/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

pd.DataFrame(history.history).to_csv(
    "reports/training_history.csv",
    index=False,
)

# Create a confusion matrix.
cm = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(encoder.classes_)),
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(cm)
ax.set_xticks(
    range(len(encoder.classes_)),
    encoder.classes_,
    rotation=35,
    ha="right",
)
ax.set_yticks(range(len(encoder.classes_)), encoder.classes_)
ax.set_xlabel("Predicted category")
ax.set_ylabel("Actual category")
ax.set_title("IT Incident Classification Confusion Matrix")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

fig.tight_layout()
fig.savefig("reports/confusion_matrix.png", dpi=160)
plt.close(fig)

print("\nTraining completed.")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Training duration: {training_seconds} seconds")
print(f"Epochs completed: {len(history.history['loss'])}")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=encoder.classes_,
    zero_division=0,
))
print("\nSaved model, tokenizer, labels, metrics and confusion matrix.")

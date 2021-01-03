from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from metrics import custom_metrics
from data import *

min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes

# Hyperparameters for simple baseline DNN
lr = 0.001  # learn rate
dropout = 0.2
hidden_units = 256

X_train, y_train, X_test, y_test = load_credit()
X_train, y_train, X_val, y_val, X_test, y_test = create_data(X_train, y_train, X_test, y_test, min_class, maj_class, imbalance=False)

model = Sequential([
    Dense(hidden_units, activation="relu", input_shape=(X_train.shape[-1],)),
    Dropout(dropout),
    Dense(hidden_units, activation="relu"),
    Dropout(dropout),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(lr), loss="binary_crossentropy")
model.fit(X_train, y_train, epochs=100, batch_size=1024, validation_data=(X_val, y_val), verbose=1)

# Predictions
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

results = custom_metrics(y_test, y_pred_test > 0.5)
print("Baseline: ", results)

# DTA: Baseline NN model with Threshold-Adjustment
# Validate for every threshold
thresholds = np.arange(0, 1, 0.01)
f1_tresholds = [custom_metrics(y_val, (y_pred_val >= t).astype(int)).get("F1") for t in thresholds]

# test results for best treshold
dta = custom_metrics(y_test, (y_pred_test >= thresholds[np.argmax(f1_tresholds)]).astype(int))

print(f"Best Threshold={thresholds[np.argmax(f1_tresholds)]}, F-Score={max(f1_tresholds)}")
print("DTA: ", dta)
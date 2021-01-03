from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from metrics import custom_metrics
from data import *

min_class = [2]  # Minority classes
maj_class = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # Majority classes

imb_ratio = 0.01  # Imbalance ratio

X_train, y_train, X_test, y_test = load_image("mnist")
X_train, y_train, X_val, y_val, X_test, y_test = create_data(X_train, y_train, X_test, y_test, min_class, maj_class, imb_ratio=imb_ratio)

model = Sequential([
Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=X_train.shape[1:]),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(32, (5, 5), strides=(1, 1),  activation='relu'),
MaxPooling2D(pool_size=(2, 2)),
Flatten(),
Dense(256, activation='relu'),
Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val), verbose=1)

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
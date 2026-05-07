import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("Training Sleep Apnea Model...")

# -----------------------------------
# 1. CREATE DATA (5 FEATURES)
# -----------------------------------
np.random.seed(42)
n_samples = 2000

SpO2 = np.random.normal(94, 3, n_samples)
heart_rate = np.random.normal(75, 12, n_samples)
breathing_rate = np.random.normal(16, 3, n_samples)
snoring = np.random.normal(0.5, 0.3, n_samples)
BMI = np.random.normal(27, 5, n_samples)

# Combine ALL 5 features
X = np.column_stack((SpO2, heart_rate, breathing_rate, snoring, BMI))

# Labels (risk)
y = (
    (SpO2 < 92) |
    (heart_rate > 85) |
    (breathing_rate > 20) |
    (snoring > 0.7) |
    (BMI > 30)
).astype(int)

# -----------------------------------
# 2. SPLIT DATA
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# 3. SCALE DATA
# -----------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------
# 4. BUILD MODEL (INPUT = 5)
# -----------------------------------
model = Sequential([
    Input(shape=(5,)),   # IMPORTANT: 5 features
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------
# 5. TRAIN
# -----------------------------------
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# -----------------------------------
# 6. EVALUATE
# -----------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", acc)

# -----------------------------------
# 7. SAVE MODEL + SCALER
# -----------------------------------
model.save("sleep_apnea_model.h5")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved!")
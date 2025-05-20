import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


df = pd.read_csv("data/parkinsons.data")
df = df.drop(['name'], axis=1)

X = df.drop(['status'], axis=1)
y = df['status']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler.fit(X_train)  # Fit the scaler to your training data

joblib.dump(scaler, "models/scaler.pkl")

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(patience=10, restore_best_weights=True)

# Train model
model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[es], verbose=1)

# Evaluate
y_pred = model.predict(X_test).flatten() > 0.5
print(classification_report(y_test, y_pred))
model.save('my_model.keras')
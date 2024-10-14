import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.imputation import MeanMedianImputer
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection, DropDuplicateFeatures
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.preprocessing import StandardScaler
import os

merged_df = pd.read_csv('V2-merged_df_noOutliers.csv')

merged_df = merged_df.reset_index(drop=True)

# FOR MODEL

X = merged_df.drop(columns = ['Standard Value','pIC50 Value','Smiles','QT-pIC50 Value','cid'])
y = merged_df['pIC50 Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 42)

# Initialize the imputer with median imputation method
imputer = MeanMedianImputer(imputation_method='median')

# Fit the imputer to the training data
imputer.fit(X_train)

# Transform the training data
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Drop constant
drop_const = DropConstantFeatures(tol=0.999)

drop_const.fit(X_train)
X_train = drop_const.transform(X_train)
X_test = drop_const.transform(X_test)

#print(X_train.shape)

# Drop duplicates
drop_dup = DropDuplicateFeatures()

drop_dup.fit(X_train)
X_train = drop_dup.transform(X_train)
X_test = drop_dup.transform(X_test)


# Initialize Winsorizer transformer
winsorizer = Winsorizer(capping_method='gaussian', tail='both')

# Fit and transform on training data
winsorizer.fit(X_train)

X_train = winsorizer.transform(X_train)
X_test = winsorizer.transform(X_test)


from feature_engine.transformation import YeoJohnsonTransformer

yeo_trans = YeoJohnsonTransformer()
yeo_trans.fit(X_train)

X_train = yeo_trans.transform(X_train)
X_test = yeo_trans.transform(X_test)

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Instantiate the StandardScaler
scaler = StandardScaler()

# Fit and transform on the training data
scaler.fit(X_train)

# Scale the training data
X_train_eda = scaler.transform(X_train)
X_test_eda = scaler.transform(X_test)

# Convert the numpy arrays back to DataFrames and assign column names
X_train = pd.DataFrame(X_train_eda, columns=X_train.columns)
X_test = pd.DataFrame(X_test_eda, columns=X_test.columns)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping


# Define the improved neural network model
model_1 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),  # Batch normalization layer
    Dropout(0.4),  # Reduced dropout rate
    Dense(64, activation='relu'),
    BatchNormalization(),  # Batch normalization layer
    Dropout(0.3),  # Reduced dropout rate
    Dense(32, activation='relu'),
    BatchNormalization(),  # Batch normalization layer
    Dropout(0.2),  # Reduced dropout ra
    Dense(16, activation='relu'),
    BatchNormalization(),  # Batch normalization layer
    Dense(1, activation='linear')  # Linear activation for regression
])


# Compile the model with an optimized optimizer (e.g., Adam) and learning rate
optimizer = Adam(learning_rate=0.0007)  # Increased learning rate
model_1.compile(
    optimizer=optimizer,
    loss='mean_squared_error',
    metrics=['mae']
)

# Apply learning rate reduction on plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.000001)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=30,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Train the model with adjusted batch size and epochs
history_improved = model_1.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=200,
    validation_data=(X_test, y_test),
    shuffle=True,
    verbose=1,
    callbacks=[reduce_lr,early_stopping]
)


# Feature Importance:
import shap
import pandas as pd
import numpy as np

# Create an explainer object for your NN model
explainer = shap.KernelExplainer(model_1.predict, X_train)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Calculate the mean absolute SHAP value for each feature
mean_shap_values = np.mean(np.abs(shap_values), axis=0)

# Ensure mean_shap_values is 1D
mean_shap_values = mean_shap_values.flatten()

# Check if the lengths of feature names and SHAP values match
if len(X_train.columns) != len(mean_shap_values):
    raise ValueError("The length of feature names and SHAP values must match")

# Create a DataFrame to store feature names and their importance coefficients
shap_importance_df = pd.DataFrame({
    'Feature': X_train.columns,  # Assuming X_train is a DataFrame with feature names
    'Importance': mean_shap_values
})

# Sort by importance (optional)
shap_importance_df = shap_importance_df.sort_values(by='Importance', ascending=False)

# Write the DataFrame to a CSV file
shap_importance_df.to_csv('V2-feature_importance_model_SHAP.csv', index=False)

print("SHAP feature importance saved to 'V2-feature_importance_model_SHAP.csv'.")


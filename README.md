# **Credit Card Fraud Detection Using Neural Networks**

## **Overview**
This project focuses on **detecting fraudulent transactions** using a neural network-based classification model. The dataset is preprocessed, features are engineered, and a shallow neural network is trained to classify transactions as fraudulent or non-fraudulent.

## **Dataset**
The dataset used is `creditcard.csv`, which contains credit card transactions with class labels:
- `0`: Non-fraudulent transactions
- `1`: Fraudulent transactions

## **Project Workflow**
### **1. Data Preprocessing**
- Load the dataset using `pandas`
- Explore basic statistics using `.describe()`
- Analyze class distribution (`df['Class'].value_counts()`)

### **2. Data Visualization**
- Histograms for feature distribution:
  ```python
  df.hist(bins=30, figsize=(30,30))
  plt.show()
  ```
- Detect outliers in `Amount` and `Time` columns

### **3. Feature Engineering**
- Apply `RobustScaler` to normalize transaction amounts
- Normalize `Time` column using **Min-Max Scaling**
- Shuffle the dataset to ensure randomness in training

### **4. Data Splitting**
The dataset is split into **training, testing, and validation** sets:
- **Training Set:** First 240,000 samples  
- **Testing Set:** Next 22,000 samples  
- **Validation Set:** Remaining samples  

Conversion to NumPy:
```python
train_np , test_np , val_np = train.to_numpy(), test.to_numpy(), val.to_numpy()
x_train, y_train = train_np[:, :-1], train_np[:, -1]
x_test, y_test = test_np[:, :-1], test_np[:, -1]
x_val, y_val = val_np[:, :-1], val_np[:, -1]
```

### **5. Neural Network Model**
The model consists of:
- **Input Layer**: Takes feature vectors
- **Dense Layer with ReLU**: Extracts key features
- **Batch Normalization Layer**: Improves training stability
- **Sigmoid Output Layer**: Predicts fraud probability

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization

shallow_nn = Sequential()
shallow_nn.add(InputLayer((x_train.shape[1],)))
shallow_nn.add(Dense(2, activation='relu'))
shallow_nn.add(BatchNormalization())
shallow_nn.add(Dense(1, activation='sigmoid'))
```

### **6. Model Training**
- Uses `Adam` optimizer with `binary_crossentropy` loss
- Training for **5 epochs** with validation on `x_val, y_val`
- Model checkpoint to save the best performing model

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('shallow_nn.keras', save_best_only=True)
shallow_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
shallow_nn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, callbacks=checkpoint)
```

### **7. Model Evaluation**
- Predictions thresholded at **0.5** for fraud detection
- Classification report generated:

```python
from sklearn.metrics import classification_report

predictions = shallow_nn.predict(x_val)
predictions = (predictions > 0.5).astype(int)

print(classification_report(y_val, predictions, target_names=['Not Fraud', 'Fraud']))
```

## **Key Learnings**
- Feature scaling impacts model performance significantly
- Balancing the dataset is crucial for fraud detection
- Neural networks can effectively classify imbalanced data


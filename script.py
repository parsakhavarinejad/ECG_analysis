from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import Preprocessing
from model import ConvNet


train_path = 'data/mitbih_train.csv.csv'
test_path = 'data/mitbih_test.csv'
preprocessing = Preprocessing(train_path, test_path)
_, ecg_test = preprocessing.vis()
new_train = preprocessing.resampling_data()
X = new_train.iloc[:,:186]
y = new_train.iloc[:,187]

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ConvNet()
model.build(input_shape=(None, 186, 1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the Model

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print(history)
fig1, ax_acc = plt.subplots()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

fig2, ax_loss = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
target_names=['0','1','2','3','4']


y_true = np.argmax(y_test, axis=1)
prediction_proba = model.predict(X_test)
prediction = np.argmax(prediction_proba, axis=1)
cnf_matrix = confusion_matrix(y_true, prediction)

plt.figure(figsize=(16, 9))
x_axis_labels = ['Predicted normal', 'Predicted Unknown', 'Predicted Ventricular', 'Predicted Supraventricular', 'Predicted Fusion']
y_axis_labels = ['normal', 'Unknown', 'Ventricular', 'Supraventricular', 'Fusion']
sns.heatmap(cnf_matrix, annot=True, fmt="d", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.show()

ecg_test[187] = ecg_test[187].astype(int)
counts = ecg_test[187].value_counts()

X_ecg_test = ecg_test.iloc[:, :186]
y_ecg_test = ecg_test.iloc[:, 187]
X_ecg_test = X_ecg_test.values.reshape(-1, 186, 1)

prediction_proba = model.predict(X_ecg_test)
prediction = np.argmax(prediction_proba, axis=1)
cnf_matrix = confusion_matrix(y_ecg_test, prediction)

plt.figure(figsize=(16, 9))
x_axis_labels = ['Predicted normal', 'Predicted Unknown', 'Predicted Ventricular', 'Predicted Supraventricular', 'Predicted Fusion']
y_axis_labels = ['normal', 'Unknown', 'Ventricular', 'Supraventricular', 'Fusion']
sns.heatmap(cnf_matrix, annot=True, fmt="d", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.show()

target_names = ['Predicted normal', 'Predicted Unknown', 'Predicted Ventricular', 'Predicted Supraventricular', 'Predicted Fusion']
print(classification_report(y_ecg_test, prediction, target_names=target_names))




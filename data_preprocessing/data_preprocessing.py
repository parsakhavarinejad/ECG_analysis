import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

ecg_test = pd.read_csv('../data/mitbih_test.csv', header=None)
ecg_train = pd.read_csv('../data/mitbih_train.csv', header=None)

ecg_train[187] = ecg_train[187].astype(int)
counts = ecg_train[187].value_counts()
counts = counts.reset_index()
counts = counts.rename(columns={counts.columns[1]: 'class'})

classes = {0: 'Non-ecotic beats (normal beat)',
           1: 'Supraventricular ectopic beats',
           2: 'Ventricular ectopic beats',
           3: 'Fusion beat',
           4: 'Unknown Beats'}

class_counts = {}
for class_number, class_name in classes.items():
    class_counts[class_name] = counts['class'][counts.index == class_number].values[0]

for class_number, class_name in class_counts.items():
    print('The length of {} is {}'.format(class_number, class_name))

plt.figure(figsize=(25, 15))
plt.figure(figsize=(25, 15))
counts['class'].plot(kind='pie', labels=['Non-ecotic beats (normal beat)',
                                         'Unknown Beats',
                                         'Ventricular ectopic beats',
                                         'Supraventricular ectopic beats',
                                         'Fusion Beats'], autopct='%1.1f%%')
plt.axis('equal')
plt.title('distribution of each class (types of diseases)')
plt.show()

for i in range(3):
    plt.figure(figsize=(14, 10))
    n = ecg_train.sample(1)
    plt.plot(n.iloc[0, :186])
    plt.title(classes[int(n[187])])

    plt.show()

plt.figure(figsize=(17, 10))
for i in range(5):
    n = ecg_train.sample(1)
    plt.plot(n.iloc[0, :186], label=classes[int(n[187])])
    plt.title('Different types')
    plt.legend(loc='best')

train_upsample1 = ecg_train[ecg_train[187] == 1]
train_upsample2 = ecg_train[ecg_train[187] == 2]
train_upsample3 = ecg_train[ecg_train[187] == 3]
train_upsample4 = ecg_train[ecg_train[187] == 4]
train_upsample0 = (ecg_train[ecg_train[187] == 0]).sample(n=10000, random_state=1)

train1_upsample = resample(train_upsample1, replace=True, n_samples=10000, random_state=43)
train2_upsample = resample(train_upsample2, replace=True, n_samples=10000, random_state=44)
train3_upsample = resample(train_upsample3, replace=True, n_samples=10000, random_state=54)
train4_upsample = resample(train_upsample4, replace=True, n_samples=10000, random_state=65)

new_train = pd.concat([train_upsample0, train1_upsample, train2_upsample, train3_upsample, train4_upsample])
new_train[187] = new_train[187].astype(int)
counts = new_train[187].value_counts()

plt.figure(figsize=(25, 15))
counts.plot(kind='pie', labels=['Non-ecotic beats (normal beat)',
                                'Unknown Beats',
                                'Ventricular ectopic beats',
                                'Supraventricular ectopic beats',
                                'Fusion Beats'], autopct='%1.1f%%')
plt.title("Upsampled data")
plt.show()

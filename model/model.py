import tensorflow as tf


class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='tanh', padding='same', input_shape=(186, 1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.maxpool1 = tf.keras.layers.MaxPool1D(5, padding='same')

        self.conv2 = tf.keras.layers.Conv1D(64, kernel_size=7, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.maxpool2 = tf.keras.layers.MaxPool1D(5, padding='same')

        self.conv3 = tf.keras.layers.Conv1D(128, kernel_size=7, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        self.maxpool3 = tf.keras.layers.MaxPool1D(5, padding='same')

        self.conv4 = tf.keras.layers.Conv1D(256, kernel_size=7, padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.relu4 = tf.keras.layers.ReLU()
        self.maxpool4 = tf.keras.layers.MaxPool1D(5, padding='same')

        self.conv5 = tf.keras.layers.Conv1D(512, kernel_size=7, padding='same')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.relu5 = tf.keras.layers.ReLU()
        self.maxpool5 = tf.keras.layers.MaxPool1D(5, padding='same')

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.fc4 = tf.keras.layers.Dense(64, activation='relu')
        self.fc5 = tf.keras.layers.Dense(32, activation='relu')
        self.fc6 = tf.keras.layers.Dense(5, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        return x

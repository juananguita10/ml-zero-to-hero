import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist # Fashion MNIST data
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Print two different boots images
#plt.imshow(training_images[0])
#print(training_labels[0])
#print(training_images[0])
#print(training_labels[42])
#print(training_images[42])

# Normalizing images (0 to 1)
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Define model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Build model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Try our model out with unseen data
model.evaluate(test_images, test_labels)

# Exercise 1
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

# Exercise 2


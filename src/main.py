########################################################################################################################################

# This code imports TensorFlow and two libraries, Matplotlib and NumPy, for data visualization. 
# The MobileNetV2 model and the ImageNet dataset are also imported.

# The code defines four functions:

# preprocess(image): resizes the image to 224x224, converts the image to a float tensor, preprocesses the input image for MobileNetV2 
# by subtracting the mean RGB values of the ImageNet dataset, and adds a batch dimension to the tensor.
# get_imagenet_label(probs): returns the label of the highest predicted probability class for an input image.
# create_adversarial_pattern(input_image, input_label): creates an adversarial perturbation for an input image, given a target label, 
# using the gradient of the loss with respect to the input image.
# display_images(image, description): displays an image along with its predicted label and confidence score.
# The code sets the parameters for Matplotlib visualization and loads a sample image of a yellow Labrador retriever. The image 
# is preprocessed using preprocess() and the MobileNetV2 model is used to make a prediction on the image.

# The code then uses create_adversarial_pattern() to generate a perturbation for the image, given a target label, and uses 
# Matplotlib to visualize the perturbation. The code then applies different levels of perturbation to the original image using a 
# range of epsilon values and displays the results using display_images().

# Finally, the code prints a message to indicate that it has finished executing.

########################################################################################################################################

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

image_path = "../img/panda.jpg"
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)

image = preprocess(image)
image_probs = pretrained_model.predict(image)

plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
# plt.show()

loss_object = tf.keras.losses.CategoricalCrossentropy()

# Get the input label of the image.
labrador_retriever_index = 208
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adversarial_pattern(image, label)
plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]
# plt.show()

epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

for i, eps in enumerate(epsilons):
  adv_x = image + eps*perturbations
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  display_images(adv_x, descriptions[i])

print("-- Done executing functions ---")
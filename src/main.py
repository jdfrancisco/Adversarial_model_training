###########################################################################################################################################################
#
# The provided code demonstrates the implementation of the Fast Gradient Sign Method (FGSM) for adversarial attacks and defenses 
# using the MobileNetV2 model. It starts by importing the necessary libraries and defining helper functions for image preprocessing, 
# label extraction, and image display. The MobileNetV2 model is loaded and configured for use. An input image is loaded, preprocessed, 
# and its predicted label is obtained. Using the FGSM method, an adversarial pattern is created by calculating the gradient of the loss 
# with respect to the input image and applying a sign operation. Multiple epsilon values are selected, and adversarial examples are generated 
# by perturbing the input image based on the adversarial pattern. The original image and adversarial examples are displayed. The code then proceeds 
# to execute the FGSM attack by applying the adversarial pattern to the input image iteratively. A defense function is defined to restore the original 
# image by subtracting the perturbation from the defense image iteratively. The defended images are displayed. Finally, the regular, attacked, 
# and defended images are evaluated and compared by obtaining the model's predictions and confidence scores. The code assumes the availability of an 
# input image file and utilizes the MobileNetV2 model pretrained on the ImageNet dataset. Adjustments may be required for file paths and model 
# parameters based on individual setups.
#
###########################################################################################################################################################

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

# Helper function to preprocess the image so that it can be inputted to MobileNetV2
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

# Helper function to extract labels from the probability vector
def get_imagenet_label(probs):
    return tf.keras.applications.mobilenet_v2.decode_predictions(probs, top=1)[0][0]

def display_images(image, description):
    _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
    plt.figure()
    plt.imshow(image[0] * 0.5 + 0.5)
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence*100))
    plt.show()

print("-- Start executing FGSM Attack ---")

###########################################################################################################################################################
#
# This function creates an adversarial pattern using the Fast Gradient Sign Method (FGSM). It calculates the gradient of the loss with respect to the 
# input image using a gradient tape. The input image is scaled by a temperature value before passing it through the pretrained MobileNetV2 model to obtain 
# predictions. The loss is calculated using the provided loss_object function. The function then computes the gradient of the loss with respect to the 
# input image and applies a sign operation to obtain the adversarial pattern.
#
###########################################################################################################################################################
def create_adversarial_pattern(input_image, input_label, temperature=1.0):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image) / temperature  # Apply temperature to soften the probabilities
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = True

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
plt.show()

loss_object = tf.keras.losses.CategoricalCrossentropy()

label_index = np.argmax(pretrained_model.predict(image), axis=1)  # Get the predicted label index
label = tf.one_hot(label_index, pretrained_model.output_shape[-1])  # Convert to one-hot encoded format

perturbations = create_adversarial_pattern(image, label)
plt.imshow(perturbations[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
plt.show()

epsilons = [0, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

for i, eps in enumerate(epsilons):
    adv_x = image + eps * perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    display_images(adv_x, descriptions[i])

print("-- End executing FGSM Attack ---")

print("-- Start executing FGSM Defense and Evaluation ---")

###########################################################################################################################################################
#
# This function performs FGSM defense by iteratively subtracting the adversarial perturbations from the input image. 
# The process is repeated for the specified number of iterations. The resulting defense image is clipped to ensure it 
# stays within the valid pixel value range.
#
###########################################################################################################################################################
def fgsm_defense(input_image, input_label, epsilon, num_iterations):
    defense_image = input_image
    for _ in range(num_iterations):
        perturbations = create_adversarial_pattern(defense_image, input_label)
        defense_image = defense_image - epsilon * perturbations
        defense_image = tf.clip_by_value(defense_image, -1, 1)
    return defense_image

epsilon = 0.001  # Set the defense strength
iterations = 10
defense_image = fgsm_defense(adv_x, label, epsilon, iterations)
display_images(defense_image, 'FGSM Defense (Epsilon = {:.3f})'.format(epsilon))

###########################################################################################################################################################
#
# This function evaluates the regular image, attacked image, and defended image by obtaining predictions and confidence scores from the MobileNetV2 
# model. The regular image, attacked image, and defended image are displayed along with their respective labels and confidence scores. The function 
# also prints the label and confidence for each image.
#
###########################################################################################################################################################
def evaluate(attacked_image, defended_image, regular_image):
    attacked_pred = pretrained_model.predict(attacked_image)
    defended_pred = pretrained_model.predict(defended_image)
    regular_pred = pretrained_model.predict(regular_image)

    attacked_label, _, attacked_confidence = get_imagenet_label(attacked_pred)
    defended_label, _, defended_confidence = get_imagenet_label(defended_pred)
    regular_label, _, regular_confidence = get_imagenet_label(regular_pred)

    # Display regular image
    plt.figure()
    plt.imshow(regular_image[0] * 0.5 + 0.5)
    plt.title('Regular Image \n {} : {:.2f}% Confidence'.format(regular_label, regular_confidence * 100))
    plt.show()

    # Display attacked image
    plt.figure()
    plt.imshow(attacked_image[0] * 0.5 + 0.5)
    plt.title('Attacked Image \n {} : {:.2f}% Confidence'.format(attacked_label, attacked_confidence * 100))
    plt.show()

    # Display defended image
    plt.figure()
    plt.imshow(defended_image[0] * 0.5 + 0.5)
    plt.title('Defended Image \n {} : {:.2f}% Confidence'.format(defended_label, defended_confidence * 100))
    plt.show()

    print("\nRegular Image:")
    print("Label: {} - Confidence: {:.2f}%".format(regular_label, regular_confidence * 100))

    print("\nAttacked Image:")
    print("Label: {} - Confidence: {:.2f}%".format(attacked_label, attacked_confidence * 100))

    print("\nDefended Image:")
    print("Label: {} - Confidence: {:.2f}%".format(defended_label, defended_confidence * 100))

evaluate(adv_x, defense_image, image)

print("-- End executing FGSM Defense and Evaluation ---")
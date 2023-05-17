README:

This code demonstrates the use of the Fast Gradient Sign Method (FGSM) for adversarial attacks and defenses using the MobileNetV2 model. It perturbs an input image to generate an adversarial example and then applies a defense mechanism to restore the image to its original class.

Prerequisites:
- Numpy
- TensorFlow
- Matplotlib

Instructions:
1. Import the required libraries.
2. Define helper functions for image preprocessing, label extraction, and image display.
3. Load the pretrained MobileNetV2 model and configure it.
4. Load an input image and preprocess it.
5. Obtain the predicted label for the input image.
6. Create the adversarial pattern using the FGSM method.
7. Choose epsilon values (perturbation strength) and generate adversarial examples.
8. Display the original image and the adversarial examples.
9. Start executing the FGSM attack.
10. Define the FGSM defense function.
11. Specify the defense parameters (epsilon and number of iterations).
12. Apply the defense to the adversarial examples.
13. Display the defended images.
14. Evaluate and compare the regular, attacked, and defended images.
15. End executing the FGSM attack and defense.

Please note that the code assumes the presence of an input image file and uses the MobileNetV2 model pretrained on the ImageNet dataset. You may need to adjust the file path and model parameters based on your specific setup.

To run the code, execute the script and observe the output images and confidence scores for the regular, attacked, and defended images.


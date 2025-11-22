import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shap


# ---------------------------------------------------------
# Grad-CAM for Image Model
# ---------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer_name="backbone.7"):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Register hooks
        for name, module in model.named_modules():
            if name == target_layer_name:
                module.register_forward_hook(self._forward_hook)
                module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, image_tensor):
        image_tensor = image_tensor.requires_grad_(True)

        out = self.model(image_tensor)
        target = out.max()

        # backward pass
        target.backward()

        # gradients and activations
        grads = self.gradients.mean(dim=(2, 3), keepdim=True)
        cams = (grads * self.activations).sum(dim=1, keepdim=True)
        cams = torch.relu(cams)

        cam = cams[0].detach().cpu().numpy()[0]
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam


# ---------------------------------------------------------
# SHAP for Tabular Sensor Data
# ---------------------------------------------------------

def shap_sensor_explain(model, sample, feature_names=None):
    """
    sample: tensor of shape (1, seq_len, features)
    """
    explainer = shap.DeepExplainer(model, sample)
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, feature_names=feature_names)


# ---------------------------------------------------------
# SHAP for Text Model (BERT)
# ---------------------------------------------------------

def shap_text_explain(model, tokenizer, text):
    tokens = tokenizer(text, return_tensors="pt")
    explainer = shap.Explainer(model, masker=tokenizer)
    values = explainer(text)

    shap.plots.text(values[0])


# ---------------------------------------------------------
# Probability Plot
# ---------------------------------------------------------

def plot_probabilities(probs, labels):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs)
    plt.ylabel("Probability")
    plt.title("Fault Prediction Confidence")
    plt.show()

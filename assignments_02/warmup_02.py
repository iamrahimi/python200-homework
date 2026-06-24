import os
import torch
import torchvision
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import torch.nn.functional as F

# --------------------
# Setup
# --------------------
os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)

# --------------------
# PyTorch Tensors
# --------------------

# Q1
a = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

b = torch.zeros(2, 3)
c = torch.ones(4)

print("\n--- Q1 ---")
for name, t in [("a", a), ("b", b), ("c", c)]:
    print(f"\n{name}")
    print("value:\n", t)
    print("shape:", t.shape)
    print("dtype:", t.dtype)
    print("device:", t.device)

# Q2
x = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])

print("\n--- Q2 ---")
print("sqrt:", torch.sqrt(x))
print("sum:", x.sum())
print("mean:", x.mean())
print("argmax:", x.argmax())

# Q3
print("\n--- Q3 ---")
a_gpu = a.to(device)
print("a_gpu device:", a_gpu.device)

a_back = a_gpu.cpu()
a_numpy = a_back.numpy()

print("numpy type:", type(a_numpy))
print("numpy values:\n", a_numpy)

# Q4
print("\n--- Q4 ---")
t = torch.arange(24).float()

print("reshape (4,6):", t.reshape(4, 6).shape)
print("reshape (2,3,4):", t.reshape(2, 3, 4).shape)

t_batch = t.reshape(4, 6).unsqueeze(0)
print("add batch dim:", t_batch.shape)

# Q5
print("\n--- Q5 ---")
np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
np_b = np.array([[5.0, 6.0], [7.0, 8.0]])

t_a = torch.tensor(np_a, dtype=torch.float32)
t_b = torch.tensor(np_b, dtype=torch.float32)

print("NumPy:\n", np.dot(np_a, np_b))
print("Torch:\n", torch.matmul(t_a, t_b))

# --------------------
# Pretrained Model
# --------------------

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

print("\n--- Model Q1 ---")
print("Total params:", sum(p.numel() for p in model.parameters()))
print("Trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))

print("\n--- Model Q2 ---")
print(model)

model = model.to(device)
model.eval()

print("\n--- Model Q3 ---")
print("Model ready on device")

preprocess = weights.transforms()

print("\n--- Model Q4 ---")
print(preprocess)

# --------------------
# Dataset Setup
# --------------------

DATA_DIR = Path("/kaggle/input/intel-image-classification/seg_test/seg_test")

LABELS = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

imagenet_classes = weights.meta["categories"]

def load_sample_image(label):
    class_dir = DATA_DIR / label
    img_path = random.choice(list(class_dir.glob("*.jpg")))
    return Image.open(img_path).convert("RGB"), img_path.name

# --------------------
# Inference Function
# --------------------

def get_top5_predictions(model, preprocess, image, device, class_labels):
    img = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)

    probs = F.softmax(out[0], dim=0)

    top_probs, top_idx = torch.topk(probs, 5)

    return [(class_labels[i], float(p)) for p, i in zip(top_probs, top_idx)]

# --------------------
# Inference Tests
# --------------------

print("\n--- Inference Q1 ---")
img, name = load_sample_image("mountain")
preds = get_top5_predictions(model, preprocess, img, device, imagenet_classes)

print(name)
for c, p in preds:
    print(c, p)

print("\n--- Inference Q2 ---")
for label in LABELS:
    img, name = load_sample_image(label)
    preds = get_top5_predictions(model, preprocess, img, device, imagenet_classes)[:3]

    print("\n", label, name)
    for c, p in preds:
        print(c, p)

print("\n--- Inference Q3 ---")
img, _ = load_sample_image("forest")
x = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)

probs = F.softmax(logits[0], dim=0)

print("logits range:", logits.min().item(), logits.max().item())
print("prob sum:", probs.sum().item())
print("top:", imagenet_classes[probs.argmax().item()])

# --------------------
# Visualization (Q4)
# --------------------

img, _ = load_sample_image("mountain")
preds = get_top5_predictions(model, preprocess, img, device, imagenet_classes)

labels = [p[0] for p in preds]
values = [p[1] for p in preds]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(img)
ax[0].axis("off")

ax[1].barh(labels[::-1], values[::-1])

plt.tight_layout()
plt.savefig("outputs/warmup_inference_viz.png")
plt.show()

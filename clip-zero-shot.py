import os
import torch
import clip
from PIL import Image

# انتخاب دستگاه
device = "cuda" if torch.cuda.is_available() else "cpu"

# بارگذاری مدل CLIP
model, preprocess = clip.load("ViT-B/32", device=device)

# لیست کلاس‌ها به زبان طبیعی
class_names = [
    "a photo of a cat","a cute kitten", "an animal that meows",
    "a photo of a girl watching fireworks", "a young woman enjoying fireworks",
    "a photo of a lighty souls", "a bright spirit glowing",
    "a photo of skyscrappers", "a sea", "a beautiful sunshine",
    "a photo pf a little snowman", "a cute smile", "little light",
    "christmas trees", "homes design for christmas", "kids walking in snow", "snowy town"
]

# تبدیل متن‌ها به embedding
text_tokens = clip.tokenize(class_names).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# پوشه تصاویر
image_folder = "images"  # مسیر پوشه شامل تصاویر

#حلقه بررسی تصاویر
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze(0)
            probs = similarity.softmax(dim=0).cpu().numpy()

        # نمایش نتیجه
        print(f"\nImage: {filename}")
        for label, prob in sorted(zip(class_names, probs), key=lambda x: -x[1]):
            print(f"  {label:<30} => {prob:.4f}")

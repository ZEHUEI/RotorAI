import cv2
import os

#resive
input_folder = "../Phase3/Input"
output_folder = "../Phase3/Output"

os.makedirs(output_folder, exist_ok=True)

images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
total = len(images)

for i, filename in enumerate(images):
    img = cv2.imread(os.path.join(input_folder, filename))
    resized = cv2.resize(img, (1280, 720))
    cv2.imwrite(os.path.join(output_folder, filename), resized)
    print(f"{i+1}/{total} — {filename}")

print(f"Done! Saved to {output_folder}")
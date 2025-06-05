from pathlib import Path
from PIL import Image
import numpy as np

# Set base directory for mock data
base_dir = Path("data/classification/train")
class_names = ["cat", "dog"]
image_size = (224, 224)

# Create dummy images inside class folders
for class_name in class_names:
    class_dir = base_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):  # Create 2 sample images per class
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(class_dir / f"{class_name}_{i}.jpg")

print(f"Mock dataset created in: {base_dir.resolve()}")

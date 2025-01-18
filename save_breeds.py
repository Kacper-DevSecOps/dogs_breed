import json
from dataset_preparation import get_data  # Import from your dataset script

# Load data
train_loader, _, _, _ = get_data(batch_size=32, workers=8)

# Extract breed names and format them correctly
class_labels = [name.split("-")[-1] for name in train_loader.dataset.dataset.classes]

# Save to JSON file
CLASS_LABELS_PATH = "app/class_labels.json"
with open(CLASS_LABELS_PATH, "w") as f:
    json.dump(class_labels, f, indent=4)

print(f"Class labels saved to {CLASS_LABELS_PATH}")

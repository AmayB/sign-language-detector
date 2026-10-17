
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "my_hand"
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

total = 0

for letter in letters:
    folder = DATA_DIR / letter
    images = list(folder.glob("*.jpg")) if folder.exists() else []

    count = len(images)
    total += count

    print(f"{letter}: {count} images")

print("-" * 25)
print(f"Total images: {total}")

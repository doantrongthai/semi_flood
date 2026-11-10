import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import shutil
import numpy as np
from PIL import Image
from data.datasets import FloodUnlabeledDataset


def test_unlabeled_dataset_zero_leakage():
    temp_dir = tempfile.mkdtemp()
    try:
        # Create 5 dummy test images
        img_names = [f"drone_{i}.jpg" for i in range(5)]
        for name in img_names:
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(os.path.join(temp_dir, name))

        # Instantiate FloodUnlabeledDataset
        dataset = FloodUnlabeledDataset(img_dir=temp_dir, file_list=img_names, size=64)

        # 1. Assert no mask attribute exists on the dataset class
        assert not hasattr(dataset, "mask_dir"), "SECURITY VIOLATION: FloodUnlabeledDataset must not hold mask_dir."
        assert not hasattr(dataset, "masks"), "SECURITY VIOLATION: FloodUnlabeledDataset must not hold masks."

        # 2. Retrieve an item and inspect returned tuple
        item = dataset[0]
        # Item format: (img_w, img_s1, img_s2, cutmix_box1, cutmix_box2, img_name)
        assert len(item) == 6, f"Expected 6 returned values, got {len(item)}"
        img_w, img_s1, img_s2, box1, box2, name = item

        assert img_w.shape == (3, 64, 64)
        assert img_s1.shape == (3, 64, 64)
        assert img_s2.shape == (3, 64, 64)
        assert box1.shape == (64, 64)
        assert box2.shape == (64, 64)
        assert isinstance(name, str)

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_unlabeled_dataset_zero_leakage()
    print("[PASS] Zero-information leakage test passed successfully!")

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import os
import shutil
from data.splits import get_or_create_trainvaltest_split, get_or_create_semi_split


def test_splits_determinism_and_nesting():
    temp_dir = tempfile.mkdtemp()
    try:
        dummy_images = [f"image_{i:04d}.jpg" for i in range(100)]

        # 1. Test train/val/test determinism across calls with seed=42
        split1 = get_or_create_trainvaltest_split(dummy_images, "testds", splits_dir=temp_dir, seed=42)
        # Clear file to re-run generation
        os.remove(os.path.join(temp_dir, "testds_trainvaltest.json"))
        split2 = get_or_create_trainvaltest_split(dummy_images, "testds", splits_dir=temp_dir, seed=42)

        assert split1["train"] == split2["train"]
        assert split1["val"] == split2["val"]
        assert split1["test"] == split2["test"]
        assert len(split1["train"]) == 70
        assert len(split1["val"]) == 15
        assert len(split1["test"]) == 15

        # 2. Test nested semi-splits
        train_list = split1["train"]
        semi_10 = get_or_create_semi_split(train_list, "testds", ratio=0.10, seed=42, splits_dir=temp_dir)
        semi_25 = get_or_create_semi_split(train_list, "testds", ratio=0.25, seed=42, splits_dir=temp_dir)
        semi_50 = get_or_create_semi_split(train_list, "testds", ratio=0.50, seed=42, splits_dir=temp_dir)

        s10_set = set(semi_10["labeled"])
        s25_set = set(semi_25["labeled"])
        s50_set = set(semi_50["labeled"])

        assert len(semi_10["labeled"]) == 7  # 10% of 70
        assert len(semi_25["labeled"]) == 18  # 25% of 70 (round 17.5 -> 18)
        assert len(semi_50["labeled"]) == 35  # 50% of 70

        # Verify labeled + unlabeled equals full train set
        assert set(semi_10["labeled"]).union(set(semi_10["unlabeled"])) == set(train_list)
        assert len(set(semi_10["labeled"]).intersection(set(semi_10["unlabeled"]))) == 0

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_splits_determinism_and_nesting()
    print("[PASS] Split determinism and nesting tests passed successfully!")

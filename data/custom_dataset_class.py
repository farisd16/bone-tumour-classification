import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
from typing import Dict, List, Optional

import pandas as pd


class CustomDataset(Dataset):
    def __init__(
        self,
        image_dir,
        json_dir,
        transform=None,
        minority_transform=None,
        minority_classes=None,
        xlsx_path=None,
        use_anatomical_location: bool = False,
    ):
        """
        Args:
            image_dir (str): Folder containing patched X-ray images
            json_dir (str): Folder containing JSON annotations
            xlsx_path (str, optional): Path to an .xlsx file
                with per-image metadata (expects an 'image_id' column and one-hot anatomical
                location columns).
            use_anatomical_location (bool): If True, labels become '<anatomical location> <tumor subtype>'.
            transform (callable, optional): torchvision transforms
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.xlsx_path = xlsx_path
        self.use_anatomical_location = bool(use_anatomical_location)

        self._tumor_classes = [
            "osteochondroma",
            "osteosarcoma",
            "multiple osteochondromas",
            "simple bone cyst",
            "giant cell tumor",
            "synovial osteochondroma",
            "osteofibroma",
        ]

        # self._anatomical_locations = [
        #     "hand",
        #     "ulna",
        #     "radius",
        #     "humerus",
        #     "foot",
        #     "tibia",
        #     "fibula",
        #     "femur",
        #     "hip bone",
        #     "ankle-joint",
        #     "knee-joint",
        #     "hip-joint",
        #     "wrist-joint",
        #     "elbow-joint",
        #     "shoulder-joint",
        # ]

        self._anatomical_locations = [
            "upper limb",
            "lower limb",
            "pelvis",
        ]

        self._location_by_image_id: Dict[str, str] = {}
        if self.use_anatomical_location:
            if not self.xlsx_path:
                raise ValueError(
                    "xlsx_path must be provided when use_anatomical_location=True"
                )
            self._location_by_image_id = self._load_location_by_image_id(
                self.xlsx_path, self._anatomical_locations
            )

        if self.use_anatomical_location:
            self.classes = [
                f"{loc} {tumor}"
                for loc in self._anatomical_locations
                for tumor in self._tumor_classes
            ]
        else:
            self.classes = list(self._tumor_classes)

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.transform = transform
        # Optional: apply different transform to specified minority classes
        self.minority_transform = minority_transform
        # store lowercased class names for comparison; default to empty set
        self.minority_classes = set([c.lower() for c in (minority_classes or [])])
        self.samples = []
        self._tumor_by_img_path: Dict[str, str] = {}

        for fname in sorted(os.listdir(image_dir)):
            if not fname.lower().endswith(".jpeg"):
                continue

            json_name = fname.replace(".jpeg", ".json")
            json_path = os.path.join(json_dir, json_name)

            with open(json_path, "r") as f:
                data = json.load(f)

            label = data["shapes"][0]["label"].lower()

            if label not in self._tumor_classes:
                continue

            img_path = os.path.join(image_dir, fname)

            if self.use_anatomical_location:
                image_id_key = os.path.basename(fname).strip().lower()
                location = self._location_by_image_id.get(image_id_key, "unknown")
                composite_label = f"{location} {label}"
                self.samples.append((img_path, composite_label))
            else:
                self.samples.append((img_path, label))

            self._tumor_by_img_path[img_path] = label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        tumor_label = self._tumor_by_img_path.get(img_path, label)

        # If a minority transform is provided and the sample's label is in the
        # configured minority set, apply that; else fall back to the default transform.
        if self.minority_transform is not None and tumor_label in self.minority_classes:
            image = self.minority_transform(image)
        elif self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )(image)

        label_idx = self.class_to_idx[label]
        return image, label_idx

    @staticmethod
    def _normalize_col_name(name: str) -> str:
        return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())

    @classmethod
    def _load_location_by_image_id(
        cls, xlsx_path: str, anatomical_locations: List[str]
    ) -> Dict[str, str]:
        df = pd.read_excel(xlsx_path)

        normalized_to_original = {
            cls._normalize_col_name(c): c for c in df.columns if str(c).strip() != ""
        }

        image_id_col = normalized_to_original.get("imageid")
        if image_id_col is None:
            raise ValueError(
                f"Expected an 'image_id' column in {xlsx_path} (case/format-insensitive)"
            )

        location_by_image_id: Dict[str, str] = {}
        for _, row in df.iterrows():
            image_id_raw = row.get(image_id_col)
            if image_id_raw is None or (
                isinstance(image_id_raw, float) and pd.isna(image_id_raw)
            ):
                continue
            image_id = os.path.basename(str(image_id_raw).strip()).lower()
            if not image_id:
                continue

            chosen_location: Optional[str] = None
            for loc in anatomical_locations:
                original_col = normalized_to_original.get(cls._normalize_col_name(loc))
                if original_col is None:
                    continue
                value = row.get(original_col)
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    continue
                try:
                    if int(value) == 1:
                        chosen_location = loc
                        break
                except (TypeError, ValueError):
                    continue

            location_by_image_id[image_id] = chosen_location or "unknown"

        return location_by_image_id

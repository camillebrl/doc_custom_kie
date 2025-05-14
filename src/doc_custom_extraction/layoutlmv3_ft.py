# Copyright 2025 Camille Barboule
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import os
import random
import shutil
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from seqeval.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, Trainer, TrainingArguments

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)


class CustomDataset(torch.utils.data.Dataset):
    """A simple PyTorch Dataset wrapping a list of dictionaries.

    This Dataset allows both indexed access and iteration over items.

    Attributes:
        items (list[Dict[str, Any]]): The list of example dictionaries to serve.
    """

    def __init__(self, items: list[dict[str, Any]]) -> None:
        """Initializes the dataset with a l of items.

        Args:
            items (list[dict[str, Any]]): list of examples, each a dict
                containing whatever fields downstream code expects
                (e.g., "image_path", "words", "bboxes", "labels").
        """
        super().__init__()
        self.items: list[dict[str, Any]] = items

    def __len__(self) -> int:
        """Returns the number of examples in the dataset.

        Returns:
            int: Length of the internal `items` list.
        """
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Retrieves a single example by index.

        Args:
            idx (int): Zero-based index of the example to retrieve.

        Returns:
            dict[str, Any]: The example dictionary at position `idx`.
        """
        return self.items[idx]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Provides an iterator over all examples in the dataset.

        Enables usage like `for example in dataset:`.

        Returns:
            Iterator[dict[str, Any]]: Iterator over the `items` list.
        """
        return iter(self.items)


def create_label_maps(categories: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    """Builds bidirectional mappings between label IDs and label strings.

    Args:
        categories (list[str]): List of entity category names (without the "O" tag).

    Returns:
        tuple[dict[int, str], dict[str, int]]:
            - id2label: Mapping from integer IDs to label names, with ID 0 → "O".
            - label2id: Inverse mapping from label names to their integer IDs.
    """
    labels = ["O"] + categories
    id2label = dict(enumerate(labels))
    label2id = {label: i for i, label in id2label.items()}
    return id2label, label2id


def load_json_data(jsonl_file: str) -> list[dict[str, Any]]:
    """Loads a JSONL file line by line, adjusting image paths if needed.

    Args:
        jsonl_file (str): Path to the input JSONL file.

    Returns:
        list[dict[str, any]]: List of deserialized JSON objects.
            Each dict may have its "image_path" prefixed.
    """
    data = []
    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Erreur de parsing JSON à la ligne: {e}")
                print(f"Ligne problématique: {line[:100]}...")
    return data


def prepare_dataset_from_jsonl(
    jsonl_file: str, split_ratio: float = 0.8
) -> tuple[CustomDataset, CustomDataset, dict[int, str], dict[str, int]]:
    """Prepares train and test datasets plus label mappings from a JSONL file.

    Reads examples, builds label mappings, splits into train/test, and
    wraps into PyTorch Datasets.

    Args:
        jsonl_file (str): Path to the JSONL annotation file.
        split_ratio (float): Fraction of data to use for training (default 0.8).

    Returns:
        tuple[
            CustomDataset,             # training dataset
            CustomDataset,             # testing dataset
            dict[int, str],            # id2label mapping
            dict[str, int]             # label2id mapping
        ]:
            - train_ds: Dataset for training.
            - test_ds: Dataset for evaluation.
            - id2label: Mapping from label IDs to label names.
            - label2id: Mapping from label names to label IDs.
    """
    data = load_json_data(jsonl_file)
    print(f"Nombre d'items chargés: {len(data)}")
    if data:
        print(f"Clés du premier item: {list(data[0].keys())}")
        print(f"Exemple de chemin d'image: {data[0].get('image_path', 'Non disponible')}")

    all_labels: list[str] = []
    for item in data:
        all_labels.extend(item.get("labels", []))
    unique_labels = set(all_labels)
    unique_labels.discard("")
    sorted_labels = sorted(unique_labels)
    id2label, label2id = create_label_maps(sorted_labels)

    train_items, test_items = train_test_split(data, test_size=1 - split_ratio, random_state=42)
    train_ds = CustomDataset(train_items)
    test_ds = CustomDataset(test_items)
    return train_ds, test_ds, id2label, label2id


class LayoutLMv3DataCollator:
    """Data collator for LayoutLMv3 token classification fine-tuning."""

    def __init__(self, processor: LayoutLMv3Processor, label2id: dict[str, int], max_length: int = 512) -> None:
        """Initializes the collator with model processor and label mapping.

        Args:
            processor (LayoutLMv3Processor): Pretrained LayoutLMv3 processor.
            label2id (dict[str, int]): Mapping from label string to label ID.
            max_length (int): Maximum sequence length for padding/truncation.
        """
        self.processor = processor
        self.label2id = label2id
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Converts a batch of features into model-ready inputs.

        Loads images, tokenizes text+boxes, and aligns labels to token ids.
        Applies a 1:1 ratio filter for the "O" label to balance positives.

        Args:
            features (list[dict[str, any]]): Batch examples, each containing:
                - "image_path": path to image file
                - "words": list of OCR tokens
                - "bboxes": list of bounding boxes
                - "labels": list of BIO labels

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys:
                - "pixel_values": Tensor of processed images
                - "input_ids", "attention_mask", "bbox", etc.
                - "labels": Tensor of label IDs aligned to tokens
        """
        images = [Image.open(f["image_path"]).convert("RGB") for f in features]
        texts = [f["words"] for f in features]
        boxes = [f["bboxes"] for f in features]
        labels = [f["labels"] for f in features]

        encoding = self.processor(
            images=images,
            text=texts,
            boxes=boxes,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        encoded_labels = []
        for i, word_labels in enumerate(labels):
            word_ids = encoding.word_ids(batch_index=i)
            raw_ids = [
                self.label2id.get(word_labels[w], self.label2id["O"]) if w is not None else -100 for w in word_ids
            ]
            # indices des O
            O_pos = [j for j, lab in enumerate(raw_ids) if lab == self.label2id["O"]]
            nonO_count = len(raw_ids) - len(O_pos)
            # ne conserver qu’un ratio 1:1
            keep_O = set(random.sample(O_pos, min(len(O_pos), nonO_count)))
            filtered = [
                lab if (lab != self.label2id["O"] or idx in keep_O) else -100 for idx, lab in enumerate(raw_ids)
            ]
            encoded_labels.append(torch.tensor(filtered))
        encoding["labels"] = torch.stack(encoded_labels)
        return encoding


def compute_metrics(p: tuple[np.ndarray, np.ndarray], id2label: dict[int, str]) -> dict[str, float]:
    """Computes accuracy and F1 score given model predictions and true labels.

    Aligns token-level predictions with labels, ignoring -100 padding tokens.

    Args:
        p (tuple[np.ndarray, np.ndarray]): Tuple of (logits, label_ids).
        id2label (dict[int, str]): Mapping from label IDs to label names.

    Returns:
        dict[str, float]: Dictionary with keys:
            - "accuracy": float accuracy score.
            - "f1": float F1 score.
    """
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_preds, true_labels = [], []
    for pred_seq, label_seq in zip(preds, labels, strict=False):
        seq_preds, seq_labels = [], []
        for p_id, l_id in zip(pred_seq, label_seq, strict=False):
            if l_id == -100:
                continue
            seq_preds.append(id2label[p_id])
            seq_labels.append(id2label[l_id])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    return {"accuracy": accuracy_score(true_labels, true_preds), "f1": f1_score(true_labels, true_preds)}


def main() -> None:
    """Entrypoint for fine-tuning LayoutLMv3 on token classification.

    Parses CLI args, prepares data, initializes model & Trainer,
    runs training and evaluation, then saves the final model and mappings.
    """
    parser = argparse.ArgumentParser(description="Fine-tuning LayoutLMv3")
    parser.add_argument("--annotation_file", type=str, default="temp_annot.jsonl")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--model_name", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--save_label_data",
        action="store_true",
        help="If set, saves label data and images into output_dir/data, else cleans temp files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    train_ds, test_ds, id2label, label2id = prepare_dataset_from_jsonl(
        args.annotation_file, split_ratio=args.split_ratio
    )
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

    # Calcul des class_weights sur l'ensemble d'entraînement
    all_labels_flat = []
    for item in train_ds:
        all_labels_flat.extend([label2id.get(l, label2id["O"]) for l in item["labels"]])
    freq = Counter(all_labels_flat)
    total = sum(freq.values())
    num_labels = len(id2label)
    # Évite les divisions par zéro si une classe est absente
    class_weights = torch.tensor([total / freq.get(i, 1) for i in range(num_labels)], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_labels

    # Initialisation du processor et modèle
    processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    # # Freeze du backbone
    # for param in model.layoutlmv3.parameters():
    #     param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = LayoutLMv3DataCollator(processor, label2id)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=args.logging_dir,
        report_to=["tensorboard"],
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval results:", metrics)

    # Sauvegarde du modèle final
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))
    with open(os.path.join(args.output_dir, "final_model", "label_mappings.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f)

    # Cleanup temp folders
    for folder in ["static", "temp_marked", "output_blocks"]:
        if os.path.isdir(folder):
            shutil.rmtree(folder)

    # Handle save_label_data flag
    images_dir = "temp_images"
    annot_file = args.annotation_file
    if args.save_label_data:
        data_dir = os.path.join(args.output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        # Copy and modify annotations
        with open(annot_file, encoding="utf-8") as f_in:
            lines = [json.loads(l) for l in f_in if l.strip()]
        for item in lines:
            if "image_path" in item:
                item["image_path"] = item["image_path"].replace("temp_images", "images")
        out_annot = os.path.join(data_dir, "annot.jsonl")
        with open(out_annot, "w", encoding="utf-8") as f_out:
            for item in lines:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        # Copy images
        images_dest = os.path.join(data_dir, "images")
        os.makedirs(images_dest, exist_ok=True)
        if os.path.isdir(images_dir):
            for fname in os.listdir(images_dir):
                shutil.copy2(os.path.join(images_dir, fname), images_dest)
    # Remove static content and temp annotation
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    if os.path.exists(annot_file):
        os.remove(annot_file)


if __name__ == "__main__":
    main()

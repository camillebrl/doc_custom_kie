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

"""Ce script est une application Flask qui permet :
  1. De détecter des blocs dans une image,
  2. D'appliquer un OCR sur chaque bloc de la même manière que le modèle entraîné,
  3. De normaliser les résultats et de prédire des étiquettes avec un notre LayoutLMv3 finetuné,
  4. De fusionner et post-traiter les entités extraites (puisque les entités sont labélisées en B- et I- pour former une seule entité),
  5. De retourner une image annotée et les entités détectées au format JSON.

Le chemin vers le modèle entraîné (directory contenant les poids et les mappings) est fourni
via l'argument de ligne de commande --model_path.
"""  # noqa: D205

import argparse
import base64
import io
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import torch
from flask import Flask, jsonify, render_template, request
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from PIL import Image as PILImage
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

app = Flask(__name__)

# Variables globales pour le modèle et les mappings
processor: LayoutLMv3Processor | None = None
model: LayoutLMv3ForTokenClassification | None = None
id2label: dict[int, str] = {}
label2id: dict[str, int] = {}
PALETTE = cast(Sequence[tuple[float, float, float, float]], cast(ListedColormap, plt.get_cmap("tab20")).colors)
COLOR_MAP: dict[str, str] = {}
device: torch.device


def initialize_model(model_path: str):
    """Charge le modèle LayoutLMv3 et les mappings de labels depuis le dossier model_path.
    Initialise les variables globales processor, model, id2label et label2id.
    """  # noqa: D205
    global processor, model, id2label, label2id, COLOR_MAP, device
    # Charger les mappings de labels
    mappings_path = os.path.join(model_path, "label_mappings.json")
    with open(mappings_path) as f:
        label_maps = json.load(f)

    # Mettre à jour les mappings
    id2label = {int(k): v for k, v in label_maps["id2label"].items()}
    label2id = label_maps["label2id"]

    print(f"Chargement du modèle depuis {model_path}...")
    # Charger le processor et le modèle
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    # Définir le device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Mode évaluation

    print("Modèle chargé avec succès!")


def get_color_for_label(label: str) -> str:
    """Return a hex color for the given label prefix, assigning one dynamically if needed.

    Cette fonction découpe le label au niveau du point pour en extraire le préfixe,
    puis renvoie la couleur hexadécimale correspondante dans le dictionnaire global
    `COLOR_MAP`. Si ce préfixe n’a pas encore de couleur assignée, en choisit une
    issue de la palette `PALETTE` de façon cyclique.

    Args:
        label (str): Label complet (ex. "ORG.1", "LOC") dont on veut la couleur.
            Le préfixe est défini comme la partie avant le premier point, ou le label
            entier s’il n’y a pas de point.

    Returns:
        str: Chaîne hexadécimale de la couleur associée au préfixe (ex. "#1f77b4").
    """
    prefix = label.split(".")[0] if "." in label else label
    if prefix not in COLOR_MAP:
        idx = len(COLOR_MAP) % len(PALETTE)
        COLOR_MAP[prefix] = matplotlib.colors.to_hex(PALETTE[idx])
    return COLOR_MAP[prefix]


def convert_nested_numpy_types(obj):
    """Recursively convert numpy types in nested data structures to Python types."""
    if isinstance(obj, np.integer | np.int64 | np.int32):
        return int(obj)
    elif isinstance(obj, np.floating | np.float64 | np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_nested_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_nested_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_nested_numpy_types(item) for item in obj)
    return obj


def detect_image_blocks(image, gradient_thresh=6000, min_size=10, smooth=1, recursive=False, depth=0, max_depth=2):
    """Détecte les blocs d'image en utilisant l'analyse de gradient
    et retourne les coordonnées des blocs plutôt que des sous-images.

    Args:
        image: Image d'entrée (BGR)
        gradient_thresh: Seuil pour la détection de gradient
        min_size: Taille minimale d'un bloc vide
        smooth: Marge pour adoucir les limites
        recursive: Si True, applique récursivement l'algorithme sur chaque bloc
        depth: Profondeur actuelle de récursion
        max_depth: Profondeur maximale de récursion

    Returns:
        Liste de tuples (x, y, w, h) représentant les coordonnées des blocs
    """  # noqa: D205
    # Obtenir les dimensions de l'image
    h, w = image.shape[:2]

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculer les gradients
    gradient_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    gradient_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    gradient_x = np.abs(gradient_x)
    gradient_y = np.abs(gradient_y)
    gradient = np.maximum(gradient_x, gradient_y)
    gradient[gradient < 50] = 0  # Supprimer le bruit de fond

    # Calculer les projections de gradient
    h_projection = np.sum(gradient, axis=0)  # Projection horizontale
    v_projection = np.sum(gradient, axis=1)  # Projection verticale

    # Trouver les séparations horizontales et verticales
    h_separations = find_separations(h_projection, gradient_thresh, min_size, smooth)
    v_separations = find_separations(v_projection, gradient_thresh, min_size, smooth)

    # Si aucune séparation n'est trouvée, retourner l'image entière
    if h_separations is None or v_separations is None:
        return [(0, 0, w, h)]

    # Convertir les séparations en régions
    h_regions = separations_to_regions(h_separations, w)
    v_regions = separations_to_regions(v_separations, h)

    # Créer la liste des coordonnées des blocs
    blocks = []
    for y_start, y_end in v_regions:
        for x_start, x_end in h_regions:
            # Vérifier que le bloc a une taille significative
            if (x_end - x_start) > 5 and (y_end - y_start) > 5:
                # Si récursif et pas atteint la profondeur maximale
                if recursive and depth < max_depth:
                    # Extraire le bloc pour l'analyse récursive
                    sub_image = image[y_start:y_end, x_start:x_end]

                    # Adapter le seuil en fonction de la taille du bloc
                    sub_thresh = gradient_thresh * (sub_image.shape[0] * sub_image.shape[1]) / (h * w)

                    # Détecter les sous-blocs
                    sub_blocks = detect_image_blocks(
                        sub_image,
                        gradient_thresh=sub_thresh,
                        min_size=min_size,
                        smooth=smooth,
                        recursive=recursive,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )

                    # Si aucun sous-bloc n'est trouvé ou un seul qui couvre tout
                    if len(sub_blocks) == 1 and sub_blocks[0] == (0, 0, sub_image.shape[1], sub_image.shape[0]):
                        blocks.append((x_start, y_start, x_end - x_start, y_end - y_start))
                    else:
                        # Ajuster les coordonnées des sous-blocs par rapport à l'image d'origine
                        for sx, sy, sw, sh in sub_blocks:
                            blocks.append((x_start + sx, y_start + sy, sw, sh))
                else:
                    blocks.append((x_start, y_start, x_end - x_start, y_end - y_start))

    return blocks


def find_separations(projection, threshold, min_size=10, smooth=1):
    """Trouve les séparations dans une projection de gradient.

    Args:
        projection: Projection du gradient (somme par ligne ou colonne)
        threshold: Seuil pour considérer une valeur comme significative
        min_size: Taille minimale d'une séparation
        smooth: Marge pour adoucir les limites

    Returns:
        Liste de séparations [début, fin]
    """
    # Trouver les indices où la projection dépasse le seuil
    indices = np.where(projection > threshold)[0]

    if len(indices) == 0:
        return None

    # Initialiser la liste des séparations
    separations = []

    # Ajouter le début si nécessaire
    if indices[0] > min_size:
        start = 0
        end = max(0, indices[0] - smooth)
        separations.append([start, end])

    # Trouver les blocs vides entre les zones d'intérêt
    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] > min_size:
            start = min(len(projection), indices[i] + smooth)
            end = max(0, indices[i + 1] - smooth)
            if start < end:
                separations.append([start, end])

    # Ajouter la fin si nécessaire
    if len(projection) - indices[-1] > min_size:
        start = min(len(projection), indices[-1] + smooth)
        end = len(projection)
        separations.append([start, end])

    return separations


def separations_to_regions(separations, size):
    """Convertit les séparations en régions.

    Args:
        separations: Liste de séparations [début, fin]
        size: Taille totale (largeur ou hauteur)

    Returns:
        Liste de régions [début, fin]
    """
    regions = []

    # Cas particulier: aucune séparation
    if len(separations) == 0:
        regions.append([0, size])
        return regions

    # Premier bloc si nécessaire
    if separations[0][0] > 0:
        regions.append([0, separations[0][0]])

    # Blocs intermédiaires
    for i in range(len(separations) - 1):
        regions.append([separations[i][1], separations[i + 1][0]])

    # Dernier bloc si nécessaire
    if separations[-1][1] < size:
        regions.append([separations[-1][1], size])

    return regions


def upscale_image(image, scale_factor=2.0, method="lanczos4"):
    """Redimensionne une image avec une méthode optimisée pour le texte et les détails.

    Args:
        image: Image à redimensionner
        scale_factor: Facteur d'agrandissement
        method: Méthode d'interpolation ('nearest', 'linear', 'cubic', 'lanczos4', 'waifu2x')

    Returns:
        Image redimensionnée
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    if method == "nearest":
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    elif method == "linear":
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    elif method == "cubic":
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif method == "lanczos4":
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # Méthode par défaut pour le texte (Lanczos)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Amélioration supplémentaire pour le texte (optionnelle)
    if method == "text_enhance":
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Reconversion en couleur
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Blend avec l'original pour conserver de la couleur
        alpha = 0.7
        resized = cv2.addWeighted(resized, alpha, enhanced_color, 1 - alpha, 0)

    return resized


def sharpen_image(image):
    """Applique un filtre de netteté à l'image pour améliorer la lisibilité du texte.

    Args:
        image: Image à améliorer

    Returns:
        Image améliorée
    """
    # Création du kernel de netteté
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Application du filtre
    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened


def extract_and_upscale_blocks(
    image_path,
    output_dir="output_blocks",
    max_depth=2,
    scale_factor=2.0,
    upscale_method="lanczos4",
    sharpen=True,
    save_quality=95,
):
    """Détecte, extrait, redimensionne et sauvegarde les blocs d'une image avec une haute qualité.

    Args:
        image_path: Chemin vers l'image d'entrée
        output_dir: Dossier de sortie pour les sous-images
        max_depth: Profondeur maximale de récursion
        scale_factor: Facteur d'agrandissement
        upscale_method: Méthode de redimensionnement
        sharpen: Si True, applique un filtre de netteté
        save_quality: Qualité de sauvegarde (0-100) pour les images JPEG
    """
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return

    # Adapter le seuil en fonction de la taille de l'image
    thresh = adaptive_threshold(image)

    # Détecter les blocs avec récursion
    blocks = detect_image_blocks(
        image, gradient_thresh=thresh, min_size=10, smooth=1, recursive=True, depth=0, max_depth=max_depth
    )

    # Sauvegarder les blocs avec haute qualité et redimensionnement
    blocks_paths = []
    for i, (x, y, w, h) in enumerate(blocks):
        # Extraire le bloc directement de l'image originale
        block_img = image[y : y + h, x : x + w]

        # Redimensionner le bloc
        upscaled_img = upscale_image(block_img, scale_factor, upscale_method)

        # Appliquer un filtre de netteté si demandé
        if sharpen:
            upscaled_img = sharpen_image(upscaled_img)

        # Déterminer le format de sortie
        _, ext = os.path.splitext(image_path)
        if ext.lower() in [".jpg", ".jpeg"]:
            # Pour JPEG, utiliser la qualité spécifiée
            output_path = os.path.join(output_dir, f"bloc_{i + 1}_upscaled.jpg")
            cv2.imwrite(output_path, upscaled_img, [cv2.IMWRITE_JPEG_QUALITY, save_quality])
        elif ext.lower() == ".png":
            # Pour PNG, utiliser la compression maximale
            output_path = os.path.join(output_dir, f"bloc_{i + 1}_upscaled.png")
            cv2.imwrite(output_path, upscaled_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            # Pour les autres formats, utiliser PNG sans perte
            output_path = os.path.join(output_dir, f"bloc_{i + 1}_upscaled.png")
            cv2.imwrite(output_path, upscaled_img)

        blocks_paths.append(output_path)

    print(
        f"{len(blocks)} blocs ont été détectés, agrandis par un facteur {scale_factor} "
        f"avec la méthode '{upscale_method}' et sauvegardés dans le dossier {output_dir}"
    )

    return blocks, blocks_paths


def adaptive_threshold(image, initial_thresh=6000, min_size=10):
    """Détermine automatiquement le seuil optimal pour la détection des blocs.

    Args:
        image: Image d'entrée
        initial_thresh: Seuil initial
        min_size: Taille minimale d'un bloc

    Returns:
        Seuil adapté
    """
    h, w = image.shape[:2]
    area = h * w

    # Adapter le seuil en fonction de la taille de l'image
    if area > 1000000:  # Grande image (> 1MP)
        return initial_thresh * 2
    elif area < 100000:  # Petite image (< 0.1MP)
        return initial_thresh / 2
    else:
        return initial_thresh


def process_image_with_block_ocr(
    image_path: str, output_dir: str = "output_blocks", tesseract_config: str = r"--oem 3 --psm 6 -l fra"
) -> dict:
    """Processes an image with block-based OCR extraction.

    This function segments an image into blocks, applies OCR to each block separately,
    and then combines the results back into a single set of coordinates relative to
    the original image. The approach improves OCR accuracy by:
    1. Breaking whole image into smaller text-only regions
    2. Upscaling each block for better text recognition
    3. Applying sharpening to enhance text clarity
    4. Applying Tesseract OCR on each block

    Args:
        image_path: Path to the input image file.
        output_dir: Directory where processed block images will be saved.
            Defaults to "output_blocks".
        tesseract_config: Configuration string for Tesseract OCR engine.
            Defaults to French language with automatic page segmentation.

    Returns:
        Dict containing:
            words: List of recognized text strings.
            bboxes: List of bounding boxes as tuples (x1, y1, x2, y2),
                with coordinates relative to the original image.
            confidences: List of confidence scores for each recognized word.
    """
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return {"words": [], "bboxes": [], "confidences": []}

    # Dimensions de l'image originale
    image_height, image_width = image.shape[:2]

    # Détecter les blocs avec récursion
    blocks, blocks_paths = extract_and_upscale_blocks(
        image_path,
        scale_factor=3.0,  # Facteur d'agrandissement (3x)
        upscale_method="lanczos4",  # Méthode optimisée pour le texte
        sharpen=True,  # Appliquer un filtre de netteté
        save_quality=100,  # Qualité maximale pour JPEG
    )

    # Initialiser le dictionnaire de résultats
    extracted_data: dict[str, list[Any]] = {
        "words": [],  # List[str] à la rigueur si vous voulez être plus précis
        "bboxes": [],  # List[Tuple[int, int, int, int]]
        "confidences": [],  # List[int] ou List[float]
    }

    # Facteur de mise à l'échelle pour compenser l'agrandissement
    scale_factor = 3.0

    for block, block_path in zip(blocks, blocks_paths, strict=False):
        # Extraire les coordonnées du bloc
        x, y, w, h = block

        # Appliquer l'OCR avec Tesseract
        block_data = pytesseract.image_to_data(
            block_path, output_type=pytesseract.Output.DICT, config=tesseract_config
        )

        # Parcourir les mots détectés
        for j in range(len(block_data["text"])):
            # Ignorer les entrées vides
            if not block_data["text"][j].strip():
                continue

            # Récupérer les coordonnées locales au bloc (ajustées pour l'échelle)
            local_x = int(block_data["left"][j] / scale_factor)
            local_y = int(block_data["top"][j] / scale_factor)
            local_w = int(block_data["width"][j] / scale_factor)
            local_h = int(block_data["height"][j] / scale_factor)

            # Convertir en coordonnées globales
            global_x = x + local_x
            global_y = y + local_y

            # Créer la bbox au format x1,y1,x2,y2 (SANS inverser les coordonnées Y)
            bbox_tuple = (global_x, image_height - (global_y + local_h), global_x + local_w, image_height - global_y)

            # Ajouter aux données
            extracted_data["words"].append(block_data["text"][j])
            extracted_data["bboxes"].append(bbox_tuple)
            extracted_data["confidences"].append(block_data["conf"][j])

    return extracted_data


def is_bbox_overlap(bbox1, bbox2, threshold=0.5):
    """Determines if two bounding boxes overlap significantly.

    Checks whether two bounding boxes overlap with a given threshold, considering
    the overlap ratio from both boxes' perspectives. This is useful for matching
    text regions between different image versions or annotations.

    Args:
        bbox1: First bounding box as [x1, y1, x2, y2].
        bbox2: Second bounding box as [x1, y1, x2, y2].
        threshold: Minimum overlap ratio required (0.0-1.0). The function takes the
            maximum ratio between (intersection/area1) and (intersection/area2),
            providing a more permissive matching. Default is 0.5.

    Returns:
        bool: True if the overlap ratio exceeds the threshold, False otherwise.
    """
    # Extraire les coordonnées
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculer la surface de l'intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right <= x_left or y_bottom <= y_top:
        return False  # Pas de chevauchement

    intersection = (x_right - x_left) * (y_bottom - y_top)

    # Calculer les surfaces des deux bboxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Éviter la division par zéro
    if area1 == 0 or area2 == 0:
        return False

    # Calculer les deux ratios de chevauchement
    ratio1 = intersection / area1
    ratio2 = intersection / area2

    # Utiliser le maximum des deux ratios (plus permissif)
    # On pourrait aussi utiliser min(ratio1, ratio2) pour être plus strict
    overlap_ratio = max(ratio1, ratio2)

    return overlap_ratio >= threshold


def get_image_dims(image_path):
    """Returns the width and height of an image, or (0, 0) if image cannot be read.
    Safe function that handles all error cases without raising exceptions.
    """  # noqa: D205
    if not image_path or not os.path.exists(image_path):
        return 0, 0

    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0, 0
        h, w = img.shape[:2]
        return w, h  # Return width, height
    except Exception as e:
        print(f"exception in get_image_dims due to {e}")
        return 0, 0


def process_ocr(image_path):
    """Traite l'OCR en découpant d'abord l'image en blocs puis en appliquant l'OCR sur chaque bloc."""
    try:
        tesseract_config = r"--oem 3 --psm 6 -l fra"  # config pour le français et les documents administratifs
        extracted_data = process_image_with_block_ocr(
            image_path,
            tesseract_config=tesseract_config,
        )
        converted_data = convert_nested_numpy_types(extracted_data)
        return converted_data

    except Exception as e:
        print(f"Erreur lors de l'extraction avec Tesseract: {e}")


def normalize_bboxes(bboxes, image_size, scale=1000):
    """Normaliser les bounding boxes à l'échelle 0-1000 (format attendu par LayoutLMv3).

    Args:
        bboxes: Liste de bounding boxes au format [x1, y1, x2, y2]
        image_size: Tuple (width, height) de l'image
        scale: Facteur d'échelle (1000 par défaut pour LayoutLMv3)

    Returns:
        Liste de bounding boxes normalisées à l'échelle 0-1000
    """
    width, height = image_size
    normalized_bboxes = []

    for box in bboxes:
        # Normaliser à l'échelle 0-1000
        normalized_box = [
            int(box[0] / width * scale),
            int(box[1] / height * scale),
            int(box[2] / width * scale),
            int(box[3] / height * scale),
        ]
        normalized_bboxes.append(normalized_box)

    return normalized_bboxes


def predict_with_model(image_path: str) -> tuple[list[str], list[tuple[int, int, int, int]], list[str]]:
    """Run model inference on an image: OCR → preprocess → LayoutLMv3 prediction.

    1. Lit l'image et applique plusieurs passes d'OCR (Tesseract, block-based).
    2. Normalise les boîtes au format attendu (0–1000).
    3. Encode et exécute le modèle en mode évaluation.
    4. Extrait les labels pour chaque mot détecté.

    Args:
        image_path (str): Chemin vers le fichier image à traiter.

    Returns:
        Tuple[List[str], List[tuple[int, int, int, int]], List[str]]:
            - words: liste des mots détectés (même ordre que dans l'OCR).
            - boxes: liste des bounding boxes normalisées (x1, y1, x2, y2).
            - predicted_labels: liste des labels prédits pour chaque mot.

    Raises:
        Exception: Si une erreur survient lors de la lecture d'image,
            de l'OCR ou de l'inférence, imprime la stacktrace et renvoie
            trois listes vides.
    """
    assert processor is not None
    assert model is not None

    try:
        # Extraire le texte et les bounding boxes avec plusieurs OCR
        ocr_data = process_ocr(image_path)

        if not ocr_data["words"]:
            print("No OCR data detected")
            return [], [], []

        # Lire l'image originale
        img = PILImage.open(image_path).convert("RGB")  # Forcer le format RGB
        img_np = np.array(img)  # Convertir en numpy pour s'assurer que le format est correct
        img = PILImage.fromarray(img_np)  # Reconvertir en PIL Image
        width, height = img.size

        # Normaliser les bounding boxes
        norm_boxes = normalize_bboxes(ocr_data["bboxes"], (width, height))

        # Préparer les données pour le modèle - même format que le script d'entraînement
        encoding = processor(
            images=img,
            text=[ocr_data["words"]],
            boxes=[norm_boxes],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Déplacer les données vers le device
        for k, v in encoding.items():
            encoding[k] = v.to(device)

        # Effectuer la prédiction en mode évaluation
        with torch.no_grad():
            outputs = model(**encoding)

        # Traiter les prédictions - même méthode que compute_metrics dans le script d'entraînement
        predictions = outputs.logits.argmax(dim=2)

        # Récupérer les labels prédits
        predicted_labels = []
        word_ids = encoding.word_ids(batch_index=0)

        # Extraire les prédictions pour chaque mot (premier token de chaque mot)
        prev_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                label_id = predictions[0, idx].item()
                predicted_labels.append(id2label[label_id])
                prev_word_id = word_id

        # S'assurer que nous avons le même nombre de prédictions que de mots
        words = ocr_data["words"]
        boxes = ocr_data["bboxes"]

        # En cas de désynchronisation entre le nombre de mots et de prédictions
        min_len = min(len(words), len(predicted_labels))
        words = words[:min_len]
        boxes = boxes[:min_len]
        predicted_labels = predicted_labels[:min_len]

        print(f"Predicted {len(predicted_labels)} labels for {len(words)} words")

        return words, boxes, predicted_labels

    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        import traceback

        traceback.print_exc()
        return [], [], []


def merge_bio_boxes(
    words: list[str],
    boxes: list[tuple[int, int, int, int]],
    labels: list[str],
) -> tuple[list[str], list[tuple[int, int, int, int]], list[str]]:
    """Merge consecutive B-/I- boxes for the same entity type.

    Parcourt la séquence de labels au format BIO et fusionne les tokens B- + I-
    consécutifs pour n'en faire qu'une seule entité.

    Args:
        words (List[str]): Liste de tokens OCR.
        boxes (List[tuple[int, int, int, int]]): Boîtes englobantes pour chaque token.
        labels (List[str]): Liste de labels BIO, ex. 'B-PER', 'I-PER', 'O'.

    Returns:
        Tuple[List[str], List[tuple[int, int, int, int]], List[str]]:
            - merged_words: mots ou groupes de mots fusionnés.
            - merged_boxes: boîtes couvrant l'ensemble de l'entité fusionnée.
            - merged_labels: type d'entité sans préfixe BIO, ex. 'PER'.
    """
    if not words or not boxes or not labels:
        return [], [], []

    merged_words = []
    merged_boxes = []
    merged_labels = []

    i = 0
    while i < len(labels):
        current_label = labels[i]

        # Si le label actuel est "O" ou ne commence pas par "B-", l'ajouter simplement
        if current_label == "O":
            merged_words.append(words[i])
            merged_boxes.append(boxes[i])
            merged_labels.append(current_label)
            i += 1
            continue

        # Si c'est un début d'entité (B-)
        if current_label.startswith("B-"):
            entity_type = current_label[2:]  # Extraire le type d'entité
            entity_words = [words[i]]

            # Initialiser la boîte fusionnée avec la boîte courante
            x1, y1, x2, y2 = boxes[i]

            # Chercher les tokens I- consécutifs pour cette entité
            j = i + 1
            while j < len(labels) and labels[j].startswith("I-") and labels[j][2:] == entity_type:
                entity_words.append(words[j])

                # Mettre à jour la boîte fusionnée pour englober la boîte courante
                bx1, by1, bx2, by2 = boxes[j]
                x1 = min(x1, bx1)
                y1 = min(y1, by1)
                x2 = max(x2, bx2)
                y2 = max(y2, by2)

                j += 1

            # Ajouter l'entité fusionnée
            merged_words.append(" ".join(entity_words))
            merged_boxes.append((x1, y1, x2, y2))
            merged_labels.append(entity_type)  # Stocker le type d'entité sans le préfixe B-/I-

            # Passer à la prochaine entité
            i = j
        else:
            # Si c'est un I- orphelin, le traiter comme un token séparé
            if current_label.startswith("I-"):
                entity_type = current_label[2:]
                merged_words.append(words[i])
                merged_boxes.append(boxes[i])
                merged_labels.append(entity_type)  # Stocker le type d'entité sans le préfixe I-
            # Sinon (cas peu probable), le conserver tel quel
            else:
                merged_words.append(words[i])
                merged_boxes.append(boxes[i])
                merged_labels.append(current_label)

            i += 1

    return merged_words, merged_boxes, merged_labels


def post_process_merged_entities(
    merged_words: list[str | None],
    merged_boxes: list[tuple[int, int, int, int] | None],
    merged_labels: list[str | None],
) -> tuple[list[str], list[tuple[int, int, int, int]], list[str]]:
    """Post-process MBIO entities to correct overlaps and refine predictions.

    Parcourt les entités fusionnées (avec des `None` marquant celles à supprimer),
    regroupe celles du même type très proches ou chevauchantes,
    puis filtre les éléments marqués pour suppression.

    Args:
        merged_words (List[Optional[str]]): Liste de mots fusionnés ou `None`.
        merged_boxes (List[Optional[Tuple[int, int, int, int]]]): Boîtes ou `None`.
        merged_labels (List[Optional[str]]): Labels ou `None`.

    Returns:
        Tuple[
            List[str],
            List[Tuple[int, int, int, int]],
            List[str],
        ]:
            - filtered_words: mots gardés (sans `None`).
            - filtered_boxes: boîtes correspondantes.
            - filtered_labels: labels correspondants.
    """
    if not merged_words:
        return [], [], []

    # Dictionnaire label -> liste de (mot, boîte, index)
    entities_by_type: dict[str, list[tuple[str, tuple[int, int, int, int], int]]] = {}

    for i, (word, box, label) in enumerate(zip(merged_words, merged_boxes, merged_labels, strict=False)):
        # Ignorer None et "O"
        if label is None or label == "O":
            continue
        if word is None or box is None:
            continue

        entities_by_type.setdefault(label, []).append((word, box, i))

    # Fusionner les entités proches du même type
    for label, entities in entities_by_type.items():  # noqa: B007
        if len(entities) <= 1:
            continue

        entities.sort(key=lambda x: x[1][0])  # Trier par x1
        j = 0
        while j < len(entities) - 1:
            w1, b1, idx1 = entities[j]
            w2, b2, idx2 = entities[j + 1]

            # Distance horizontale
            h_dist = b2[0] - b1[2]
            if h_dist < 50:
                # Fusion
                new_word = f"{w1} {w2}"
                new_box = (
                    min(b1[0], b2[0]),
                    min(b1[1], b2[1]),
                    max(b1[2], b2[2]),
                    max(b1[3], b2[3]),
                )
                entities[j] = (new_word, new_box, idx1)

                # Marquer la suivante pour suppression
                merged_words[idx2] = None
                merged_boxes[idx2] = None
                merged_labels[idx2] = None

                entities.pop(j + 1)
            else:
                j += 1

    # Filtrer les None et reconstruire les listes finales
    filtered_words: list[str] = []
    filtered_boxes: list[tuple[int, int, int, int]] = []
    filtered_labels: list[str] = []

    for w, b, l in zip(merged_words, merged_boxes, merged_labels, strict=False):
        if w is not None and b is not None and l is not None:
            filtered_words.append(w)
            filtered_boxes.append(b)
            filtered_labels.append(l)

    return filtered_words, filtered_boxes, filtered_labels


def process_image(image_path):
    """Traite l'image et crée une visualisation avec les entités détectées."""
    # Prédire les labels
    words, boxes, labels = predict_with_model(image_path)

    if not words:
        return None, "Aucun texte détecté dans l'image."

    # Fusionner les boîtes BIO
    merged_words, merged_boxes, merged_labels = merge_bio_boxes(words, boxes, labels)

    # Post-traitement pour améliorer les résultats
    merged_words, merged_boxes, merged_labels = post_process_merged_entities(merged_words, merged_boxes, merged_labels)

    # Lire l'image originale
    img = PILImage.open(image_path)
    img_arr = np.array(img)
    img_height, img_width = img_arr.shape[:2]

    # Créer une figure pour la visualisation
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_arr)
    ax.axis("off")

    # Dictionnaire pour stocker les entités trouvées
    found_entities = {}

    # Ajouter les boîtes fusionnées
    for word, box, label in zip(merged_words, merged_boxes, merged_labels, strict=False):
        # Ne pas afficher les labels "O"
        if label == "O":
            continue

        # Trouver la couleur appropriée pour ce type de label
        color = get_color_for_label(label)

        x1, y1, x2, y2 = box

        # Ajouter l'entité au dictionnaire
        if label not in found_entities:
            found_entities[label] = []
        found_entities[label].append(word)

        # Créer un rectangle
        rect = Rectangle(
            (x1, img_height - y2),  # Inverser y2 car l'axe y de matplotlib part du bas
            (x2 - x1),
            (y2 - y1),
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            alpha=0.7,
        )
        ax.add_patch(rect)

        # Ajouter un label
        ax.text(
            x1,
            img_height - y1,  # Inverser y1 car l'axe y de matplotlib part du bas
            f"{label}",
            color="white",
            fontsize=10,
            bbox={"facecolor": color, "alpha": 0.7},
        )

    # Sauvegarder la figure en tant qu'image
    fig.tight_layout()
    os.makedirs("temp_results", exist_ok=True)
    result_path = os.path.join("temp_results", os.path.basename(image_path))
    plt.savefig(result_path, bbox_inches="tight")
    plt.close(fig)

    # Convertir en base64 pour l'affichage HTML
    buffer = io.BytesIO()
    plt.figure(figsize=(10, 10))
    plt.imshow(plt.imread(result_path))
    plt.axis("off")
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close()
    buffer.seek(0)

    # Encoder en base64
    b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Cleanup temp folders
    for folder in ["temp_results", "temp_uploads"]:
        if os.path.isdir(folder):
            shutil.rmtree(folder)

    return b64_image, found_entities


# Routes Flask
@app.route("/")
def index():
    """Render the inference page template.

    Affiche la page HTML pour charger une image et lancer l'inférence.

    Returns:
        flask.Response: Template HTML 'index_inference.html'.
    """
    return render_template("index_inference.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload, image processing and return JSON response.

    Vérifie la présence et le nom du fichier, le sauvegarde, le traite (OCR + prédiction
    + visualisation) puis renvoie le résultat encodé en base64 et les entités détectées.

    Returns:
        flask.Response: Objet Response JSON contenant soit :
            - success (bool): True si tout s'est bien passé.
            - image (str): Image annotée encodée en base64.
            - entities (List[Dict[str, Any]]): Liste des entités détectées,
              chacune avec 'label', 'words', 'color'.
            - error (str): Message d'erreur en cas d'échec.
    """
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier téléchargé"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné"})

    if file:
        # Sauvegarder le fichier
        os.makedirs("temp_uploads", exist_ok=True)
        file_path = os.path.join("temp_uploads", file.filename)
        file.save(file_path)

        # Traiter l'image
        try:
            b64_image, entities = process_image(file_path)

            if not b64_image:
                return jsonify({"error": "Aucun texte détecté dans l'image"})

            # Préparer les résultats
            results = []
            for label, words in entities.items():
                results.append({"label": label, "words": words, "color": get_color_for_label(label)})

            return jsonify({"success": True, "image": b64_image, "entities": results})

        except Exception as e:
            import traceback

            traceback.print_exc()
            return jsonify({"error": f"Erreur lors du traitement de l'image: {str(e)}"})

    return jsonify({"error": "Une erreur est survenue lors du téléchargement du fichier"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app pour l'inférence OCR et LayoutLMv3.")
    parser.add_argument("--model_path", required=True, help="Chemin vers le dossier contenant le modèle entraîné")
    args = parser.parse_args()
    initialize_model(args.model_path)
    print("Starting Text Annotation Tool on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)

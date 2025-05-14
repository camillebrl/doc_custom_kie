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


"""Text Annotation Tool for Document Processing.

This module implements a Flask web application that allows users to upload documents,
perform OCR using block-based extraction, annotate text regions with custom labels,
and export the annotations in JSONL format for machine learning training.

The tool supports image processing with a customized OCR, data augmentation, clean labelisation to help
build robust training datasets for document understanding models.
"""

import os
import random
from pathlib import Path
from typing import Any, TypedDict

import cv2
import jsonlines
import Levenshtein
import numpy as np
import pytesseract
from flask import Flask, jsonify, render_template, request, send_from_directory
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

app = Flask(__name__)

# Temporary storage directories
TEMP_DIR = Path("temp_images")
TEMP_DIR.mkdir(exist_ok=True)
MARKED_DIR = Path("temp_marked")
MARKED_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)


class OcrResultEntry(TypedDict):
    """Type definition for storing OCR processing results.

    This TypedDict defines the structure for results returned by OCR processing
    functions. It captures all necessary information about recognized text:
    the words themselves, their positions in the image, and confidence scores.

    Attributes:
        words: List of strings containing the recognized text words.
        bboxes: List of bounding box coordinates as tuples (x1, y1, x2, y2),
            where each tuple represents the position of a word in the image.
            Coordinates follow the format:
            - x1: Left coordinate
            - y1: Top coordinate
            - x2: Right coordinate
            - y2: Bottom coordinate
        confidences: List of float values representing the confidence score
            (typically 0-100) for each recognized word.
    """

    words: list[str]
    bboxes: list[tuple[int, int, int, int]]
    confidences: list[float]


class Dimensions(TypedDict):
    """Type definition for storing image dimensions.

    This TypedDict defines a simple structure for representing the width and height
    of an image in pixels.

    Attributes:
        width: Integer representing the image width in pixels.
        height: Integer representing the image height in pixels.
    """

    width: int
    height: int


class TextRegion(TypedDict, total=False):
    """Type definition for storing text region annotations.

    This TypedDict defines the structure for annotated text regions in an image.
    The 'total=False' parameter indicates that all fields are optional, which
    allows for flexibility in different annotation contexts.

    Attributes:
        bbox: Tuple of 4 integers (x1, y1, x2, y2) representing the bounding box
            coordinates of the text region.
        text: String containing the recognized or annotated text.
        label: String representing the semantic label assigned to this text
            (e.g., "NAME", "ADDRESS", "TOTAL", etc.).
        match_confidence: Float value representing the confidence score when this
            annotation was matched or transferred from another image. This field
            is only present for annotations that were automatically generated through
            matching algorithms.
    """

    bbox: tuple[int, int, int, int]
    text: str
    label: str
    match_confidence: float  # présent uniquement sur certaines annotations


class ImageAnnotations(TypedDict):
    """Type definition for storing all annotations related to a single image.

    This TypedDict defines the structure for the complete set of annotations
    for an image, including its dimensions and all text regions.

    Attributes:
        dimensions: A Dimensions object containing the width and height of the image.
        text_regions: List of TextRegion objects representing all annotated
            text areas in the image.
    """

    dimensions: Dimensions
    text_regions: list[TextRegion]


# Dictionary to store OCR results
ocr_results: dict[str, OcrResultEntry] = {}

# Path for storing jsonl annotations
JSONL_ANNOT_FILE = Path("temp_annot.jsonl")

# Initialiser les annotations en mémoire avec un dictionnaire vide
annotations: dict[str, ImageAnnotations] = {}


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


def clear_annotation_files():
    """Clear annotation files on startup."""
    # Vider le fichier JSONL
    with open(JSONL_ANNOT_FILE, "w") as f:
        f.write("")

    print("Cleared annotation files on startup")


def get_existing_files():
    """Retrieves all supported image files from the temporary directory.

    Scans the temporary directory for PNG, JPG, and JPEG files and returns
    their paths as a list of strings. Returns an empty list if the directory
    doesn't exist.

    Returns:
        list: Paths to all image files found in the temporary directory.
    """
    if not TEMP_DIR.exists():
        return []

    files = []
    for file in TEMP_DIR.glob("*"):
        if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            files.append(str(file))
    return files


def save_document_only(file):
    """Save uploaded file to temp_images.
    Convert PDFs to PNG pages.
    Do NOT process OCR.
    Return list of image filepaths.
    """  # noqa: D205
    if not file:
        return []

    saved = []
    filename = secure_filename(file.filename)
    dst = TEMP_DIR / filename
    file.save(dst)

    if dst.suffix.lower() == ".pdf":
        pages = convert_from_path(dst)
        for i, page in enumerate(pages, start=1):
            img_name = f"{dst.stem}_page{i}.png"
            img_path = TEMP_DIR / img_name
            page.save(img_path, "PNG")
            saved.append(str(img_path))
    else:
        saved.append(str(dst))

    return saved


def apply_augmentation(image_path, transform_idx, aug_id):
    """Applique une augmentation à une image et sauvegarde le résultat.

    Args:
        image_path: Chemin de l'image originale
        transform_idx: Index de la transformation à appliquer
        aug_id: Identifiant unique pour cette augmentation

    Returns:
        Chemin de l'image augmentée ou None en cas d'erreur
    """
    try:
        # Lire l'image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erreur: Impossible de lire l'image {image_path}")
            return None

        # Pour cette version simplifiée, on applique juste quelques transformations basiques
        # plutôt que d'utiliser les transformations albumentations

        # Liste de transformations simples de base
        transformations = [
            # 0. Augmenter le contraste
            lambda image: cv2.convertScaleAbs(image, alpha=1.3, beta=10),
            # 1. Ajouter du bruit gaussien
            lambda image: cv2.add(image, np.random.normal(0, 15, image.shape).astype(np.uint8)),
            # 2. Rotation légère
            lambda image: cv2.warpAffine(
                image,
                cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), random.uniform(-3, 3), 1.0),
                (image.shape[1], image.shape[0]),
            ),
            # 3. Flou léger
            lambda image: cv2.GaussianBlur(image, (3, 3), 0),
            # 4. Distorsion de perspective
            lambda image: apply_perspective_transform(image),
            # 5. Ajustement de luminosité
            lambda image: cv2.convertScaleAbs(image, alpha=random.uniform(0.8, 1.2), beta=random.uniform(-10, 10)),
        ]

        # Choisir la transformation en fonction de l'index
        transform = transformations[transform_idx % len(transformations)]

        # Appliquer la transformation
        augmented = transform(img)

        # Créer un nom pour l'image augmentée
        path = Path(image_path)
        aug_filename = f"{path.stem}_aug{aug_id}{path.suffix}"
        aug_path = TEMP_DIR / aug_filename

        # Sauvegarder l'image augmentée
        cv2.imwrite(str(aug_path), augmented)
        print(f"Image augmentée sauvegardée: {aug_path}")

        return str(aug_path)

    except Exception as e:
        print(f"Erreur lors de l'augmentation de l'image: {e}")
        import traceback

        traceback.print_exc()
        return None


def apply_perspective_transform(image):
    """Applies a random moderate perspective transformation to an image.

    Simulates document skew or imperfect scanning by randomly shifting each corner
    of the image by 0-30 pixels. Useful for data augmentation in OCR model training
    to improve robustness against perspective variations while preserving text readability.

    Args:
        image: Input image (numpy array)

    Returns:
        Transformed image with same dimensions as input
    """
    h, w = image.shape[:2]

    # Définir les points source (les coins de l'image)
    src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])

    # Définir les points destination (légèrement déplacés)
    dst_points = np.float32(
        [
            [np.random.randint(0, 30), np.random.randint(0, 30)],
            [w - 1 - np.random.randint(0, 30), np.random.randint(0, 30)],
            [np.random.randint(0, 30), h - 1 - np.random.randint(0, 30)],
            [w - 1 - np.random.randint(0, 30), h - 1 - np.random.randint(0, 30)],
        ]
    )

    # Calculer la matrice de transformation
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Appliquer la transformation
    transformed = cv2.warpPerspective(image, M, (w, h))

    return transformed


def generate_augmentations(image_path, n_augmentations=5):
    """Génère des versions augmentées d'une image et réapplique l'OCR.

    Args:
        image_path: Chemin de l'image originale
        n_augmentations: Nombre d'augmentations à générer

    Returns:
        Liste des chemins des images augmentées
    """
    # Vérifier si l'image existe
    if not os.path.exists(image_path):
        print(f"L'image {image_path} n'existe pas.")
        return []

    # Vérifier si nous avons des annotations pour cette image
    if image_path not in annotations or "text_regions" not in annotations[image_path]:
        print(f"Aucune annotation trouvée pour {image_path}")
        return []

    # Récupérer les annotations originales
    original_annotations = annotations[image_path]["text_regions"]
    if not original_annotations:
        print(f"Aucune annotation trouvée pour {image_path}")
        return []

    augmented_paths = []

    # Générer les augmentations
    for i in range(n_augmentations):
        # Générer un ID unique pour cette augmentation
        aug_id = str(i + 1)

        # Appliquer une augmentation aléatoire
        aug_path = apply_augmentation(image_path, i, aug_id)
        if not aug_path:
            continue

        augmented_paths.append(aug_path)

        # Traiter l'OCR pour l'image augmentée
        success, _ = process_ocr(aug_path)
        if not success or aug_path not in ocr_results:
            print(f"Échec de l'OCR pour {aug_path}")
            continue

        # Initialiser les annotations pour l'image augmentée
        if aug_path not in annotations:
            # Récupérer les dimensions de l'image augmentée
            img = cv2.imread(aug_path)
            if img is None:
                continue
            height, width = img.shape[:2]

            annotations[aug_path] = {"dimensions": {"width": width, "height": height}, "text_regions": []}

        # Transférer les labels des annotations originales vers les nouveaux mots OCR
        transfer_annotations(image_path, aug_path)

        # Sauvegarder les annotations au format JSONL
        save_to_jsonl(aug_path)

    return augmented_paths


def transfer_annotations(original_path, augmented_path):
    """Transfère les annotations de l'image originale vers l'image augmentée en utilisant
    la correspondance par distance de Levenshtein entre les textes OCR.
    """  # noqa: D205
    if original_path not in ocr_results or augmented_path not in ocr_results:
        return False

    augmented_ocr = ocr_results[augmented_path]

    # Récupérer les annotations originales
    original_annotations = annotations[original_path]["text_regions"]

    # Dictionnaire pour stocker les labels par texte
    text_to_label = {}

    # Construire une correspondance texte -> label à partir des annotations originales
    for ann in original_annotations:
        text = ann["text"]
        label = ann.get("label", "O")
        text_to_label[text] = label

    # Seuil de similarité accepté pour la correspondance (plus petit = plus strict)
    SIMILARITY_THRESHOLD = 0.2  # 20% de différence maximale tolérée

    # Correspondance par similarité de texte
    for i, (aug_word, aug_bbox) in enumerate(zip(augmented_ocr["words"], augmented_ocr["bboxes"], strict=False)):
        best_match = None
        best_ratio = 0

        # Chercher le mot le plus similaire dans les mots originaux annotés
        for orig_text, label in text_to_label.items():
            # Calculer la similarité avec la distance de Levenshtein normalisée
            length = max(len(aug_word), len(orig_text))
            if length == 0:  # Éviter la division par zéro
                continue

            # Calculer la distance de Levenshtein
            distance = Levenshtein.distance(aug_word.lower(), orig_text.lower())

            # Convertir la distance en ratio de similarité (1 = identique, 0 = complètement différent)
            similarity_ratio = 1 - (distance / length)

            # Garder le meilleur match
            if similarity_ratio > best_ratio and similarity_ratio > (1 - SIMILARITY_THRESHOLD):
                best_ratio = similarity_ratio
                best_match = (orig_text, label)

        # Si un match suffisamment bon a été trouvé, ajouter l'annotation
        if best_match:
            orig_text, label = best_match

            # Ajouter une annotation pour ce mot dans l'image augmentée
            annotations[augmented_path]["text_regions"].append(
                {
                    "bbox": aug_bbox,
                    "text": aug_word,  # Utiliser le texte OCR de l'image augmentée
                    "label": label,  # Mais le label du texte original
                    "match_confidence": best_ratio,  # Stocker le niveau de confiance pour référence
                }
            )

            print(f"Match trouvé: '{aug_word}' -> '{orig_text}' (confiance: {best_ratio:.2f}, label: {label})")

    return True


def draw_annotations(image_path, current_bbox=None):
    """Renders annotations on an image for visualization.

    This function draws bounding boxes around text regions detected by OCR and user annotations.
    All boxes are drawn in black and white for a cleaner, more uniform appearance without color-coding.

    Args:
        image_path (str): Path to the image file to be annotated
        current_bbox (list, optional): Current selection box coordinates [x1, y1, x2, y2]

    Returns:
        str: Filename of the marked image saved in the static directory, or None if processing failed
    """
    if not image_path or not os.path.exists(image_path):
        return None

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width = img.shape[:2]

    # Store dimensions if not already stored
    if image_path in annotations:
        if "dimensions" not in annotations[image_path]:
            annotations[image_path]["dimensions"] = {"width": width, "height": height}
    else:
        annotations[image_path] = {"dimensions": {"width": width, "height": height}, "text_regions": []}

    # Draw OCR detected text boxes (light gray)
    if image_path in ocr_results:
        for bbox in ocr_results[image_path]["bboxes"]:
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Get all annotations
    all_annotations = []
    if image_path in annotations and "text_regions" in annotations[image_path]:
        all_annotations = annotations[image_path]["text_regions"]

    # Draw existing annotations (all in black & white)
    for ann in all_annotations:
        bbox = ann["bbox"]
        text = ann["text"]
        label = ann.get("label", "O")

        # Use black for all annotations
        color = (0, 0, 0)

        # Draw semi-transparent highlight for text
        overlay = img_rgb.copy()
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, img_rgb, 0.7, 0, img_rgb)

        # Draw border around text
        cv2.rectangle(img_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Draw text and label on top
        display_text = text
        if label and label != "O":
            display_text = f"{text} [{label}]"
        cv2.putText(img_rgb, display_text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw current selection
    if current_bbox:
        x1, y1, x2, y2 = current_bbox
        overlay = img_rgb.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), -1)  # Gray instead of green
        cv2.addWeighted(overlay, 0.3, img_rgb, 0.7, 0, img_rgb)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (50, 50, 50), 1)  # Darker gray border

    # Save marked image to static directory for display
    output_filename = f"marked_{Path(image_path).name}"
    output_path = STATIC_DIR / output_filename

    # Convert back to BGR for saving
    cv2.imwrite(str(output_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    return output_filename


def add_annotation(image_path, text_content, x1, y1, x2, y2, label="O", save_to_file=False):
    """Add text annotation with coordinates and label
    If save_to_file is True, also save to JSON and JSONL files.
    """  # noqa: D205
    if not image_path or not text_content:
        return False, "No image or text specified."

    try:
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        bbox = [x1, y1, x2, y2]

        # Initialize image annotations if not exist
        if image_path not in annotations:
            # Get image dimensions
            img = cv2.imread(image_path)
            if img is None:
                return False, "Cannot read image."
            height, width = img.shape[:2]

            annotations[image_path] = {"dimensions": {"width": width, "height": height}, "text_regions": []}
        elif "text_regions" not in annotations[image_path]:
            annotations[image_path]["text_regions"] = []

        # Add new text region
        annotations[image_path]["text_regions"].append({"bbox": bbox, "text": text_content, "label": label})

        # Save to file ONLY if explicitly requested
        if save_to_file:
            # Save to JSONL
            save_to_jsonl(image_path)

        # Draw annotations on image
        draw_annotations(image_path)

        return True, f"Saved annotation for {Path(image_path).name}."
    except Exception as e:
        return False, f"Error saving annotation: {str(e)}"


def save_to_jsonl(image_path=None):
    """Sauvegarde les annotations au format JSONL attendu pour l'entraînement
    Si image_path est None, sauvegarder toutes les images.
    """  # noqa: D205
    # Si aucun chemin d'image n'est fourni, traiter toutes les images
    if image_path is None:
        # Traiter chaque image annotée
        for img_path in annotations:
            save_single_image_to_jsonl(img_path)

        return True, f"Saved annotations for {len(annotations)} images to {JSONL_ANNOT_FILE}"
    else:
        # Traiter uniquement l'image spécifiée
        if image_path not in annotations or "text_regions" not in annotations[image_path]:
            return False, f"No annotations found for {image_path}"

        save_single_image_to_jsonl(image_path)
        return True, f"Saved annotations for {image_path} to {JSONL_ANNOT_FILE}"


def save_single_image_to_jsonl(image_path):
    """Sauvegarde les annotations d'une seule image au format JSONL
    avec prise en charge des tags BIO.
    """  # noqa: D205
    if image_path not in annotations or "text_regions" not in annotations[image_path]:
        return

    # Récupérer les dimensions de l'image
    width = annotations[image_path]["dimensions"]["width"]
    height = annotations[image_path]["dimensions"]["height"]
    image_size = (width, height)

    # Préparer les données
    words = []
    bboxes = []
    labels = []

    # Si nous avons des résultats OCR pour cette image
    if image_path in ocr_results:
        # Utiliser tous les mots de l'OCR comme base
        words = ocr_results[image_path]["words"]
        bboxes = ocr_results[image_path]["bboxes"]
        # Par défaut, tous les mots sont étiquetés "O" (Outside)
        labels = ["O"] * len(words)

        # Mettre à jour les labels pour les mots annotés
        for region in annotations[image_path]["text_regions"]:
            region_text = region["text"]
            region_bbox = region["bbox"]
            region_label = region.get("label", "O")

            # Trouver le mot correspondant dans les résultats OCR
            for i, (word, bbox) in enumerate(zip(words, bboxes, strict=False)):
                # Vérifier si le mot correspond (exact match ou contenu dans le texte région)
                if word == region_text or (word in region_text and is_bbox_overlap(bbox, region_bbox)):
                    labels[i] = region_label
    else:
        # Si pas d'OCR, utiliser uniquement les régions annotées
        for region in annotations[image_path]["text_regions"]:
            words.append(region["text"])
            bboxes.append(region["bbox"])
            labels.append(region.get("label", "O"))
    # Normaliser les bounding boxes
    norm_boxes = normalize_bboxes(bboxes, image_size)

    # Créer le dictionnaire à sauvegarder
    jsonl_data = {"image_path": image_path, "words": words, "bboxes": norm_boxes, "labels": labels}

    # Sauvegarder au format JSONL (append)
    with jsonlines.open(JSONL_ANNOT_FILE, mode="a") as writer:
        writer.write(jsonl_data)


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
        ocr_results[image_path] = converted_data
        return True, f"Found {len(converted_data['words'])} words in {image_path} using block-based OCR"

    except Exception as e:
        print(f"Erreur lors de l'extraction avec Tesseract: {e}")
        import traceback

        traceback.print_exc()
        return False, f"Error during OCR processing: {str(e)}"


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


def clean_jsonl_file(file_path):
    """Nettoie le fichier JSONL en ne gardant que la dernière occurrence de chaque image_path
    Retourne le nombre d'entrées après nettoyage.
    """  # noqa: D205
    if not os.path.exists(file_path):
        return 0

    # Lire toutes les entrées du fichier
    entries = []
    with jsonlines.open(file_path, mode="r") as reader:
        for line in reader:
            entries.append(line)

    # Garder seulement la dernière entrée pour chaque image_path
    unique_entries = {}
    for entry in entries:
        image_path = entry.get("image_path")
        if image_path:
            unique_entries[image_path] = entry

    # Écrire les entrées uniques dans le fichier
    with jsonlines.open(file_path, mode="w") as writer:
        for entry in unique_entries.values():
            writer.write(entry)

    return len(unique_entries)


# Flask routes
@app.route("/")
def index():
    """Renders the main application page.

    Gets a list of available image files and renders the index.html template.

    Returns:
        Rendered HTML template for the main application interface.
    """
    files = get_existing_files()
    return render_template("index.html", files=files)


@app.route("/upload_only", methods=["POST"])
def upload_only():
    """Upload file without OCR processing."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"})

    try:
        saved = save_document_only(file)

        if not saved:
            return jsonify({"status": "error", "message": "Failed to save document"})

        return jsonify({"status": "success", "message": "File uploaded successfully", "files": get_existing_files()})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing file: {str(e)}"})


@app.route("/get_files", methods=["GET"])
def get_files():
    """Get list of all available files."""
    files = get_existing_files()
    return jsonify(files)


@app.route("/image/<path:filename>")
def image(filename):
    """Serves image files from the temporary directory.

    Args:
        filename: Name of the image file to serve.

    Returns:
        The requested image file from the temporary directory.
    """
    return send_from_directory(TEMP_DIR, os.path.basename(filename))


@app.route("/process_ocr/<path:filename>")
def run_ocr(filename):
    """Run OCR processing on an image."""
    if os.path.exists(filename):
        success, message = process_ocr(filename)
        if success:
            return jsonify({"status": "success", "message": message, "data": ocr_results.get(filename, {})})
        else:
            return jsonify({"status": "error", "message": message})

    return jsonify({"status": "error", "message": "File not found"})


@app.route("/annotate", methods=["POST"])
def annotate():
    """Processes annotation requests from the client.

    Handles POST requests containing annotation data (text, coordinates, label).
    Adds the annotation to the specified image and returns updated information.

    The function also handles BIO tagging format by automatically adding a "B-" prefix
    to labels that don't already have a prefix.

    Returns:
        Flask JSON response containing:
        - status: "success" or "error"
        - message: Description of the result
        - marked_file: Path to the newly annotated image (if successful)
        - annotations: List of all current annotations for the image (if successful)
    """
    data = request.json
    image_path = data.get("image_path")
    text = data.get("text")
    x1 = int(data.get("x1", 0))
    y1 = int(data.get("y1", 0))
    x2 = int(data.get("x2", 0))
    y2 = int(data.get("y2", 0))
    label = data.get("label", "O")
    save_to_file = data.get("save_to_file", False)

    # Validation for B-/I- label format
    if label != "O" and not (label.startswith("B-") or label.startswith("I-")):
        # If the label doesn't already have a B- or I- prefix, assume it's an error
        # and add B- by default
        label = "B-" + label

    # Ajouter l'annotation à l'image originale
    success, message = add_annotation(image_path, text, x1, y1, x2, y2, label, save_to_file)

    if success:
        marked_file = draw_annotations(image_path)
        current_annotations = annotations.get(image_path, {}).get("text_regions", [])

        # Ne PAS générer les augmentations ici

        return jsonify(
            {"status": "success", "message": message, "marked_file": marked_file, "annotations": current_annotations}
        )
    else:
        return jsonify({"status": "error", "message": message})


@app.route("/export_annotations", methods=["POST"])
def export_annotations():
    """Export all annotations to JSONL file."""
    try:
        success, message = save_to_jsonl(None)  # Exporter toutes les images
        if success:
            return jsonify({"status": "success", "message": message})
        else:
            return jsonify({"status": "error", "message": message})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error exporting annotations: {str(e)}"})


@app.route("/export_single_annotation", methods=["POST"])
def export_single_annotation():
    """Export annotations for a single image to JSONL file."""
    try:
        data = request.json
        image_path = data.get("image_path")

        if not image_path:
            return jsonify({"status": "error", "message": "No image path provided"})

        if image_path not in annotations:
            return jsonify({"status": "error", "message": f"No annotations found for {image_path}"})

        # Sauvegarder uniquement les annotations de l'image spécifiée dans JSONL
        success, message = save_to_jsonl(image_path)

        if success:
            return jsonify({"status": "success", "message": f"Annotations for {Path(image_path).name} saved to file."})
        else:
            return jsonify({"status": "error", "message": message})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error exporting annotation: {str(e)}"})


@app.route("/dimensions/<path:filename>")
def dimensions(filename):
    """Gets image dimensions for the specified file.

    Retrieves the width and height of an image file and returns them as a JSON response.

    Args:
        filename: Path to the image file whose dimensions should be retrieved.

    Returns:
        Flask JSON response containing the width and height of the image.
    """
    width, height = get_image_dims(filename)
    return jsonify({"width": width, "height": height})


@app.route("/annotations/<path:filename>")
def get_annotations(filename):
    """Retrieves user annotations for a given image file.

    Returns all text region annotations that have been created for the specified image.

    Args:
        filename: Path to the image file whose annotations should be retrieved.

    Returns:
        Flask JSON response containing a list of text region annotations or an empty list
        if no annotations are found.
    """
    if filename in annotations and "text_regions" in annotations[filename]:
        return jsonify(annotations[filename]["text_regions"])
    return jsonify([])


@app.route("/ocr_results/<path:filename>")
def get_ocr_results(filename):
    """Retrieves OCR results for a given file, processing OCR if needed.

    This endpoint returns OCR results for the specified file. If results are not
    available in memory, it attempts to process OCR first. Results are returned
    as a JSON response containing words and bounding boxes.

    Args:
        filename: Path to the image file for OCR processing.

    Returns:
        Flask JSON response containing OCR results (words, bounding boxes) or
        empty lists if no results are available.
    """
    if filename in ocr_results:
        return jsonify(ocr_results[filename])
    else:
        # If the results OCR are not yet available, try to process the OCR now
        if os.path.exists(filename):
            success, _ = process_ocr(filename)
            if success and filename in ocr_results:
                return jsonify(ocr_results[filename])

    return jsonify({"words": [], "bboxes": [], "page_numbers": []})


@app.route("/get_labels", methods=["GET"])
def get_labels():
    """Get all unique labels from annotations, removing B-/I- prefixes."""
    unique_base_labels = {"O"}  # Always include "O" (Outside) label

    for _, ann_data in annotations.items():
        if "text_regions" in ann_data:
            for region in ann_data["text_regions"]:
                if "label" in region and region["label"]:
                    label = region["label"]

                    # Remove B- or I- prefix if present
                    if label.startswith("B-") or label.startswith("I-"):
                        base_label = label[2:]
                        unique_base_labels.add(base_label)
                    else:
                        unique_base_labels.add(label)

    # Convert set to list for JSON serialization
    return jsonify(list(unique_base_labels))


@app.route("/add_label", methods=["POST"])
def add_label():
    """Add a new label to be used in annotations."""
    data = request.json
    new_label = data.get("label")

    if not new_label or not isinstance(new_label, str):
        return jsonify({"status": "error", "message": "Invalid label"})

    # We don't need to store labels separately as they're attached to annotations
    # Just return success
    return jsonify({"status": "success", "message": f"Added label: {new_label}"})


@app.route("/reset_annotations", methods=["POST"])
def reset_annotations():
    """Reset all annotations in memory."""
    global annotations
    annotations = {}

    # Vider également les fichiers
    clear_annotation_files()

    return jsonify({"status": "success", "message": "All annotations have been reset"})


@app.route("/check_ocr_status/<path:filename>")
def check_ocr_status(filename):
    """Vérifie si l'OCR a déjà été traité pour cette image sans relancer le traitement."""
    if filename in ocr_results and ocr_results[filename]["words"]:
        return jsonify(
            {
                "status": "success",
                "message": "OCR results already available",
                "has_ocr": True,
                "word_count": len(ocr_results[filename]["words"]),
            }
        )
    else:
        return jsonify({"status": "info", "message": "No OCR results found for this image", "has_ocr": False})


@app.after_request
def add_header(response):  # noqa: D103
    # Désactiver le cache pour toutes les réponses
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/finish", methods=["POST"])
def finish_annotation():
    """Nettoie le fichier JSONL en supprimant les doublons et arrête l'application."""
    try:
        # Nettoyer le fichier JSONL
        count = clean_jsonl_file(JSONL_ANNOT_FILE)

        # Préparer la réponse
        response = {
            "status": "success",
            "message": f"Cleaned annotation file: {count} unique entries saved.",
            "shutdown": True,
        }

        # Planifier l'arrêt de l'application
        def shutdown_server():
            import time

            time.sleep(2)  # Attendre 2 secondes pour que la réponse soit envoyée
            os._exit(0)  # Arrêter l'application

        import threading

        threading.Thread(target=shutdown_server).start()

        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during finish: {str(e)}", "shutdown": False})


@app.route("/generate_augmentations", methods=["POST"])
def generate_augments():
    """Route dédiée à la génération d'augmentations après sauvegarde dans le fichier JSONL."""
    try:
        data = request.json
        image_path = data.get("image_path")
        n_augmentations = data.get("n_augmentations", 5)

        if not image_path:
            return jsonify({"status": "error", "message": "No image path provided"})

        if image_path not in annotations:
            return jsonify({"status": "error", "message": f"No annotations found for {image_path}"})

        # Génère les augmentations
        augmented_paths = generate_augmentations(image_path, n_augmentations=n_augmentations)

        return jsonify(
            {
                "status": "success",
                "message": f"Generated {len(augmented_paths)} augmented images",
                "augmented_images": augmented_paths,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error generating augmentations: {str(e)}"})


if __name__ == "__main__":
    # Ensure directories exist
    TEMP_DIR.mkdir(exist_ok=True)
    MARKED_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)

    # Create templates directory if it doesn't exist
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    # Copy index.html to templates directory if necessary
    index_html_path = templates_dir / "index.html"

    print("Starting Text Annotation Tool on http://127.0.0.1:5001")
    app.run(debug=True, host="127.0.0.1", port=5001)

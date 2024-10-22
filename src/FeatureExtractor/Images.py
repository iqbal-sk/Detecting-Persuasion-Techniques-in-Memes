import pickle
import torch
import os
from clip import clip

import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from config.logger import get_logger

logger = get_logger(__name__)

def resnet50_features(directory, output_filepath, device):

    logger.info("Initializing ResNet50 0.")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    model.eval()  # Set the model to evaluation mode
    logger.debug("Model set to evaluation mode.")

    # Remove the last classification layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    logger.debug("Removed the last classification layer.")

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    logger.debug("Model parameters frozen.")

    model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Define a transform to preprocess the images
    logger.info("Defining image transformations.")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_dict = {}
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    total_images = len(image_files)
    logger.info(f"Found {total_images} image files in directory '{directory}'.")

    for idx, image_name in enumerate(image_files, 1):
        image_path = os.path.join(directory, image_name)
        try:
            logger.debug(f"Processing image {idx}/{total_images}: {image_name}")
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            # Add a batch dimension
            image = image.unsqueeze(0)
            image = image.to(device)

            with torch.no_grad():
                features = model(image)
            features = features.cpu().squeeze().squeeze().numpy()
            features_dict[image_name] = features

            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{total_images} images.")

        except Exception as e:
            logger.error(f"Error processing image '{image_name}': {e}")

    try:
        with open(f'{output_filepath}', 'wb') as f:
            pickle.dump(features_dict, f)
        logger.info(f"Features extracted and stored in {output_filepath}")
    except Exception as e:
        logger.error(f"Failed to save features to {output_filepath}: {e}")


def extract_clip_vit_features(directory, output_file_path, device):

    logger.info(f"Using device: {device}")

    # Load the CLIP model and preprocess
    try:
        model, preprocess = clip.load('ViT-B/32', device=device)
        logger.info("CLIP model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        return

    features_dict = {}
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    total_images = len(image_files)
    logger.info(f"Found {total_images} image files in directory '{directory}'.")

    for idx, image_name in enumerate(image_files, 1):
        image_path = os.path.join(directory, image_name)
        try:
            logger.debug(f"Processing image {idx}/{total_images}: {image_name}")
            image = Image.open(image_path).convert('RGB')

            # Preprocess the image
            image_input = preprocess(image).unsqueeze(0).to(device)
            logger.debug(f"Image '{image_name}' preprocessed.")

            with torch.no_grad():
                # Encode image using the CLIP model
                image_features = model.encode_image(image_input)
                logger.debug(f"Image '{image_name}' encoded.")

            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features_dict[image_name] = image_features.cpu().squeeze().numpy()

            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{total_images} images.")

        except Exception as e:
            logger.error(f"Error processing image '{image_name}': {e}")

    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(features_dict, f)
        logger.info(f"Features extracted and stored in {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save features to {output_file_path}: {e}")


def extract_image_features(model, directory, output_file_path, device):
    parent_directory = os.path.dirname(output_file_path)
    os.makedirs(parent_directory, exist_ok=True)
    if model == 'ResNet50':
        resnet50_features(directory, output_file_path, device)
    elif model == 'CLIP':
        extract_clip_vit_features(directory, output_file_path, device)
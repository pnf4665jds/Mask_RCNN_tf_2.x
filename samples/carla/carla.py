"""
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 carla.py train --dataset=/path/to/carla/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 carla.py train --dataset=/path/to/carla/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 carla.py train --dataset=/path/to/carla/dataset --weights=imagenet

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
from skimage import io
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import cv2
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CarlaConfig(Config):
    """Configuration for training on my Carla dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "carla"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + traffic sign

     # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CarlaDataset(utils.Dataset):
    def load_carla(self, dataset_dir, subset):
        """Load a subset of the Carla dataset.
        dataset_dir: The root directory of the Carla dataset.
        subset: What to load (train, val)
        """
        self.add_class("traffic_sign", 1, "traffic_sign")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load all file names under dataset_dir/[train or val]
        for f in os.listdir(dataset_dir):
            if os.isfile(os.path.join(dataset_dir, f)):
                imagePath = os.path.join(dataset_dir, f)
                image = io.imread(imagePath)
                height, width = image.shape[:2]

                # extract contour from mask image
                maskPath = os.path.join(os.path.join(dataset_dir, "mask"), f)
                image = io.imread(maskPath)
                image = cv2.Canny(image, 30, 150)
                contours, _ = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                self.add_image("traffic_sign", f, imagePath, width=width, height=height, contours=contours)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "traffic_sign":
            return super(self.__class__, self).load_mask(image_id)

        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["contours"])],
                        dtype=np.uint8)

        for i, cnt in enumerate(image_info["contours"]):
            # Get indexes of pixels inside the polygon and set them to 1
            pointsX, pointsY = cnt
            rr, cc = skimage.draw.polygon(pointsY, pointsX)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)  

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "traffic_sign":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CarlaDataset()
    dataset_train.load_carla(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CarlaDataset()
    dataset_val.load_carla(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

def evaluate(model):
    print("eval not implemnted")

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN for Carla')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    args = parser.parse_args()

    print("Command: ", args.command)
    print("Dataset: ", args.dataset)
    print("Weights: ", args.weights)

    # Configurations
    if args.command == "train":
        config = CarlaConfig()
    else:
        class InferenceConfig(CarlaConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        weights_path = COCO_MODEL_PATH
         # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.model.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "evaluate":
        # Validation dataset
        evaluate(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

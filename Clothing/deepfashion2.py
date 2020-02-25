"""
Usage:

    python deepfashion2.py
        train/infer/evaluate                     3种模式
        --datasetImg  preset/数据集图像具体路径            infer模式下不用填
        --datasetAnno preset/数据集标注具体路径            infer模式下不用填
        --weights coco/last/.h5文件具体路径
        --logs              存放logs和checkpoints的目录，默认值为 ./MaskRCNN_DeepFashion2_logs/
        --image                                  infer模式下使用

快速用例：
    python deepfashion2.py train --datasetImg preset --datasetAnno preset --weights coco
    python deepfashion2.py train --datasetImg preset --datasetAnno preset --weights last

    python deepfashion2.py infer --weights coco --image 【图像路径】
    python deepfashion2.py infer --weights last --image 【图像路径】
    python deepfashion2.py infer --weights 【weights】 --image 【图像路径】

    F:\MachineLearning-Datasets\DeepFashion2_Dataset\test\test\image\000001.jpg
"""


"""
使用 Mask RCNN 类的 .train() 方法：

def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
 augmentation=None,custom_callbacks=None,no_augmentation_sources=None)
 
可以通过 layers 指定被训练的层，正则表达式 / 预定义的值
预定义的值包括：
    heads: The RPN, classifier and mask heads of the network
    all: All the layers
    3+: Train ResNet stage 3 and up
    4+: Train ResNet stage 4 and up
    5+: Train ResNet stage 5 and up
 
"""

import sys
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import imgaug           # 若用到 数据增强 时，会用到这个库
from skimage import io as skio
from datetime import datetime
import tensorflow as tf

PRJ_ROOT_DIR = os.path.abspath("../")  # 工程根目录
DEFAULT_LOGS_DIR = os.path.join(PRJ_ROOT_DIR, "MaskRCNN_DeepFashion2_logs")
COCO_WEIGHTS_PATH = os.path.join(PRJ_ROOT_DIR, "mask_rcnn_coco.h5")


# 数据集中的训练集 图像 目录
DATASET_TRAIN_IMG_DIR = r'F:\MachineLearning-Datasets\DeepFashion2_Dataset\train\image\image01\image01'
# 数据集中的训练集 标注文件
DATASET_TRAIN_ANNO_FILE = r'F:\MachineLearning-Datasets\DeepFashion2_API_cache\trainImagePart1_1000.json'

# 数据集中的验证集 图像 目录
DATASET_VAL_IMG_DIR = r'F:\MachineLearning-Datasets\DeepFashion2_Dataset\validation\validation\image'
# 数据集中的验证集 标注文件
DATASET_VAL_ANNO_FILE = r'F:\MachineLearning-Datasets\DeepFashion2_API_cache\valImage_1000.json'

sys.path.append(PRJ_ROOT_DIR)
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import utils, visualize

CLASS_NAMES_EN = ['short sleeve top', 'long sleeve top', 'short sleeve outwear',
                  'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers',
                  'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress',
                  'sling dress']


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

############################################################
#  Training Configurations
############################################################


class DeepFashion2Config(Config):
    """Configuration for training on DeepFashion2.
    Derives from the base Config class and overrides values specific
    to the DeepFashion2 dataset.
    """
    # Give the configuration a recognizable name
    NAME = "deepfashion2"

    # We use a GPU with 12GB memory, which can fit two images(1024, 1024).
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13

    # TODO:考虑换成 resnet50?
    # BACKBONE = 'resnet101'

    # 以下值在这里并不需要与训练集的大小匹配。 它的大小只是用来控制保存checkpoints的频率
    # STEPS_PER_EPOCH = 1000

    # TODO:也许可以把这个改一下减小内存负载？
    # IMAGE_RESIZE_MODE = "square"
    IMAGE_MAX_DIM = 640         # 默认值为 1024
    IMAGE_MIN_DIM = 600         # 默认值为 800

    # USE_MINI_MASK = True      # 减小内存负载，默认值就是 True

    # DETECTION_MIN_CONFIDENCE = 0.7

    # LEARNING_RATE = 0.001


############################################################
#  Dataset
############################################################


class DeepFashion2Dataset(utils.Dataset):
    def load_coco(self, image_dir, json_path, class_ids=None, return_coco=False):
        """Load the DeepFashion2 dataset.
        class_ids: If provided, only loads images that have the given classes.

        return_coco: If True, returns the COCO object.
        """

        coco = COCO(json_path)

        # Load all classes or a subset?
        if not class_ids:       # class_ids is not provided
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("deepfashion2", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "deepfashion2", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_keypoint(self, image_id):
        """
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion2":
            return super(DeepFashion2Dataset, self).load_mask(image_id)

        instance_keypoints = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))
            if class_id:
                keypoint = annotation['keypoints']

                instance_keypoints.append(keypoint)
                class_ids.append(class_id)

        keypoints = np.stack(instance_keypoints, axis=1)
        class_ids = np.array(class_ids, dtype=np.int32)
        return keypoints, class_ids

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
        # If not a deepfashion2 image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "deepfashion2":
            print("MY_TAG: image_info中的source不等于deepfashion2 导致调用父类的load_mask()返回空的masks")
            return super(DeepFashion2Dataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "deepfashion2.{}".format(annotation['category_id']))

            # utils.Dataset.map_source_class_id():
            # 根据 internal class ID 返回 源 dataset 中的 class ID （即 self.class_info中的 id）

            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            print("MY_TAG: class_ids为空 导致调用父类的load_mask()返回空的masks")
            return super(DeepFashion2Dataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        super(DeepFashion2Dataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def train_maskrcnn(model, config):

    dataset_train = DeepFashion2Dataset()
    dataset_train.load_coco(DATASET_TRAIN_IMG_DIR, DATASET_TRAIN_ANNO_FILE)
    dataset_train.prepare()

    ''''''
    dataset_val = DeepFashion2Dataset()
    dataset_val.load_coco(DATASET_VAL_IMG_DIR, DATASET_VAL_ANNO_FILE)
    # TODO：注意这里临时使用了
    # dataset_val.load_coco(DATASET_TRAIN_IMG_DIR, DATASET_TRAIN_ANNO_FILE)
    dataset_val.prepare()

    """
    # 只训练 heads 的简化版训练：
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
    """

    """"""
    # 带数据增强的分阶段训练:
    # Image Augmentation: Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=8,       # 原：40
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,     # 原：120
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=12,     # 原：160
                layers='all',
                augmentation=augmentation)



def infer_image(model):
    print("Running on {}".format(args.image))

    image = skio.imread(args.image)         # 读入图像，类型为 NumPy ndarray
    r = model.detect([image], verbose=1)[0]
    '''
    .detect()方法返回一个dicts组成的list,每个dict对应一张图像, 每个dict包括：
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
    '''

    '''
    visualize.display_instances(image=image, boxes=r['rois'], masks=r['masks'],
                                class_ids=r['class_ids'], class_names=CLASS_NAMES_EN)

    '''
    visualize.display_instances(image=image, boxes=r['rois'], masks=r['masks'],
                                class_ids=r['class_ids'], class_names=CLASS_NAMES_EN,
                                scores=r['scores'])

    for i in r['class_ids']:
        print(i, CLASS_NAMES_EN[i])


if __name__ == "__main__":

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Match R-CNN for DeepFashion2.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'infer' or 'evaluate'")
    parser.add_argument("--datasetImg", required=False,
                        metavar="/path/to/deepfashion2/image/",
                        help="Image directory of the DeepFashion2 dataset or 'preset'")
    parser.add_argument("--datasetAnno", required=False,
                        metavar="/path/to/deepfashion2/anno/",
                        help="Anno directory of the DeepFashion2 dataset or 'preset'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'last' or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=./MaskRCNN_DeepFashion2_logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path to image",
                        help='The path of the image to detect in infer mode')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.datasetImg, "Argument --datasetImg is required for training"
        assert args.datasetAnno, "Argument --datasetAnno is required for training"
    elif args.command == "infer":
        assert args.image, "Provide --image to detect clothes"

    # 当训练用的图片和标注文件被指定为具体目录而非 preset 时：
    if args.datasetImg != "preset":
        DATASET_TRAIN_IMG_DIR = args.datasetImg
    if args.datasetAnno != "preset":
        DATASET_TRAIN_ANNO_FILE = args.datasetAnno

    print("Dataset Img: ", args.datasetImg)
    print("Dataset Anno: ", args.datasetAnno)
    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DeepFashion2Config()
    else:
        class InferenceConfig(DeepFashion2Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.5
        config = InferenceConfig()

    config.display()

    # Create model
    if args.command == "train":
        model = MaskRCNN(mode="training", config=config,
                         model_dir=args.logs)
    elif args.command == "infer":
        model = MaskRCNN(mode="inference", config=config,
                         model_dir=args.logs)
    elif args.command == "evaluate":
        # TODO: evaluate mode，可以复用DeepFashion2官方给出的代码
        print(" evaluate 部分的代码还未编写")
        exit(0)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'infer' or 'evaluate'".format(args.command))
        exit(-1)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            print("The COCO weights DOES NOT EXIST!")
            exit(-1)
    elif args.weights.lower() == "last":
        # Find the last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    print("正在使用的 weights: ", weights_path)

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

    # Train or infer or evaluate
    if args.command == "train":
        train_maskrcnn(model=model, config=config)
    elif args.command == "infer":
        infer_image(model)
    elif args.command == "evaluate":
        pass


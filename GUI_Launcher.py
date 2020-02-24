import os
import sys
import random
import skimage.io
import cv2

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit,\
    QFileDialog, QButtonGroup, QRadioButton, QTabWidget, QListView, QAbstractItemView
from PyQt5.QtCore import Qt, QStringListModel


# COCO 类别名称
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# 导入 Mask-RCNN
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib
from mrcnn import visualize

# 导入 COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# 保存logs和trained model的目录
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# 模型文件路径：
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


def prepare_model():
    # 配置：
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # 创建全局模型对象 model
    global model
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # 加载基于 MS-COCO 训练的模型参数
    model.load_weights(COCO_MODEL_PATH, by_name=True)


# ------------------------------------------------------------------------------- #

class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()

        self.picSourceType = ''     # 'test' 或 'custom'
        self.picFilePaths = []
        self.videoPath = ''       # 视频文件路径

        self.init_pic_ui()
        self.init_video_ui()
        self.initUI()

    def initUI(self):
        box_main = QVBoxLayout()
        tab_main = QTabWidget()
        label_1 = QLabel('<strong>在下方选择检测类型：</strong>')
        tab_pic = QWidget()
        tab_video = QWidget()
        tab_camera = QWidget()
        tab_main.addTab(tab_pic, '图片')
        tab_main.addTab(tab_video, '视频')
        tab_main.addTab(tab_camera, '摄像头')

        tab_pic.setLayout(self.box_pic_main)
        tab_video.setLayout(self.box_video_main)

        box_camera = QVBoxLayout()
        btn_launch_camera = QPushButton('Launch')
        btn_launch_camera.clicked.connect(lambda: mask_rcnn_detect_video_camera('0'))

        box_camera.addWidget(btn_launch_camera, alignment=Qt.AlignCenter)
        tab_camera.setLayout(box_camera)

        box_main.addWidget(label_1)
        box_main.addWidget(tab_main)

        self.setLayout(box_main)
        self.setWindowTitle('Mask R-CNN Demo')
        self.setGeometry(300, 300, 800, 500)
        self.show()

    def init_pic_ui(self):
        self.box_pic_main = QVBoxLayout()

        box_pic_source = QVBoxLayout()
        btngrp_pic_source = QButtonGroup(self)
        radio_random_test = QRadioButton('在测试图片中随机选取一张')
        radio_random_test.toggled.connect(lambda: self.source_radio_state_changed(radio_random_test))
        radio_local_pic = QRadioButton('选择本地图片文件:')
        radio_local_pic.toggled.connect(lambda: self.source_radio_state_changed(radio_local_pic))
        btngrp_pic_source.addButton(radio_random_test)
        btngrp_pic_source.addButton(radio_local_pic)
        box_pic_source.addWidget(radio_random_test)
        box_pic_source.addWidget(radio_local_pic)

        box_pic_choose = QHBoxLayout()

        self.list_pic_path = QListView()
        self.list_pic_path.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.string_model = QStringListModel()

        self.btn_explore_files = QPushButton('浏览')
        self.btn_explore_files.setDisabled(True)
        self.btn_explore_files.clicked.connect(self.show_file_dialog)
        box_pic_choose.addWidget(self.list_pic_path)
        box_pic_choose.addWidget(self.btn_explore_files)

        self.btn_launch = QPushButton('Launch')
        self.btn_launch.setDisabled(True)
        self.btn_launch.clicked.connect(lambda: mask_rcnn_detect_pic(self.picSourceType, self.picFilePaths))

        self.box_pic_main.addLayout(box_pic_source)
        self.box_pic_main.addLayout(box_pic_choose)
        self.box_pic_main.addWidget(self.btn_launch, alignment=Qt.AlignCenter)

    def source_radio_state_changed(self, radio):
        self.picFilePaths = []
        if radio.isChecked():
            if radio.text() == '在测试图片中随机选取一张':
                self.string_model.setStringList([])
                self.list_pic_path.setModel(self.string_model)
                self.btn_explore_files.setDisabled(True)
                self.btn_launch.setDisabled(False)
                self.picSourceType = 'test'
            else:
                self.btn_launch.setDisabled(True)
                self.btn_explore_files.setDisabled(False)
                self.picSourceType = 'custom'

    def show_file_dialog(self):
        file_paths = QFileDialog.getOpenFileNames(caption='选择图片文件', filter='Image Files(*.jpg)')[0]
        self.string_model.setStringList(file_paths)
        self.list_pic_path.setModel(self.string_model)
        if len(file_paths) != 0:
            self.btn_launch.setDisabled(False)
            self.picFilePaths = file_paths

    def init_video_ui(self):
        self.box_video_main = QVBoxLayout()

        box_video_source = QHBoxLayout()

        self.line_edit_file_path = QLineEdit()
        self.line_edit_file_path.setReadOnly(True)
        btn_explore_video = QPushButton('浏览')
        btn_explore_video.clicked.connect(self.show_video_file_dialog)
        box_video_source.addWidget(self.line_edit_file_path)
        box_video_source.addWidget(btn_explore_video)

        self.btn_launch_video = QPushButton('Launch')
        self.btn_launch_video.setDisabled(True)
        self.btn_launch_video.clicked.connect(lambda: mask_rcnn_detect_video_camera(self.videoPath))

        self.box_video_main.addLayout(box_video_source)
        self.box_video_main.addWidget(self.btn_launch_video, alignment=Qt.AlignCenter)

    def show_video_file_dialog(self):
        file_path = QFileDialog.getOpenFileName(caption='选择视频文件', filter='视频文件(*.mp4)')[0]
        self.line_edit_file_path.setText(file_path)
        if file_path != '':
            self.btn_launch_video.setDisabled(False)
            self.videoPath = file_path


def mask_rcnn_detect_pic(pic_source_type, pic_file_paths):
    if pic_source_type == 'test':
        # 测试图像文件目录
        IMAGE_DIR = os.path.join(ROOT_DIR, "images")

        # 在测试图像文件夹中随机加载一张图像
        file_names = next(os.walk(IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

        # 进行detect
        results = model.detect([image], verbose=1)

        # 结果可视化
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
    else:
        images = []
        for im_path in pic_file_paths:
            images.append(skimage.io.imread(im_path))

        for im in images:
            results = model.detect([im], verbose=1)
            r = results[0]
            visualize.display_instances(im, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'])


def mask_rcnn_detect_video_camera(source):
    colors = visualize.random_colors(len(class_names))

    cap = cv2.VideoCapture(0) if source == '0' else cv2.VideoCapture(source)
    while True:
        _, frame = cap.read()
        predictions = model.detect([frame],
                                   verbose=1)  # We are replicating the same image to fill up the batch_size
        p = predictions[0]

        output = visualize.display_instances(frame, p['rois'], p['masks'], p['class_ids'],
                                             class_names, p['scores'], colors=colors, real_time=True)
        cv2.imshow("Mask R-CNN", output)

        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    prepare_model()
    w = Window()
    sys.exit(app.exec_())

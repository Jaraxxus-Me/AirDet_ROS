# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os
from tqdm import tqdm

import cv2
from detectron2.utils.visualizer import ColorMode,Visualizer
from detectron2.structures.instances import Instances
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from fewx.data.build import build_detection_train_loader, build_detection_test_loader
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from fewx.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from skimage import io

# constants
WINDOW_NAME = "COCO detections"

@torch.no_grad()
def Visualize(im, output, train_metadata):
    meta = train_metadata
    # meta.thing_colors = []
    for i in range(len(train_metadata.thing_classes)):
        # meta.thing_colors.append([0,255,0])
        meta.thing_colors[i] = [0,255,0]
    v = Visualizer(im[:, :, ::-1],
                metadata=meta,
                scale=1,
                instance_mode=ColorMode.SEGMENTATION
                 )
    all_instances = output["instances"].to("cpu")
        # Top K mode
    draw_instances =  Instances(all_instances.image_size)
    draw_instances.pred_boxes = all_instances.pred_boxes[0:]
    draw_instances.scores = all_instances.scores[0:]
    draw_instances.pred_classes = all_instances.pred_classes[0:]
    out = v.draw_instance_predictions(draw_instances)
    return out.get_image()[:, :, ::-1]

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def norm_image(image):
    """
    | ~G~G~F~L~V~[~C~O
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    ~T~_~H~PCAM~[
    :param image: [H,W,C],~N~_~K~[~C~O
    :param mask: [H,W],~L~C~[0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # ~P~H并heatmap~H~N~_~K~[~C~O
    cam = heatmap + np.float32(image)/255
    return norm_image(cam), heatmap

def gen_gb(grad):
    """
    ~T~_guided back propagation ~S~E~[~C~O~Z~D梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # | ~G~G~F~L~V
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb

def img_with_box(img, box):
    return

def save_image(image_dicts, input_image_name, network='AirShot-101', output_dir='./results/final-circuit/canary1/5_left'):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/fsod/test_R_101_subt3_final.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def main(args):
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    print(cfg)
    build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    # metadata = MetadataCatalog.get('voc_2012_val')
    #coco
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    # ~J| 载~[~C~O
    path = os.path.expanduser(args.input)
    for img_name in tqdm(os.listdir(path)):
        img_path = os.path.join(path, img_name)
        original_image = read_image(img_path, format="BGR")
        height, width = original_image.shape[:2]
        transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        image = transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(False)

        inputs = {"image": image, "height": height, "width": width}

        model.eval()
        model.zero_grad()
        out = model([inputs])[0]  # cam mask
        #
        image_dict = {}
        img = original_image[..., ::-1]
        image_dict['im_pred_box'] = Visualize(img,out,metadata)
        save_image(image_dict, os.path.basename(img_path),output_dir=args.output)


if __name__ == "__main__":
    """
    Usage:export KMP_DUPLICATE_LIB_OK=TRUE
    python detection/demo.py --config-file detection/faster_rcnn_R_50_C4.yaml \
      --input ./examples/pic1.jpg \
      --opts MODEL.WEIGHTS ./model_final_b1acc2.pkl MODEL.DEVICE cpu
    """
    mp.set_start_method("spawn", force=True)
    arguments = get_parser().parse_args()
    main(arguments)

#  python -m debugpy --listen 0.0.0.0:5680 --wait-for-client grad_head.py --config-file configs/fsod/finetune_R_101_C4_1x_coco2.yaml --input ./examples/head/000000082807.jpg
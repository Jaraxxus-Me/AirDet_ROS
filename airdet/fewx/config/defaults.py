from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER.HEAD_LR_FACTOR = 1.0
_C.SOLVER.BACKBONE_LR_FACTOR = 1.0
_C.SOLVER.START_SAVE_ITER = 115000
_C.SOLVER.CHECKPOINT_PERIOD_EVA = 500
_C.SEED = 1
_C.TEST.VIS = False
_C.TEST.VIS_DIR = 'vis/f12'
_C.TEST.VIS_THRESH = 0.9

_C.PATH = "/CVPR2022/AirDet_ROS/src/airdet_ros/airdet"

# ---------------------------------------------------------------------------- #
# Few shot setting
# ---------------------------------------------------------------------------- #
_C.INPUT.FS = CN()
_C.INPUT.FS.FEW_SHOT = False
_C.INPUT.FS.SUPPORT_WAY = 2
_C.INPUT.FS.SUPPORT_SHOT = 10
_C.DATASETS.TESTSHOTS = 1
_C.DATASETS.CLS = ("vent",)
_C.DATASETS.FINAL = True

# ---------------------------------------------------------------------------- #
# HiFT setting
# ---------------------------------------------------------------------------- #
_C.MODEL.HIFT = CN()
_C.MODEL.HIFT.CHANNEL = 256
_C.MODEL.HIFT.HEADS = 4
_C.MODEL.HIFT.LAYERS_ENC = 1
_C.MODEL.HIFT.LAYERS_DEC = 2
_C.MODEL.HIFT.DIM_FFN = 512

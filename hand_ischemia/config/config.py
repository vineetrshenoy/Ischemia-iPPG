from yacs.config import CfgNode as CN


__all__ = ['get_cfg_defaults', 'convert_to_dict']

_C = CN()


_C.OUTPUT = CN()
_C.OUTPUT.OUTPUT_DIR = "output/"

_C.INPUT = CN()
_C.INPUT.GT_FILEPATH = "/cis/net/r22a/data/vshenoy/merl/MMSE_HR_GT/MMSE_HR_processed_all_subjects_missing_face_untested/40_Subjects_GT_all_cases/"
_C.INPUT.TIME_SERIES_FILEPATH = "/cis/net/r22a/data/vshenoy/merl/time_series_results/biosignals_time_series/unrolled-ippg-time_series/time_series"
_C.INPUT.TRAIN_JSON_PATH = "/cis/home/vshenoy/durr_hand/timeScale_Ischemia/hand_ischemia/data/ischemia_train.json"
_C.INPUT.TEST_JSON_PATH = "/cis/home/vshenoy/durr_hand/timeScale_Ischemia/hand_ischemia/data/ischemia_train.json"
_C.INPUT.CHANNEL = 'PPG_filt_redovergreen'
_C.INPUT.TRAIN_ISCHEMIC = 0
_C.INPUT.TRAIN_PERFUSE = 0
_C.INPUT.TEST_ISCHEMIC = 0
_C.INPUT.TEST_PERFUSE = 0

_C.TIME_SCALE_PPG = CN()
_C.TIME_SCALE_PPG.FPS = 25
_C.TIME_SCALE_PPG.PASSBAND_FREQ = 25
_C.TIME_SCALE_PPG.CUTOFF_FREQ = 160
_C.TIME_SCALE_PPG.NUM_TAPS = 25
_C.TIME_SCALE_PPG.FRAME_STRIDE = 0.4 #units: seconds
_C.TIME_SCALE_PPG.FRAME_STRIDE_TEST = 10.0 #units: seconds
_C.TIME_SCALE_PPG.MIN_WINDOW_SEC = 10.0
_C.TIME_SCALE_PPG.TIME_WINDOW_SEC = 10.0
_C.TIME_SCALE_PPG.CLS_MODEL_TYPE = "SPEC" # TiSc or SPEC
_C.TIME_SCALE_PPG.USE_DENOISER = True

_C.DENOISER = CN()
_C.DENOISER.BATCH_SIZE = 8
_C.DENOISER.EPOCHS = 8
_C.DENOISER.LR = 0.0003
_C.DENOISER.WEIGHT_DECAY = 0.0
_C.DENOISER.SCHEDULER_MILESTONE = 2

_C.TEST = CN()
_C.TEST.EVAL_PERIOD = 10
_C.TEST.PLOT_INPUT_OUTPUT = True
_C.TEST.PLOT_UNROLL = False
_C.TEST.PLOT_LAST = True
_VALID_TYPES = {tuple, list, str, int, float, bool}


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a configuration node into a dictionary format

    Args:
        cfg_node (CfgNode): CfgNode as described in YACS
        key_list (list, optional): _description_. Defaults to [].

    Returns:
        dict: Dictionary representation of CfgNode
    """
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES),)
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


def get_cfg_defaults():
    """Return the base configuration file

    Returns:
        CfgNode: A base configuration node
    """
    return _C.clone()

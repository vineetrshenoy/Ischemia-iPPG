import sys
import os
import argparse


__all__ = ["default_argument_parser"]


def default_argument_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--config-file", default="",
                        metavar="FILE", help="path to config file", required=False)
    parser.add_argument("--test_only", action='store_true')
    parser.add_argument("--test_CV", action='store_true')
    parser.add_argument("--checkpoint", default="",
                        metavar="FILE", help="path to location of checkpoint", required=False)
    parser.add_argument("--experiment_id", type=str)
    parser.add_argument("--cls_experiment_id", type=str)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser

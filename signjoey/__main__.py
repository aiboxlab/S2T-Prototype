import argparse
import os

import sys
from signjoey.training import train
from signjoey.prediction import test
from signjoey.prediction import prediction
from signjoey.prediction import prepare_model

from signjoey.data import load_dictionary

from signjoey.helpers import load_config

sys.path.append("/vol/research/extol/personal/cihan/code/SignJoey")


def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test", "prediction"], help="train a model or test")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction")

    ap.add_argument(
        "--output_path", type=str, help="path for saving translation output"
    )
    ap.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        train(cfg_file=args.config_path)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)
    elif args.mode == "prediction":

        cfg_file="configs/dan-nl_usuario16.yaml"
        cfg = load_config(cfg_file)
        gls_vocab, txt_vocab, sequence_field, signer_field, sgn_field, gls_field, txt_field = load_dictionary(data_cfg=cfg["data"])
        model = prepare_model(cfg, gls_vocab, txt_vocab, ckpt=args.ckpt)
        prediction(model, gls_vocab, txt_vocab, sequence_field, signer_field, sgn_field, gls_field, txt_field, cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)

    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()


import os
import torch
from os.path import exists
from FeatureExtractor.s2t_feature_extractor import extract_features
from FeatureExtractor.models.pytorch_i3d import InceptionI3d
from signjoey.prediction import prediction, prepare_model
from signjoey.data import load_dictionary
from signjoey.helpers import load_config

class TranslateService_S2T:

	def __init__(self, model_project_path):

		cfg_file = "configs/dan-nl_usuario16.yaml"
		cfg = load_config(cfg_file)
		gls_vocab, txt_vocab, sequence_field, signer_field, sgn_field, gls_field, txt_field = load_dictionary(data_cfg=cfg["data"])

		i3d = InceptionI3d(400, in_channels=3)
		i3d.replace_logits(2000)
		i3d.load_state_dict(torch.load(model_project_path)) # Network's Weight
		i3d.cuda()
		i3d.train(False)  # Set model to evaluate mode

		self.i3d = i3d
		self.cfg_file = cfg_file
		self.cfg = cfg
		self.gls_vocab = gls_vocab
		self.txt_vocab = txt_vocab
		self.sequence_field = sequence_field
		self.signer_field = signer_field
		self.sgn_field = sgn_field
		self.gls_field = gls_field
		self.txt_field = txt_field
		self.model = prepare_model(cfg, gls_vocab, txt_vocab, ckpt=None)

	def translate_from_keypoints(self, keypoints_data):
		raise NotImplementedError("Not implemented for this Model.")


	def translate_from_video(self, video_path):

		span = 16
		if not exists(video_path):
			print(video_path,": Video file was not found!")
			return ""

		try:
			print("extracting.")
			extract_features(video_path, self.i3d)
		except:
			print(video_path, ": An error occurred during extraction!")
			return ""

		output = ""
		try:
			print("predicting.")
			output = prediction(
				self.model,
				self.gls_vocab,
				self.txt_vocab,
				self.sequence_field,
				self.signer_field,
				self.sgn_field,
				self.gls_field,
				self.txt_field,
				self.cfg_file,
				None,
				output_path="output"
			)
		except:
			print(video_path,": An error occurred during prediction!")

		os.remove("./data/test16.gzip")

		return output[0]

if __name__ == "__main__":
	ts = TranslateService_S2T('./FeatureExtractor/checkpoints/archive/nslt_2000_065538_0.514762.pt')
	print(ts.translate_from_video("..."))
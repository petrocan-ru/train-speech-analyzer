# проверка указанной модели на указанном манифесте 
#!/usr/bin/env python3

import sys
import os
import argparse, sys
import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)
import nemo.collections.asr as nemo_asr
import json
from pathlib import Path
import librosa
from tqdm import tqdm
from functools import partialmethod
from pydub import AudioSegment
from evaluate import load

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model to check (name or file path)")
parser.add_argument("--manifest", help="Manifest file to check")
parser.add_argument("--compare", help="Set 1 to turn on comparing with original nVidia model")
args = parser.parse_args()

asr_model_original = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_ru_conformer_transducer_large")

if (args.model == "original"):
	asr_model = asr_model_original
else:
	asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.model)
predictions = []
references = []

with Path(args.manifest).open(mode="r") as inp:
	for json_str in inp:
		manifest_string_dict = json.loads(json_str)
		print(80*"=" + " " + os.path.basename(manifest_string_dict['audio_filepath']))
		
		if(os.path.isfile(manifest_string_dict['audio_filepath'])):
			path = manifest_string_dict['audio_filepath']
		elif(os.path.isfile(os.path.dirname(args.manifest) + '/' + manifest_string_dict['audio_filepath'].replace('\\', '/'))):
			path = os.path.dirname(args.manifest) + '/' + manifest_string_dict['audio_filepath'].replace('\\', '/')
		else:
			print('File not found!')
			print(os.path.dirname(args.manifest) + manifest_string_dict['audio_filepath'])
			continue
		sound = AudioSegment.from_file(path)
		t1 = manifest_string_dict['offset'] * 1000
		t2 = manifest_string_dict['duration'] * 1000 + t1
		sound = sound[t1:t2]
		sound.export('tmpaudio.wav', format="wav")
		result = asr_model.transcribe(['tmpaudio.wav'])
		
		print("Распознано человеком: " + manifest_string_dict['text'])
		references.append(manifest_string_dict['text'])
		print("Дообученная   модель: " + result[0][0])
		predictions.append(result[0][0])
		if args.compare == "1":
			result_orig = asr_model_original.transcribe(['tmpaudio.wav'])
			print("Оригинальная  модель: " + result_orig[0][0])
		
wer = load("wer")
wer_score = wer.compute(predictions=predictions, references=references)
print("WER score (lower is better): ", wer_score)
		

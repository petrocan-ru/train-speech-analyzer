import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import json
#import pydub

aug_train_ds = open('aug_train.manifest', 'w+')

#WAV_FILE_NAME = '29к_874 КВ - 02.05.2024 01_08_44.wav'
#WAV_FILE_NAME = '67к_823 КВ - 02.05.2024 05_01_42.wav'
#WAV_FILE_NAME = '30к_872 КВ - 02.05.2024 08_40_27.wav'
record_path = os.getcwd()# + '\\conv\\'
aug_path = os.path.join(os.getcwd(),'augmented/')
#lowcut = 400.0
#highcut = 4500.0
FRAME_RATE = 8000

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.7 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass_filter(buffer):
    return butter_bandpass_filter(buffer, lowcut, highcut, FRAME_RATE, order=2)

def aug(amplify):
	train_ds = open('train.manifest', 'r')
	for idx, row in enumerate(train_ds):
		#WAV_FILE_NAME = 'tmpaudio.wav'
		manifest_string_dict = json.loads(row)
		#print(80*"=" + " " + os.path.basename(manifest_string_dict['audio_filepath']))
		
		if(os.path.isfile(manifest_string_dict['audio_filepath'])):
			path = manifest_string_dict['audio_filepath']
		elif(os.path.isfile(os.path.dirname(args.manifest) + '/' + manifest_string_dict['audio_filepath'].replace('\\', '/'))):
			path = os.path.dirname(args.manifest) + '/' + manifest_string_dict['audio_filepath'].replace('\\', '/')
		else:
			print('File not found!')
	#		print(os.path.dirname(args.manifest) + manifest_string_dict['audio_filepath'])
			continue
		sound = AudioSegment.from_file(path)
		t1 = manifest_string_dict['offset'] * 1000
		t2 = manifest_string_dict['duration'] * 1000 + t1
		sound = sound[t1:t2]
		sound.export('tmpaudio.wav', format="wav")
		
		samplerate, data = wavfile.read('tmpaudio.wav')
		assert samplerate == FRAME_RATE
	#	filtered = np.apply_along_axis(bandpass_filter, 0, data).astype('int16')*100
		filtered = np.apply_along_axis(bandpass_filter, 0, data).astype('int16')*amplify
#		filtered.export('filtered_tmp.wav', format="wav")
		wavfile.write('filtered_tmp.wav', samplerate, filtered)
		
		filtered_input = AudioSegment.from_file('filtered_tmp.wav', "wav")
		nc = effects.normalize(filtered_input)
		out_wav_file = os.path.join(aug_path, os.path.basename(path)[:-4] + f'_aug_{idx}_{amplify}.wav')
		nc.export(out_wav_file, format="wav")
		
		manifest_string_dict['audio_filepath'] = out_wav_file
		manifest_string_dict['offset'] = 0
		out_row = (json.dumps(manifest_string_dict, ensure_ascii=False) + '\n')
		aug_train_ds.write(out_row)
		
		#wavfile.write(os.path.join(record_path, f'filtered_tmp.wav'), samplerate, filtered)

		#rawsound = AudioSegment.from_file(os.path.join(record_path, f'filtered_{WAV_FILE_NAME}'), "wav")
		#normalizedsound = effects.normalize(rawsound)  
		#normalizedsound.export(os.path.join(record_path, f'normalized_{WAV_FILE_NAME}'), format="wav")
		'''
		sound = AudioSegment.from_file(os.path.join(record_path, f'filtered_tmp.wav'), "wav")
		chunks = split_on_silence(
			sound,
			min_silence_len=1500, # ms
			silence_thresh=-50, # dB
			keep_silence=2000 # ms
		)
		'''
lowcut = 450.0
highcut = 4600.0
aug(100)
lowcut = 900.0
highcut = 4100.0
aug(210)

train_ds.close()
aug_train_ds.close()

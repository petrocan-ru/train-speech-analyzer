# мини веб-приложение для демонстрации инференса нейросети nVidia NeMo ASR
#!/usr/bin/env python3

import sys
import os
import argparse, sys
import nemo.collections.asr as nemo_asr
import json
import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)
from pathlib import Path
from flask import Flask, request, redirect, url_for, render_template, send_file, send_from_directory

from omegaconf import OmegaConf, open_dict

ALLOWED_EXTENSIONS = {'wav', 'wave', 'mp3'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------- страницы --------------------
@app.route('/static/<path:path>')
def static_res(path):
    return send_from_directory('static', path)

@app.route('/', methods=['GET'])
def index_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        #flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        #flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = 'audio_test_upload.wav'
        file.save(filename)
        return transcribe_to_phrases(filename)

def transcribe_to_phrases(filename):
    # specify flag `return_hypotheses=True``
    hypotheses = asr_model.transcribe([filename], return_hypotheses=True)

    # if hypotheses form a tuple (from RNNT), extract just "best" hypotheses
    if type(hypotheses) == tuple and len(hypotheses) == 2:
        hypotheses = hypotheses[0]

    timestamp_dict = hypotheses[0].timestep # extract timesteps from hypothesis of first (and only) audio file

    last_abs_delta = 0
    prev_offset = 0
    buffer = ""
    phrases = []

    for c in timestamp_dict["char"]:
        last_abs_delta = abs(prev_offset - c["end_offset"])
        prev_offset = c["end_offset"]

        phrase_buffer = ""
        for char in c["char"]:
            if(char == ''):
                phrase_buffer += ' '
            else:
                phrase_buffer += char

        if(last_abs_delta < 50):
            buffer = buffer + phrase_buffer
        else:
            if(len(buffer) > 1):
                phrases.append({"role": "null", "text": buffer})
            buffer = phrase_buffer
    
    return json.dumps(phrases, ensure_ascii=False)
        

asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(os.path.dirname(__file__) + '/ASR-Model-Language-ru.nemo')
# update decoding config to preserve alignments and compute timestamps
decoding_cfg = asr_model.cfg.decoding
with open_dict(decoding_cfg):
    decoding_cfg.preserve_alignments = True
    decoding_cfg.compute_timestamps = True
    asr_model.change_decoding_strategy(decoding_cfg)
print("Model Loaded")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4567)
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from tqdm.auto import tqdm
import json
from collections import defaultdict
import torch
import torch.nn as nn
import pytorch_lightning as ptl
from nemo.utils import logging, exp_manager
import copy
import os
import glob
import subprocess
from omegaconf import OmegaConf, open_dict

TOKENIZER_TYPE = "bpe"
VOCAB_SIZE = 34
TOKENIZER_DIR = f"out_manifests/tokenizer_dir/tokenizer_spe_{TOKENIZER_TYPE}_v{VOCAB_SIZE}/"
LANGUAGE = "ru"

manifests = "out_manifests"
train_manifest = f"{manifests}/train.manifest"
dev_manifest = f"{manifests}/dev.manifest"
#test_manifest = f"{manifests}/test.manifest"
test_manifest = f"/home/petrocan/nvidia-asr/etalon_test_dataset/test.manifest"

#model = nemo_asr.models.ASRModel.from_pretrained("stt_en_citrinet_512", map_location='cpu')
#model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_ru_conformer_transducer_large", map_location='cpu')
model = nemo_asr.models.EncDecRNNTBPEModel.restore_from("/home/petrocan/nvidia-asr/experiments/lang-ru/ASR-Model-Language-ru/2023-11-10_17-33-41/checkpoints/ASR-Model-Language-ru.nemo")

# Preserve the decoder parameters in case weight matching can be done later
pretrained_decoder = model.decoder.state_dict()

model.change_vocabulary(new_tokenizer_dir=TOKENIZER_DIR, new_tokenizer_type=TOKENIZER_TYPE)

# Insert preserved model weights if shapes match
#if model.decoder.decoder_layers[0].weight.shape == pretrained_decoder['decoder_layers.0.weight'].shape:
#    model.decoder.load_state_dict(pretrained_decoder)
#    logging.info("Decoder shapes matched - restored weights from pre-trained model")
#else:
#    logging.info("\nDecoder shapes did not match - could not restore decoder weights from pre-trained model.")

def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

freeze_encoder = True #@param ["False", "True"] {type:"raw"}
freeze_encoder = bool(freeze_encoder)
if freeze_encoder:
    model.encoder.freeze()
    model.encoder.apply(enable_bn_se)
    logging.info("Model encoder has been frozen")
else:
    model.encoder.unfreeze()
    logging.info("Model encoder has been un-frozen")
    
cfg = copy.deepcopy(model.cfg)

# Setup new tokenizer
cfg.tokenizer.dir = TOKENIZER_DIR
cfg.tokenizer.type = TOKENIZER_TYPE

# Set tokenizer config
model.cfg.tokenizer = cfg.tokenizer

# Setup train/val/test configs
print(OmegaConf.to_yaml(cfg.train_ds))

# Setup train, validation, test configs
with open_dict(cfg):
    # Train dataset
    cfg.train_ds.manifest_filepath = f"{train_manifest},{dev_manifest}"
    cfg.train_ds.batch_size = 2
    cfg.train_ds.num_workers = 12
    cfg.train_ds.pin_memory = True
    cfg.train_ds.use_start_end_token = True
    cfg.train_ds.trim_silence = True
    cfg.train_ds.normalize_transcripts = False

    # Validation dataset
    cfg.validation_ds.manifest_filepath = dev_manifest
    cfg.validation_ds.batch_size = 2
    cfg.validation_ds.num_workers = 12
    cfg.validation_ds.pin_memory = True
    cfg.validation_ds.use_start_end_token = True
    cfg.validation_ds.trim_silence = True
    cfg.validation_ds.normalize_transcripts = False

    # Test dataset
    cfg.test_ds.manifest_filepath = test_manifest
    cfg.test_ds.batch_size = 2
    cfg.test_ds.num_workers = 12
    cfg.test_ds.pin_memory = True
    cfg.test_ds.use_start_end_token = True
    cfg.test_ds.trim_silence = True
    
# setup model with new configs
model.setup_training_data(cfg.train_ds)
model.setup_multiple_validation_data(cfg.validation_ds)
model.setup_multiple_test_data(cfg.test_ds)

def analyse_ctc_failures_in_model(model):
    count_ctc_failures = 0
    am_seq_lengths = []
    target_seq_lengths = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    mode = model.training
    
    train_dl = model.train_dataloader()

    with torch.no_grad():
        model = model.eval()
        for batch in tqdm(train_dl, desc='Checking for CTC failures'):
            x, x_len, y, y_len = batch
            x, x_len = x.to(device), x_len.to(device)
            x_logprobs, x_len = model(input_signal=x, input_signal_length=x_len)

            # Find how many CTC loss computation failures will occur
            for xl, yl in zip(x_len, y_len):
                if xl <= yl:
                    count_ctc_failures += 1

            #  Record acoustic model lengths=
            am_seq_lengths.extend(x_len.to('cpu').numpy().tolist())

            # Record target sequence lengths
            target_seq_lengths.extend(y_len.to('cpu').numpy().tolist())
          
            del x, x_len, y, y_len, x_logprobs #, greedy_predictions
    
    if mode:
        model = model.train()
      
    return count_ctc_failures, am_seq_lengths, target_seq_lengths
    
results = analyse_ctc_failures_in_model(model)
num_ctc_failures, am_seq_lengths, target_seq_lengths = results

if num_ctc_failures > 0:
    logging.warning(f"\nCTC loss will fail for {num_ctc_failures} samples ({num_ctc_failures * 100./ float(len(am_seq_lengths))} % of samples)!\n"
                  f"Increase the vocabulary size of the tokenizer so that this number becomes close to zero !")
else:
    logging.info("No CTC failure cases !")

# Compute average ratio of T / U
avg_T = sum(am_seq_lengths) / float(len(am_seq_lengths))
avg_U = sum(target_seq_lengths) / float(len(target_seq_lengths))

avg_length_ratio = 0
for am_len, tgt_len in zip(am_seq_lengths, target_seq_lengths):
    avg_length_ratio += (am_len / float(tgt_len))
avg_length_ratio = avg_length_ratio / len(am_seq_lengths)

print(f"Average Acoustic model sequence length = {avg_T}")
print(f"Average Target sequence length = {avg_U}")
print()
print(f"Ratio of Average AM sequence length to target sequence length = {avg_length_ratio}")

print(OmegaConf.to_yaml(cfg.optim))

with open_dict(model.cfg.optim):
    model.cfg.optim.lr = 0.025
    model.cfg.optim.weight_decay = 0.001
    model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
    model.cfg.optim.sched.warmup_ratio = 0.10  # 10 % warmup
    model.cfg.optim.sched.min_lr = 1e-9

with open_dict(model.cfg.spec_augment):
    model.cfg.spec_augment.freq_masks = 2
    model.cfg.spec_augment.freq_width = 25
    model.cfg.spec_augment.time_masks = 10
    model.cfg.spec_augment.time_width = 0.05

model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)

#@title Metric
use_cer = True #@param ["False", "True"] {type:"raw"}
log_prediction = True #@param ["False", "True"] {type:"raw"}

#model._wer.use_cer = use_cer
#model._wer.log_prediction = log_prediction

if torch.cuda.is_available():
    accelerator = 'gpu'
else:
    accelerator = 'gpu'

EPOCHS = 600  # 100 epochs would provide better results
torch.set_float32_matmul_precision("medium")
trainer = ptl.Trainer(devices=1, 
                      accelerator=accelerator, 
                      max_epochs=EPOCHS, 
                      accumulate_grad_batches=2,
                      enable_checkpointing=False,
                      logger=False,
                      log_every_n_steps=30,
                      check_val_every_n_epoch=10)

# Setup model with the trainer
model.set_trainer(trainer)

# finally, update the model's internal config
model.cfg = model._cfg

# Environment variable generally used for multi-node multi-gpu training.
# In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
os.environ.pop('NEMO_EXPM_VERSION', None)

config = exp_manager.ExpManagerConfig(
    exp_dir=f'experiments/lang-{LANGUAGE}/',
    name=f"ASR-Model-Language-{LANGUAGE}",
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
    ),
)

config = OmegaConf.structured(config)

logdir = exp_manager.exp_manager(trainer, config)

#%%time
trainer.fit(model)

save_path = f"Model-{LANGUAGE}.nemo"
model.save_to(f"{save_path}")
print(f"Model saved at path : {os.getcwd() + os.path.sep + save_path}")

import os
import os.path as osp
import shutil
import time
import random
import logging
import warnings
warnings.simplefilter('ignore')

import yaml
import click
import numpy as np
import torch
import torch.nn.functional as F
from munch import Munch

from models import *
from meldataset import build_dataloader, get_bin_from_mel_input_length
from utils import *
from utils import enable_gradient_checkpointing, configure_cuda_allocator
from losses import *
from optimizers import build_optimizer
from train_common import (Amp, ProgressLR, setup_speed_flags, is_oom, free_memory,
                          select_segments, s2s_loss, all_finite, resolve_precision,
                          phase_scale)

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from torch.utils.tensorboard import SummaryWriter

logger = get_logger(__name__, log_level="DEBUG")

# Modules the first stage never calls: they are neither wrapped in DDP nor
# given optimizer state.
_UNUSED_IN_FIRST_STAGE = ('bert', 'bert_encoder', 'predictor', 'predictor_encoder',
                          'diffusion', 'wd')


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    if config.get('cuda_expandable_segments', True):
        configure_cuda_allocator(expandable_segments=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Mixed precision is handled by train_common.Amp (one GradScaler that is
    # correct for the several optimizers stepped per iteration), not by
    # Accelerate, which is used for device placement / DDP only.
    accelerator = Accelerator(project_dir=log_dir, split_batches=True,
                              mixed_precision='no', kwargs_handlers=[ddp_kwargs])
    is_main = accelerator.is_main_process
    world = accelerator.num_processes
    if is_main:
        shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
        writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.logger.addHandler(file_handler)
    logging.getLogger('meldataset').addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = accelerator.device
    setup_speed_flags(config)

    epochs = config.get('epochs_1st', 200)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)
    dynamic_batch = config.get('dynamic_batch', True)
    # Frame budget per batch (bin of padded length L -> batch size budget // L).
    max_batch_frames = int(config.get('max_batch_frames') or (batch_size * max_len))
    # With dynamic batching whole utterances are used; optionally cap them.
    seg_cap = config.get('max_segment_len', None) if dynamic_batch else max_len
    batch_size_file = osp.join(log_dir, 'batch_sizes_1st.json')

    num_workers = int(config.get('num_workers', 2))
    disc_update_freq = max(1, int(config.get('disc_update_freq', 1)))
    enable_gradient_checkpointing(config.get('gradient_checkpointing', False))
    amp = Amp(resolve_precision(config), device=str(device), clip_norm=config.get('grad_clip_norm', None))

    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker

    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        dataset_config={'multispeaker': multispeaker},
                                        device=device,
                                        dynamic_batch=dynamic_batch,
                                        batch_size_file=batch_size_file,
                                        max_batch_frames=max_batch_frames,
                                        max_batch_size=int(config.get('max_batch_size', 64)),
                                        num_replicas=world if dynamic_batch else 1,
                                        rank=accelerator.process_index if dynamic_batch else 0)
    batch_manager = train_dataloader.batch_manager

    # Validation: batch size 1, whole utterances -> deterministic, no cropping.
    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=1,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={'multispeaker': False})

    with accelerator.main_process_first():
        # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        from Utils.PLBERT.util import load_plbert
        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)

    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    # The pitch extractor is only ever run under no_grad here (the original
    # "stepped" it after TMA_epoch, a no-op without gradients): freeze it.
    for p in model.pitch_extractor.parameters():
        p.requires_grad = False
    for key in _UNUSED_IN_FIRST_STAGE:
        for p in model[key].parameters():
            p.requires_grad = False

    best_loss = float('inf')

    loss_params = Munch(config['loss_params'])
    TMA_epoch = loss_params.TMA_epoch

    for k in model:
        if any(p.requires_grad for p in model[k].parameters()):
            model[k] = accelerator.prepare(model[k])
        else:
            model[k] = model[k].to(device)

    if not dynamic_batch:
        # Legacy random batches: let Accelerate shard them (global batch_size).
        train_dataloader = accelerator.prepare(train_dataloader)

    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                scheduler_params_dict={key: {} for key in model},
                                lr=float(config['optimizer_params'].get('lr', 1e-4)),
                                use_8bit=config.get('use_8bit_optimizer', False))
    lr_sched = ProgressLR(optimizer, epochs, kind=config.get('lr_schedule', 'constant'))

    start_epoch, iters = 0, 0
    with accelerator.main_process_first():
        if config.get('pretrained_model', '') != '':
            load_only = config.get('load_only_params', True)
            model, optimizer, start_epoch, iters = load_checkpoint(
                model, optimizer, config['pretrained_model'], load_only_params=load_only)
            if not load_only:
                start_epoch += 1  # the saved epoch was completed
                amp.load_state_dict(torch.load(config['pretrained_model'], map_location='cpu',
                                               weights_only=False).get('amp_scaler'))

    raw = Munch({k: accelerator.unwrap_model(model[k]) for k in model})
    n_down = raw.text_aligner.n_down

    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model,
                   model.wd,
                   sr,
                   model_params.slm.sr).to(device)

    loss_test = float('nan')
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        n_running = 0
        start_time = time.time()
        tma = epoch >= TMA_epoch

        _ = [model[key].train() for key in model]
        model.pitch_extractor.eval()

        if batch_manager is not None:
            # Memory grows when the aligner/discriminators join (TMA); learned
            # sizes are kept per phase.
            phase = 'first_tma' if tma else 'first'
            batch_manager.set_phase(phase, scale=phase_scale(config, phase))
        if epoch == TMA_epoch:
            log_print('Epoch %d: TMA starts (text aligner + discriminators join training)' % epoch, logger)

        steps_in_epoch = max(1, len(train_dataloader))
        n_skipped = 0
        for i, batch in enumerate(train_dataloader):
            lr_sched.set_progress(epoch + min(i / steps_in_epoch, 1.0))
            mel_input_length = None
            try:
                waves = batch[0]
                batch = [b.to(device, non_blocking=True) for b in batch[1:]]
                texts, input_lengths, _, _, mels, mel_input_length, _ = batch
                bs = mels.shape[0]

                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    text_mask = length_to_mask(input_lengths).to(device)

                # Before TMA the aligner is not trained: run it without a graph.
                with torch.set_grad_enabled(tma), amp.autocast():
                    ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)

                with torch.no_grad():
                    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                    attn_mask = (attn_mask < 1)

                s2s_attn = s2s_attn.masked_fill(attn_mask, 0.0)

                with torch.no_grad():
                    mask_ST = mask_from_lens(s2s_attn.float(), input_lengths, mel_input_length // (2 ** n_down))
                    s2s_attn_mono = maximum_path(s2s_attn.float(), mask_ST)

                with amp.autocast():
                    t_en = model.text_encoder(texts, input_lengths, text_mask)
                    # 50% of chance of using monotonic version
                    if bool(random.getrandbits(1)):
                        asr = (t_en @ s2s_attn.to(t_en.dtype))
                    else:
                        asr = (t_en @ s2s_attn_mono.to(t_en.dtype))

                en, gt, wav, st, _ = select_segments(
                    asr, mels, waves, mel_input_length, seg_cap, device,
                    full_length=dynamic_batch, st_mode='random' if multispeaker else 'same')

                # clip too short to be used by the style encoder
                if gt.shape[-1] < 80:
                    continue

                with torch.no_grad(), amp.autocast():
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))

                with amp.autocast():
                    s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
                    y_rec = model.decoder(en, F0_real, real_norm, s)

                # discriminator loss
                if tma and (iters % disc_update_freq) == 0:
                    optimizer.zero_grad()
                    with amp.autocast():
                        d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                    amp.backward(d_loss)
                    amp.step(optimizer, ['msd', 'mpd'])
                    amp.update()
                else:
                    d_loss = 0

                # generator loss
                optimizer.zero_grad()
                loss_mel = stft_loss(y_rec, wav)

                if tma:  # start TMA training
                    loss_s2s = s2s_loss(s2s_pred, texts, input_lengths)
                    loss_mono = F.l1_loss(s2s_attn.float(), s2s_attn_mono) * 10
                    with amp.autocast():
                        loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
                    loss_slm = wl(wav.detach(), y_rec).mean()

                    g_loss = loss_params.lambda_mel * loss_mel + \
                        loss_params.lambda_mono * loss_mono + \
                        loss_params.lambda_s2s * loss_s2s + \
                        loss_params.lambda_gen * loss_gen_all + \
                        loss_params.lambda_slm * loss_slm
                else:
                    loss_s2s = 0
                    loss_mono = 0
                    loss_gen_all = 0
                    loss_slm = 0
                    g_loss = loss_mel

                if not all_finite(g_loss):
                    # Still run backward so DDP ranks stay in lock-step; Amp
                    # refuses to apply the non-finite gradients.
                    n_skipped += 1
                amp.backward(g_loss)
                keys = ['text_encoder', 'style_encoder', 'decoder']
                if tma:
                    keys.append('text_aligner')
                amp.step(optimizer, keys)
                amp.update()

                if all_finite(loss_mel):
                    running_loss += loss_mel.item()
                    n_running += 1
                iters = iters + 1
                if batch_manager is not None:
                    batch_manager.on_success(get_bin_from_mel_input_length(mel_input_length), bs)

                if (i+1) % log_interval == 0 and is_main:
                    mel_avg = running_loss / max(1, n_running)
                    log_print('Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f'
                              % (epoch+1, epochs, i+1, len(train_dataloader), mel_avg, loss_gen_all, d_loss, loss_mono, loss_s2s, loss_slm), logger)

                    writer.add_scalar('train/mel_loss', mel_avg, iters)
                    writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                    writer.add_scalar('train/d_loss', d_loss, iters)
                    writer.add_scalar('train/mono_loss', loss_mono, iters)
                    writer.add_scalar('train/s2s_loss', loss_s2s, iters)
                    writer.add_scalar('train/slm_loss', loss_slm, iters)

                    running_loss = 0
                    n_running = 0

                    print('Time elasped:', time.time()-start_time)

            except RuntimeError as e:
                if not is_oom(e):
                    raise
                if world > 1:
                    raise RuntimeError(
                        'CUDA out of memory in multi-GPU training. OOM recovery only '
                        'works in a single process (the other ranks would hang); '
                        'lower max_batch_frames, or run one epoch on a single GPU '
                        'first so batch_sizes_1st.json learns sizes that fit.') from e
                bin_id = get_bin_from_mel_input_length(mel_input_length) if mel_input_length is not None else None
                log_print('WARNING: OOM at step %d (bin %s), skipping batch' % (i, bin_id), logger)
                # Release the partial graph before the next batch allocates.
                batch = waves = mels = texts = None
                ppgs = s2s_pred = s2s_attn = s2s_attn_mono = t_en = asr = None
                en = gt = wav = st = s = y_rec = F0_real = real_norm = None
                d_loss = loss_mel = loss_s2s = loss_mono = loss_gen_all = loss_slm = g_loss = None
                optimizer.zero_grad()
                amp._unscaled.clear()
                free_memory()
                if batch_manager is not None and bin_id is not None:
                    batch_manager.on_oom(bin_id, bs)

        if n_skipped and is_main:
            log_print('Epoch %d: %d steps had non-finite losses and were not applied' % (epoch + 1, n_skipped), logger)

        # ------------------------------------------------------------------
        # Validation (main process, bare modules, whole utterances)
        # ------------------------------------------------------------------
        accelerator.wait_for_everyone()
        if is_main:
            loss_test = 0
            iters_test = 0
            _ = [raw[key].eval() for key in raw]
            samples = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, _, _, mels, mel_input_length, _ = batch

                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    text_mask = length_to_mask(input_lengths).to(device)
                    with amp.autocast():
                        ppgs, s2s_pred, s2s_attn = raw.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                    attn_mask = (attn_mask < 1)
                    s2s_attn = s2s_attn.masked_fill(attn_mask, 0.0)

                    with amp.autocast():
                        t_en = raw.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn.to(t_en.dtype))

                    en, gt, wav, _, _ = select_segments(asr, mels, waves, mel_input_length,
                                                        None, device, full_length=True)
                    if gt.shape[-1] < 80:
                        continue

                    with amp.autocast():
                        F0_real, _, F0 = raw.pitch_extractor(gt.unsqueeze(1))
                        s = raw.style_encoder(gt.unsqueeze(1))
                        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                        y_rec = raw.decoder(en, F0_real, real_norm, s)

                    loss_mel = stft_loss(y_rec, wav)
                    if not all_finite(loss_mel):
                        continue
                    loss_test += loss_mel.item()
                    iters_test += 1
                    if len(samples) < 7:
                        samples.append((y_rec.float().cpu().numpy().squeeze(), waves[0],
                                        s2s_attn[0].float().cpu().numpy()))

            loss_test = loss_test / max(1, iters_test)
            print('Epochs:', epoch + 1)
            log_print('Validation loss: %.3f' % loss_test + '\n\n\n\n', logger)
            print('\n\n\n')
            writer.add_scalar('eval/mel_loss', loss_test, epoch + 1)
            if samples:
                writer.add_figure('eval/attn', get_image(samples[0][2]), epoch)
            for bib, (y, y_gt, _) in enumerate(samples):
                writer.add_audio('eval/y' + str(bib), y, epoch, sample_rate=sr)
                if epoch == 0:
                    writer.add_audio('gt/y' + str(bib), y_gt.squeeze(), epoch, sample_rate=sr)

            if epoch % saving_epoch == 0:
                if loss_test < best_loss:
                    best_loss = loss_test
                print('Saving..')
                state = {
                    'net':  {key: raw[key].state_dict() for key in raw},
                    'optimizer': optimizer.state_dict(),
                    'amp_scaler': amp.state_dict(),
                    'iters': iters,
                    'val_loss': loss_test,
                    'epoch': epoch,
                }
                save_path = osp.join(log_dir, 'epoch_1st_%05d.pth' % epoch)
                torch.save(state, save_path)
        accelerator.wait_for_everyone()

    if is_main:
        print('Saving..')
        state = {
            'net':  {key: raw[key].state_dict() for key in raw},
            'optimizer': optimizer.state_dict(),
            'amp_scaler': amp.state_dict(),
            'iters': iters,
            'val_loss': loss_test,
            'epoch': epochs - 1,
        }
        save_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
        torch.save(state, save_path)


if __name__=="__main__":
    main()

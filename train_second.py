"""Second-stage (joint) training, and fine-tuning (train_finetune.py).

Both scripts share this implementation; ``finetune=True`` reproduces the
differences of the original train_finetune.py:

* starts from a full second-stage checkpoint (``pretrained_model``);
* the text aligner (and text encoder) keep training with the TMA losses;
* the decoder, style encoder and discriminators train from the first epoch
  (no ``joint_epoch`` gating for them), always against the recording;
* the style vectors are computed from the target segment itself.
"""
import os
import os.path as osp
import copy
import random
import shutil
import time
import logging
from logging import StreamHandler
import warnings
warnings.simplefilter('ignore')

import yaml
import click
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from munch import Munch
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader, get_bin_from_mel_input_length
from models import *
from losses import *
from utils import *
from utils import enable_gradient_checkpointing, configure_cuda_allocator
from train_common import (Amp, ProgressLR, resolve_device, resolve_precision, setup_speed_flags,
                          is_oom, free_memory, all_finite, select_segments, global_styles,
                          duration_losses, s2s_loss, module_grad_norm, phase_scale,
                          set_requires_grad)

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer


# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def run(config_path, finetune=False):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)
    logging.getLogger('meldataset').addHandler(file_handler)

    # Must run before any tensor is moved to CUDA.
    if config.get('cuda_expandable_segments', True):
        configure_cuda_allocator(expandable_segments=True)
    device = resolve_device(config)
    setup_speed_flags(config)

    batch_size = config.get('batch_size', 10)
    epochs = config.get('epochs' if finetune else 'epochs_2nd', 200)
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

    num_workers = int(config.get('num_workers', 2))
    disc_update_freq = max(1, int(config.get('disc_update_freq', 1)))
    diff_steps = (int(config.get('diffusion_num_steps_min', 3)),
                  int(config.get('diffusion_num_steps_max', 5)))
    if diff_steps[1] <= diff_steps[0]:
        diff_steps = (diff_steps[0], diff_steps[0] + 1)
    enable_gradient_checkpointing(config.get('gradient_checkpointing', False))
    amp = Amp(resolve_precision(config), device=device,
              clip_norm=config.get('grad_clip_norm', None))

    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch

    optimizer_params = Munch(config['optimizer_params'])
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker

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
                                        batch_size_file=osp.join(
                                            log_dir, 'batch_sizes_%s.json' % ('ft' if finetune else '2nd')),
                                        max_batch_frames=max_batch_frames,
                                        max_batch_size=int(config.get('max_batch_size', 64)))
    batch_manager = train_dataloader.batch_manager
    if not dynamic_batch:
        logger.warning('dynamic_batch is off: batches are cropped to their shortest '
                       'utterance (the original behaviour); dynamic_batch: true is recommended')

    # Validation: batch size 1, whole utterances -> deterministic, no cropping.
    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=1,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={'multispeaker': multispeaker})

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load PL-BERT model
    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    # build model
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]

    # The pitch extractor is only ever used under no_grad.  In the second
    # stage the text aligner is too; when fine-tuning it keeps training (TMA
    # losses) because it produces the duration targets on the new data -
    # freezing it there causes stuttered/repeated syllables.
    set_requires_grad(model.pitch_extractor, False)
    model.pitch_extractor.eval()
    if not finetune:
        set_requires_grad(model.text_aligner, False)
        model.text_aligner.eval()

    # DP
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = MyDataParallel(model[key])

    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)

    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, _, _ = load_checkpoint(
                model, None, first_stage_path, load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder',
                                'msd', 'mpd', 'wd', 'diffusion'])
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError('You need to specify the path to the first stage model.')

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model,
                   model.wd,
                   sr,
                   model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
        clamp=False
    )

    # Per-module learning rates / AdamW hyper-parameters (as in the original).
    lr_dict = {key: {'max_lr': optimizer_params.lr} for key in model}
    lr_dict['bert']['max_lr'] = optimizer_params.bert_lr
    lr_dict['decoder']['max_lr'] = optimizer_params.ft_lr
    lr_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                scheduler_params_dict=lr_dict, lr=optimizer_params.lr,
                                use_8bit=config.get('use_8bit_optimizer', False))
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['weight_decay'] = 0.01
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['weight_decay'] = 1e-4

    # load models if there is a model
    if load_pretrained:
        load_only = config.get('load_only_params', True)
        model, optimizer, start_epoch, iters = load_checkpoint(
            model, optimizer, config['pretrained_model'], load_only_params=load_only)
        if not load_only:
            start_epoch += 1  # the saved epoch was completed
            amp.load_state_dict(torch.load(config['pretrained_model'], map_location='cpu',
                                           weights_only=False).get('amp_scaler'))

    lr_sched = ProgressLR(optimizer, epochs,
                          kind=config.get('lr_schedule', 'constant'),
                          warmup=float(config.get('lr_warmup', 0.1)))

    n_down = model.text_aligner.n_down
    style_dim = model_params.style_dim

    best_loss = float('inf')  # best test loss
    stft_loss = MultiResolutionSTFTLoss().to(device)

    # running estimate of sigma_data (mean of the batch-wise std estimates)
    sigma_sum, sigma_count = 0.0, 0

    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(model, wl, sampler,
                                slmadv_params.min_len,
                                slmadv_params.max_len,
                                batch_percentage=slmadv_params.batch_percentage,
                                skip_update=slmadv_params.iter,
                                sig=slmadv_params.sig,
                                diffusion_steps=diff_steps)

    stage = 'ft' if finetune else 'second'
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        n_running = 0
        start_time = time.time()
        diff = epoch >= diff_epoch
        joint = epoch >= joint_epoch
        # In the second stage the discriminators start with the diffusion
        # model; when fine-tuning they train from the start.
        start_ds = finetune or diff
        # decoder / style encoder are trained from the start when fine-tuning,
        # only in joint training otherwise
        train_acoustic = finetune or joint

        _ = [model[key].eval() for key in model]
        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()
        if finetune:
            model.text_aligner.train()
            model.text_encoder.train()
        model.pitch_extractor.eval()

        # Modules that are not updated in this phase do not need parameter
        # gradients: skipping them saves memory and backward time.
        set_requires_grad(model.decoder, train_acoustic)
        set_requires_grad(model.style_encoder, train_acoustic)

        phase = stage + ('_joint' if joint else '_diff' if diff else '')
        if batch_manager is not None:
            batch_manager.set_phase(phase, scale=phase_scale(config, phase))
        if epoch == diff_epoch:
            logger.info('Epoch %d: style diffusion%s starts' % (
                epoch, '' if finetune else ' and adversarial training'))
        if epoch == joint_epoch:
            logger.info('Epoch %d: joint (SLM adversarial) training starts' % epoch)

        steps_in_epoch = max(1, len(train_dataloader))
        n_nonfinite = 0
        n_slm = n_slm_skipped = 0
        for i, batch in enumerate(train_dataloader):
            lr_sched.set_progress(epoch + min(i / steps_in_epoch, 1.0))
            mel_input_length = None
            bs = 0
            try:
                waves = batch[0]
                batch = [b.to(device, non_blocking=True) for b in batch[1:]]
                texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                bs = mels.shape[0]

                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    text_mask = length_to_mask(input_lengths).to(device)

                    # reference styles (multi-speaker diffusion / SLM adversarial)
                    ref = None
                    if multispeaker and (diff or joint):
                        with amp.autocast():
                            ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                            ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                        ref = torch.cat([ref_ss, ref_sp], dim=1).float()

                with torch.set_grad_enabled(finetune), amp.autocast():
                    ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)

                with torch.no_grad():
                    mask_ST = mask_from_lens(s2s_attn.float(), input_lengths, mel_input_length // (2 ** n_down))
                    s2s_attn_mono = maximum_path(s2s_attn.float(), mask_ST)

                with torch.set_grad_enabled(finetune), amp.autocast():
                    t_en = model.text_encoder(texts, input_lengths, text_mask)
                    if finetune and bool(random.getrandbits(1)):
                        # 50% of chance of using the soft (trainable) alignment
                        asr = (t_en @ s2s_attn.to(t_en.dtype))
                    else:
                        asr = (t_en @ s2s_attn_mono.to(t_en.dtype))
                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                with amp.autocast():
                    # style of the entire utterance
                    s_dur_global, gs = global_styles(model, mels, mel_input_length)
                    s_trg = torch.cat([gs, s_dur_global], dim=-1).detach().float()  # denoiser target

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

                # denoiser training
                if diff:
                    num_steps = np.random.randint(*diff_steps)
                    if model_params.diffusion.dist.estimate_sigma_data:
                        sigma = s_trg.std(axis=-1).mean().item()  # batch-wise std estimation
                        model.diffusion.module.diffusion.sigma_data = sigma
                        sigma_sum += sigma
                        sigma_count += 1

                    with amp.autocast():
                        noise = torch.randn_like(s_trg).unsqueeze(1)
                        if multispeaker:
                            s_preds = sampler(noise=noise,
                                              embedding=bert_dur,
                                              embedding_scale=1,
                                              features=ref,  # reference from the same speaker as the embedding
                                              embedding_mask_proba=0.1,
                                              num_steps=num_steps).squeeze(1)
                            loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean()  # EDM loss
                        else:
                            s_preds = sampler(noise=noise,
                                              embedding=bert_dur,
                                              embedding_scale=1,
                                              embedding_mask_proba=0.1,
                                              num_steps=num_steps).squeeze(1)
                            loss_diff = model.diffusion.module.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean()  # EDM loss
                        loss_sty = F.l1_loss(s_preds.float(), s_trg)  # style reconstruction loss
                else:
                    loss_sty = 0
                    loss_diff = 0

                with amp.autocast():
                    d, p = model.predictor(d_en, s_dur_global,
                                           input_lengths,
                                           s2s_attn_mono.to(d_en.dtype),
                                           text_mask)

                en, gt, wav, st, p_en = select_segments(
                    asr, mels, waves, mel_input_length, seg_cap, device, p=p,
                    full_length=dynamic_batch,
                    st_mode='random' if (multispeaker and not finetune) else 'same')

                if gt.size(-1) < 80:
                    continue

                with amp.autocast():
                    s_src = st if (multispeaker and not finetune) else gt
                    s_dur = model.predictor_encoder(s_src.unsqueeze(1))
                    with torch.set_grad_enabled(train_acoustic):
                        s = model.style_encoder(s_src.unsqueeze(1))

                with torch.no_grad(), amp.autocast():
                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                    N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

                    y_rec_gt = wav.unsqueeze(1)
                    if finetune and not joint:
                        y_rec_gt_pred = None  # only needed by SLM adversarial training
                    else:
                        y_rec_gt_pred = model.decoder(en, F0_real, N_real, s).float()

                # target: the recording once the decoder is trained, otherwise
                # its own reconstruction (the decoder is fixed)
                target = y_rec_gt if train_acoustic else y_rec_gt_pred

                with amp.autocast():
                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)
                    y_rec = model.decoder(en, F0_fake, N_fake, s)

                loss_F0_rec = (F.smooth_l1_loss(F0_real.float(), F0_fake.float())) / 10
                loss_norm_rec = F.smooth_l1_loss(N_real.float(), N_fake.float())

                if start_ds and (iters % disc_update_freq) == 0:
                    optimizer.zero_grad()
                    with amp.autocast():
                        d_loss = dl(target.detach(), y_rec.detach()).mean()
                    amp.backward(d_loss)
                    amp.step(optimizer, ['msd', 'mpd'])
                    amp.update()
                else:
                    d_loss = 0

                # generator loss
                optimizer.zero_grad()
                # The discriminators only provide gradients to the generator
                # here; don't build their parameter gradients.
                set_requires_grad(model.msd, False)
                set_requires_grad(model.mpd, False)

                loss_mel = stft_loss(y_rec, target)
                if start_ds:
                    with amp.autocast():
                        loss_gen_all = gl(target, y_rec).mean()
                else:
                    loss_gen_all = 0
                loss_lm = wl(target.detach(), y_rec).mean()

                loss_ce, loss_dur = duration_losses(d, d_gt, input_lengths)

                g_loss = loss_params.lambda_mel * loss_mel + \
                    loss_params.lambda_F0 * loss_F0_rec + \
                    loss_params.lambda_ce * loss_ce + \
                    loss_params.lambda_norm * loss_norm_rec + \
                    loss_params.lambda_dur * loss_dur + \
                    loss_params.lambda_gen * loss_gen_all + \
                    loss_params.lambda_slm * loss_lm + \
                    loss_params.lambda_sty * loss_sty + \
                    loss_params.lambda_diff * loss_diff

                if finetune:
                    loss_s2s = s2s_loss(s2s_pred, texts, input_lengths)
                    loss_mono = F.l1_loss(s2s_attn.float(), s2s_attn_mono) * 10
                    g_loss = g_loss + loss_params.lambda_mono * loss_mono + \
                        loss_params.lambda_s2s * loss_s2s
                else:
                    loss_s2s = loss_mono = 0

                if not all_finite(g_loss):
                    # (the original dropped into an interactive debugger here)
                    n_nonfinite += 1
                    optimizer.zero_grad()
                    set_requires_grad(model.msd, True)
                    set_requires_grad(model.mpd, True)
                    continue

                amp.backward(g_loss)
                set_requires_grad(model.msd, True)
                set_requires_grad(model.mpd, True)

                keys = ['bert_encoder', 'bert', 'predictor', 'predictor_encoder']
                if diff:
                    keys.append('diffusion')
                if train_acoustic:
                    keys += ['style_encoder', 'decoder']
                if finetune:
                    keys += ['text_encoder', 'text_aligner']
                amp.step(optimizer, keys)
                amp.update()

                d_loss_slm, loss_gen_lm = 0, 0
                if joint:
                    # randomly pick whether to use in-distribution text
                    use_ind = np.random.rand() < 0.5
                    if use_ind:
                        ref_lengths = input_lengths
                        ref_texts = texts

                    with amp.autocast():
                        slm_out = slmadv(i,
                                         y_rec_gt,
                                         y_rec_gt_pred,
                                         waves,
                                         mel_input_length,
                                         ref_texts,
                                         ref_lengths, use_ind, s_trg,
                                         ref if multispeaker else None)

                    if slm_out is None:
                        n_slm_skipped += 1
                    else:
                        n_slm += 1
                        d_loss_slm, loss_gen_lm, _ = slm_out

                        # SLM generator loss
                        optimizer.zero_grad()
                        amp.backward(loss_gen_lm)
                        slm_keys = ['bert_encoder', 'bert', 'predictor', 'diffusion']
                        amp.unscale(optimizer, list(model.keys()))

                        # gradient scaling (norm of the predictor's gradient)
                        total_norm = module_grad_norm(model.predictor)
                        if total_norm > slmadv_params.thresh:
                            for key in model.keys():
                                for q in model[key].parameters():
                                    if q.grad is not None:
                                        q.grad *= (1 / total_norm)

                        for module in (model.predictor.duration_proj, model.predictor.lstm, model.diffusion):
                            for q in module.parameters():
                                if q.grad is not None:
                                    q.grad *= slmadv_params.scale

                        amp.step(optimizer, slm_keys)
                        amp.update()

                        # SLM discriminator loss
                        if isinstance(d_loss_slm, torch.Tensor):
                            optimizer.zero_grad()
                            amp.backward(d_loss_slm)
                            amp.step(optimizer, ['wd'])
                            amp.update()

                if all_finite(loss_mel):
                    running_loss += loss_mel.item()
                    n_running += 1
                iters = iters + 1
                if batch_manager is not None:
                    batch_manager.on_success(get_bin_from_mel_input_length(mel_input_length), bs)

                if (i+1) % log_interval == 0:
                    mel_avg = running_loss / max(1, n_running)
                    logger.info('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f, S2S Loss: %.5f, Mono Loss: %.5f'
                                % (epoch+1, epochs, i+1, len(train_dataloader), mel_avg, d_loss, loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_lm, loss_gen_all, loss_sty, loss_diff, d_loss_slm, loss_gen_lm, loss_s2s, loss_mono))

                    writer.add_scalar('train/mel_loss', mel_avg, iters)
                    writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                    writer.add_scalar('train/d_loss', d_loss, iters)
                    writer.add_scalar('train/ce_loss', loss_ce, iters)
                    writer.add_scalar('train/dur_loss', loss_dur, iters)
                    writer.add_scalar('train/slm_loss', loss_lm, iters)
                    writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                    writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                    writer.add_scalar('train/sty_loss', loss_sty, iters)
                    writer.add_scalar('train/diff_loss', loss_diff, iters)
                    writer.add_scalar('train/d_loss_slm', d_loss_slm, iters)
                    writer.add_scalar('train/gen_loss_slm', loss_gen_lm, iters)
                    if finetune:
                        writer.add_scalar('train/s2s_loss', loss_s2s, iters)
                        writer.add_scalar('train/mono_loss', loss_mono, iters)

                    running_loss = 0
                    n_running = 0

                    print('Time elasped:', time.time()-start_time)
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                bin_id = get_bin_from_mel_input_length(mel_input_length) if mel_input_length is not None else None
                logger.warning('WARNING: OOM at step %d (bin %s), skipping batch' % (i, bin_id))
                # Release the partial graph before the next batch allocates.
                batch = waves = texts = mels = ref_mels = ref_texts = None
                ppgs = s2s_pred = s2s_attn = s2s_attn_mono = t_en = asr = ref = None
                s_dur_global = gs = s_trg = bert_dur = d_en = s_preds = None
                loss_diff = loss_sty = d = p = en = gt = wav = st = p_en = None
                s = s_dur = F0_real = F0 = N_real = y_rec_gt = y_rec_gt_pred = target = None
                F0_fake = N_fake = y_rec = d_loss = loss_mel = loss_gen_all = loss_lm = None
                loss_ce = loss_dur = g_loss = loss_s2s = loss_mono = slm_out = None
                d_loss_slm = loss_gen_lm = None
                set_requires_grad(model.msd, True)
                set_requires_grad(model.mpd, True)
                optimizer.zero_grad()
                amp._unscaled.clear()
                free_memory()
                if batch_manager is not None and bin_id is not None:
                    batch_manager.on_oom(bin_id, bs)

        if n_nonfinite:
            logger.warning('Epoch %d: %d batches had a non-finite loss and were skipped'
                           % (epoch + 1, n_nonfinite))
        if joint and n_slm_skipped:
            logger.info('Epoch %d: SLM adversarial step ran on %d batches, skipped on %d '
                        '(utterances or predicted OOD speech shorter than slmadv_params.min_len)'
                        % (epoch + 1, n_slm, n_slm_skipped))

        # ------------------------------------------------------------------
        # Validation (batch size 1, whole utterances)
        # ------------------------------------------------------------------
        loss_test = 0
        loss_align = 0
        loss_f = 0
        iters_test = 0
        n_audio = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(device)

                with amp.autocast():
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)

                mask_ST = mask_from_lens(s2s_attn.float(), input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn.float(), mask_ST)

                with amp.autocast():
                    t_en = model.text_encoder(texts, input_lengths, text_mask)
                    asr = (t_en @ s2s_attn_mono.to(t_en.dtype))
                    d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    s_dur_global, gs = global_styles(model, mels, mel_input_length)

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
                    d, p = model.predictor(d_en, s_dur_global,
                                           input_lengths,
                                           s2s_attn_mono.to(d_en.dtype),
                                           text_mask)

                en, gt, wav, _, p_en = select_segments(asr, mels, waves, mel_input_length,
                                                       None, device, p=p, full_length=True)
                if gt.size(-1) < 80:
                    continue

                with amp.autocast():
                    s_dur = model.predictor_encoder(gt.unsqueeze(1))
                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)
                    s = model.style_encoder(gt.unsqueeze(1))
                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))

                _, loss_dur = duration_losses(d, d_gt, input_lengths)
                loss_mel = stft_loss(y_rec, wav)
                loss_F0 = F.l1_loss(F0_real.float(), F0_fake.float()) / 10
                if not all_finite(loss_mel, loss_dur, loss_F0):
                    continue

                loss_test += loss_mel.item()
                loss_align += loss_dur.item()
                loss_f += loss_F0.item()
                iters_test += 1

                # audio examples for tensorboard
                if n_audio < 6:
                    if epoch < joint_epoch and not finetune:
                        # reconstruction with GT duration
                        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                        with amp.autocast():
                            y_gt_f0 = model.decoder(en, F0_real, real_norm, s)
                        writer.add_audio('eval/y' + str(n_audio), y_gt_f0.float().cpu().numpy().squeeze(), epoch, sample_rate=sr)
                        writer.add_audio('pred/y' + str(n_audio), y_rec.float().cpu().numpy().squeeze(), epoch, sample_rate=sr)
                    elif epoch >= diff_epoch:
                        # speech sampled from text (diffusion style + predicted durations)
                        out = _synthesize(model, sampler, amp, texts, input_lengths, text_mask,
                                          bert_dur, d_en, t_en, ref_mels, multispeaker, style_dim)
                        writer.add_audio('pred/y' + str(n_audio), out, epoch, sample_rate=sr)
                    if epoch == 0 or epoch == start_epoch:
                        writer.add_audio('gt/y' + str(n_audio), waves[0].squeeze(), epoch, sample_rate=sr)
                    n_audio += 1

        iters_test = max(1, iters_test)
        print('Epochs:', epoch + 1)
        logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test) + '\n\n\n')
        print('\n\n\n')
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/dur_loss', loss_align / iters_test, epoch + 1)
        writer.add_scalar('eval/F0_loss', loss_f / iters_test, epoch + 1)

        do_save = ((epoch + 1) % saving_epoch == 0 or epoch == epochs - 1) if finetune \
            else (epoch % saving_epoch == 0 or epoch == epochs - 1)
        if do_save:
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model},
                'optimizer': optimizer.state_dict(),
                'amp_scaler': amp.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)

            # if estimate sigma, save the estimated sigma (only once it exists;
            # the original wrote NaN here before the diffusion epoch)
            if model_params.diffusion.dist.estimate_sigma_data and sigma_count > 0:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(sigma_sum / sigma_count)

                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)


def _synthesize(model, sampler, amp, texts, input_lengths, text_mask, bert_dur, d_en, t_en,
                ref_mels, multispeaker, style_dim):
    """Text-to-speech for the first item of a validation batch."""
    bib = 0
    with amp.autocast():
        noise = torch.randn((1, 2 * style_dim), device=texts.device).unsqueeze(1)
        if multispeaker:
            ref_ss = model.style_encoder(ref_mels[bib:bib + 1].unsqueeze(1))
            ref_sp = model.predictor_encoder(ref_mels[bib:bib + 1].unsqueeze(1))
            ref_s = torch.cat([ref_ss, ref_sp], dim=1).float()
            s_pred = sampler(noise=noise, embedding=bert_dur[bib].unsqueeze(0),
                             embedding_scale=1, features=ref_s, num_steps=5).squeeze(1)
        else:
            s_pred = sampler(noise=noise, embedding=bert_dur[bib].unsqueeze(0),
                             embedding_scale=1, num_steps=5).squeeze(1)

        s = s_pred[:, style_dim:]
        ref = s_pred[:, :style_dim]
        n = int(input_lengths[bib])

        d = model.predictor.text_encoder(d_en[bib, :, :n].unsqueeze(0),
                                         s, input_lengths[bib, ...].unsqueeze(0),
                                         text_mask[bib, :n].unsqueeze(0))
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration.float()).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze(0)).clamp(min=1)
        pred_dur = torch.nan_to_num(pred_dur, nan=1.0).long()
        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(n, int(pred_dur.sum()), device=texts.device)
        c_frame = 0
        for k in range(n):
            pred_aln_trg[k, c_frame:c_frame + int(pred_dur[k])] = 1
            c_frame += int(pred_dur[k])

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(d.dtype))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        asr = t_en[bib, :, :n].unsqueeze(0) @ pred_aln_trg.unsqueeze(0).to(t_en.dtype)
        out = model.decoder(asr, F0_pred, N_pred, ref)
    return out.float().cpu().numpy().squeeze()


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    run(config_path, finetune=False)


if __name__ == "__main__":
    main()

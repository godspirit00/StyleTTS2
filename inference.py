import torch

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import os

# load packages
import random
import yaml
import torchaudio
import librosa
from munch import Munch
from transformers import AlbertConfig

import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

from nltk.tokenize import word_tokenize, sent_tokenize

from models import *
from text_utils import TextCleaner
from Utils.PLBERT.util import CustomAlbert as PlBert

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"


class StyleTTS2Model:
    def __init__(self, config_path='', weights_path = '', device="cpu"):
        self.tokenizer = TextCleaner()

        self.config = recursive_munch(yaml.safe_load(open(config_path, encoding='utf-8')))

        if "plbert_params" in self.config:      ## Compatible with the config from StyleTTS2Inference
            plbert_config=self.config.plbert_params
        else:
            BERT_path = self.config.get('PLBERT_dir', False)
            plbert_config = recursive_munch(yaml.safe_load(open(os.path.join(BERT_path,"config.yml"),encoding="utf-8")))
            plbert_config=plbert_config.model_params
        plbert=PlBert(AlbertConfig(**plbert_config))
        plbert_encoder=torch.nn.Linear(plbert_config.hidden_size, self.config.model_params.hidden_dim)

        args=self.config.model_params

        if args.decoder.type == "istftnet":
            from Modules.istftnet import Decoder
            decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                    resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                    upsample_rates = args.decoder.upsample_rates,
                    upsample_initial_channel=args.decoder.upsample_initial_channel,
                    resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                    upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
                    gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size) 
        else:
            from Modules.hifigan import Decoder
            decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                    resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                    upsample_rates = args.decoder.upsample_rates,
                    upsample_initial_channel=args.decoder.upsample_initial_channel,
                    resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                    upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
            
        text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
        
        predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
        
        style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
        predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder
            
        # define diffusion model
        if args.multispeaker:
            transformer = StyleTransformer1d(channels=args.style_dim*2, 
                                        context_embedding_features=plbert_config.hidden_size,
                                        context_features=args.style_dim*2, 
                                        **args.diffusion.transformer)
        else:
            transformer = Transformer1d(channels=args.style_dim*2, 
                                        context_embedding_features=plbert_config.hidden_size,
                                        **args.diffusion.transformer)
        
        diffusion = AudioDiffusionConditional(
            in_channels=1,
            embedding_max_length=plbert_config.max_position_embeddings,
            embedding_features=plbert_config.hidden_size,
            embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
            channels=args.style_dim*2,
            context_features=args.style_dim*2,
        )
        
        diffusion.diffusion = KDiffusion(
            net=diffusion.unet,
            sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
            sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
            dynamic_threshold=0.0 
        )
        diffusion.diffusion.net = transformer
        diffusion.unet = transformer

        self.model=Munch(
            bert=plbert,
            bert_encoder=plbert_encoder,

            predictor=predictor,
            decoder=decoder,
            text_encoder=text_encoder,

            predictor_encoder=predictor_encoder,
            style_encoder=style_encoder,
            diffusion=diffusion,
       )        
        
        params_whole = torch.load(weights_path, map_location='cpu', weights_only=True)
        if "net" in params_whole:
            params = params_whole['net']
        else:
            params = params_whole


        for key in self.model:
            if key in params:
                print('%s loaded' % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])

        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(device) for key in self.model]

        self.computed_styles={}

        self.multispeaker=self.config['model_params']['multispeaker']

        from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

        self.sampler = DiffusionSampler(
                self.model.diffusion.diffusion,
                sampler=ADPM2Sampler(),
                sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
                clamp=False
            )
        
    def compute_style(self, path):
        if path in self.computed_styles: return self.computed_styles[path]      ## Only compute styles once
        wave, sr = librosa.load(path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        output=torch.cat([ref_s, ref_p], dim=1)

        self.computed_styles[path]=output

        return output

    def inference(self, phonemes:str, ref_s:torch.Tensor, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, speed=1.0, s_prev=None, t=0.7):
        '''
        Inference for Multispeaker models. 

        phonemes: tokenized phonemes of the text to be spoken. 
        '''
        tokens = self.tokenizer(phonemes)
        tokens.insert(0, 0)


        tokens_tensors = torch.LongTensor(tokens).to(device).unsqueeze(0)
        with torch.inference_mode():
            input_lengths = torch.LongTensor([tokens_tensors.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            t_en = self.model.text_encoder(tokens_tensors, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens_tensors, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)


            if s_prev is not None:
            # convex combination of previous and current style
                s_pred = t * s_prev + (1 - t) * s_pred
                
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,
                                            s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_dur[-1] = pred_dur[-1].clamp(max=6)

            if not phonemes[-1].isalnum():  pred_dur[-1] = 0        # Eliminate potential noise at the end of the audio during generation.

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            if self.config.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
            if self.config.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))

            out = out.squeeze()

        return out[..., :int(-100/speed)], s_pred # weird pulse at the end of the model, need to be fixed later


    def inference2(self, phonemes:str, speed=1.0, diffusion_steps=5, embedding_scale=1, s_prev=None, alpha=0.7):
        '''
        Inference for single-speaker models. 

        phonemes: tokenized phonemes of the text to be spoken. 
        '''

        tokens=self.tokenizer(phonemes.strip())
        tokens.insert(0, 0)

        tokens_tensors = torch.LongTensor(tokens).to(device).unsqueeze(0)
        
        with torch.inference_mode():
            input_lengths = torch.LongTensor([tokens_tensors.shape[-1]]).to(tokens_tensors.device)
            text_mask = length_to_mask(input_lengths).to(tokens_tensors.device)

            t_en = self.model.text_encoder(tokens_tensors, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens_tensors, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2) 

            s_pred = self.sampler(noise = torch.randn(1,1,256).to(device), 
                embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                embedding_scale=embedding_scale).squeeze(0)

            if s_prev is not None:
                # convex combination of previous and current style
                s_pred = alpha * s_prev + (1 - alpha) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_dur[-1] = pred_dur[-1].clamp(max=6)

            # pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)), 
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            out=out.squeeze()
            
        return out, s_pred


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser("StyleTTS2 English Models Inference")
    parser.add_argument("-c",type=str,help="Path to a model config YAML",required=True)
    parser.add_argument("-w",type=str,help="Path to a model weight/checkpoint",required=True)
    parser.add_argument("-t",type=str,help="Text to synthesize",required=True)
    parser.add_argument("-s",type=float,help="Speech rate",default=1.,required=False)
    parser.add_argument("-ap",type=str,help="Path to an audio prompt for multispeaker models",default=None,required=False)
    parser.add_argument("-o",type=str,help="Output the speech to",default="output.wav",required=False)
    args=parser.parse_args()

    model=StyleTTS2Model(config_path=args.c, weights_path=args.w, device=device)
    if model.multispeaker: assert args.ap is not None,"An audio prompt is needed for a multispeaker model."

    text = args.t.strip()
    text = text.replace('"', '')
    pace=args.s

    parts=sent_tokenize(text)  # split long text by sentences

    i = 0
    while i < len(parts):
        word_count = len(word_tokenize(parts[i]))
        if word_count < 6:
            if i < len(parts) - 1:
                # 非最后一句：并入下一句
                parts[i+1] = parts[i].rstrip() + " " + parts[i+1].lstrip()
                del parts[i]
                # 不递增 i，继续检查当前位置（已成为原来的下一句）
                continue
            elif i > 0:
                # 最后一小句：并入前一句
                parts[i-1] = parts[i-1].rstrip() + " " + parts[i].lstrip()
                del parts[i]
                break
        i += 1

    s_prev = None

    wavs = []
    for p in parts:
        pss=global_phonemizer.phonemize([p]) 
        pss=" ".join(word_tokenize(pss[0]))
        
        if model.multispeaker:
            ref_s = model.compute_style(args.ap)       
            wav, s_prev = model.inference(pss, ref_s, speed=pace, s_prev=s_prev)
        else:
            wav, s_prev = model.inference2(phonemes=pss, speed=pace, s_prev=s_prev)


        wav = wav.cpu()

        wavs.append(wav)

        if device == "cuda":
            torch.cuda.empty_cache()

    wav_out = torch.concatenate(wavs)
    torchaudio.save(args.o,wav_out.unsqueeze(0), 24000)
    print("Audio saved to %s"%args.o)

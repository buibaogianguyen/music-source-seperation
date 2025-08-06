import torch
import torchaudio
import os
from model.dttnet import DTTNet
from data.preprocessing import Preprocessor
from torch.amp import autocast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer(model, preprocessor, device, input_audio_path="C:/Users/buiba/UNET/UNET/master__a_hound.wav", output_dir="output"):
    try:
        logger.info(f"Loading {input_audio_path}")
        audio, sr = torchaudio.load(input_audio_path)
        if sr != 44100:
            audio = torchaudio.transforms.Resample(sr, 44100)(audio)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        audio = audio.to(device)
        
        segment_len = 44100 * 2  # 2 seconds
        vocals = torch.zeros_like(audio)
        background = torch.zeros_like(audio)
        window = torch.hann_window(segment_len, device=device)
        
        total_length = audio.shape[-1]
        overlap = segment_len // 2
        num_segments = max(1, (total_length - segment_len) // overlap + 1)
        logger.info(f"Processing {num_segments} segments")
        
        model.eval()
        with torch.no_grad():
            for i in range(num_segments):
                start = i * overlap
                end = min(start + segment_len, total_length)
                segment = audio[:, start:end]
                
                if segment.shape[-1] < segment_len:
                    segment = torch.nn.functional.pad(segment, (0, segment_len - segment.shape[-1]))
                
                with autocast('cuda'):
                    mix_spec = preprocessor.waveform_to_spectrogram(segment.unsqueeze(0))
                    mix_mag = preprocessor.normalize_spectrogram(mix_spec)
                    masks = model(mix_mag)
                    vocals_spec = masks[:, :, 0, :, :]
                    bg_spec = masks[:, :, 1, :, :]
                    mix_phase = preprocessor.get_phase(mix_spec)
                    vocals_complex = vocals_spec * torch.exp(1j * mix_phase)
                    bg_complex = bg_spec * torch.exp(1j * mix_phase)
                    pred_vocals = preprocessor.spectrogram_to_waveform(vocals_complex, segment_len)
                    pred_bg = preprocessor.spectrogram_to_waveform(bg_complex, segment_len)
                
                pred_vocals = pred_vocals.squeeze(0) * window[:pred_vocals.shape[-1]]
                pred_bg = pred_bg.squeeze(0) * window[:pred_bg.shape[-1]]
                vocals[:, start:end] += pred_vocals[:, :end-start]
                background[:, start:end] += pred_bg[:, :end-start]
        
        overlap_count = torch.ones_like(audio)
        for i in range(num_segments):
            start = i * overlap
            end = min(start + segment_len, total_length)
            overlap_count[:, start:end] += window[:end-start]
        vocals = vocals / (overlap_count + 1e-8)
        background = background / (overlap_count + 1e-8)
        
        # Normalize amplitudes
        vocals = vocals / (vocals.abs().max() + 1e-8)
        background = background / (background.abs().max() + 1e-8)
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        torchaudio.save(os.path.join(output_dir, "vocals.wav"), vocals.cpu(), 44100)
        torchaudio.save(os.path.join(output_dir, "background.wav"), background.cpu(), 44100)
        logger.info(f"Saved to {output_dir}/vocals.wav and {output_dir}/background.wav")
        
        return vocals, background
    
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DTTNet(in_channels=2, num_sources=2, fft_bins=1024).to(device)
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        preprocessor = Preprocessor(fft_bins=1024, hop_len=256, sample_rate=44100)
        infer(model, preprocessor, device, input_audio_path="C:/Users/buiba/UNET/UNET/master__a_hound.wav", output_dir="output")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise
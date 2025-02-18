# Zonos.py
# Resource: https://github.com/sdbds/Zonos-for-windows/blob/main/gradio_interface.py

import os.path
from .Install import Install

import sys
import torch
import torchaudio
import hashlib
import folder_paths
import tempfile
import io
import re
import shutil
import subprocess

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale

import torch._dynamo.config
import torch._inductor.config

zonos_path  = os.path.join(Install.zonosPath)
sys.path.insert(0, zonos_path)

Install.check_install()

from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as DEFAULT_BACKBONE
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.backbone import BACKBONES

device = mm.get_torch_device()

ZONOS_MODEL_PATH = os.path.join(folder_paths.models_dir, "zonos")

# Ensure directories exist
os.makedirs(ZONOS_MODEL_PATH, exist_ok=True)

def check_espeak_installation():
    """Check espeak installation and return the path"""
    try:
        # First check if espeak is in PATH
        espeak_path = shutil.which('espeak')
        if espeak_path:
            return espeak_path
            
        # Check common Windows installation paths
        common_paths = [
            r"C:\Program Files\eSpeak NG\espeak-ng.exe",
            r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                # Verify it works by running a simple test
                try:
                    subprocess.run([path, "--version"], capture_output=True, text=True)
                    return path
                except:
                    continue
                    
        return None
    except Exception:
        return None

class ZonosGenerate:
    voice_reg = re.compile(r"\{([^\}]+)\}")
    model_types = ["Zyphra/Zonos-v0.1-transformer", "Zyphra/Zonos-v0.1-hybrid"]
    tooltip_seed = "Seed. -1 = random"
    tooltip_speed = "Speed. >1.0 slower. <1.0 faster"
    
    # Move class variables to top level for clarity
    CURRENT_MODEL_TYPE = None
    CURRENT_MODEL = None
    CURRENT_SPEAKER_HASH = None
    CURRENT_SPEAKER_EMBEDDING = None
    
    @staticmethod
    def hash_audio(waveform, sample_rate):
        """Compute hash of audio content for caching"""
        m = hashlib.sha256()
        m.update(waveform.cpu().numpy().tobytes())
        m.update(str(sample_rate).encode())
        return m.digest().hex()

    @staticmethod
    def get_model_path(model_name):
        """Get paths to model and config files, downloading if needed"""
        if model_name not in ZonosGenerate.model_types:
            raise ValueError(f"Invalid model type: {model_name}")
            
        # Create model-specific directory
        model_dir = os.path.join(ZONOS_MODEL_PATH, model_name.split('/')[-1])
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = "model.safetensors"
        config_file = "config.json"
        model_path = os.path.join(model_dir, model_file)
        config_path = os.path.join(model_dir, config_file)
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"Downloading Zonos model {model_name} to: {model_dir}")
            from huggingface_hub import snapshot_download
            
            # Download model files
            snapshot_download(
                repo_id=model_name,
                allow_patterns=["*.safetensors", "*.json"],
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            
            # Download speaker embedding model if not exists
            speaker_model = "Zyphra/Zonos-v0.1-speaker-embedding"
            speaker_path = os.path.join(ZONOS_MODEL_PATH, "speaker_embedding.safetensors")
            speaker_config = os.path.join(ZONOS_MODEL_PATH, "config.json")
            
            if not os.path.exists(speaker_path) or not os.path.exists(speaker_config):
                print(f"Downloading speaker embedding model to: {ZONOS_MODEL_PATH}")
                snapshot_download(
                    repo_id=speaker_model,
                    allow_patterns=["*.safetensors", "*.json"],
                    local_dir=ZONOS_MODEL_PATH,
                    local_dir_use_symlinks=False
                )
            
        return model_path, config_path

    def is_voice_name(self, word):
        return self.voice_reg.match(word.strip())

    def split_text(self, speech):
        reg1 = r"(?=\{[^\}]+\})"
        return re.split(reg1, speech)

    def generate_audio(self, voices, chunks, seed, language="en-us", cfg_scale=2.0, min_p=0.15, audio_prefix_codes=None):
        if seed >= 0:
            torch.manual_seed(seed)
        else:
            torch.random.seed()

        generated_audio_segments = []
        
        for text in chunks:
            match = self.is_voice_name(text)
            if match:
                voice = match[1]
            else:
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
                
            text = ZonosGenerate.voice_reg.sub("", text)
            gen_text = text.strip()
            if gen_text == "":
                continue

            voice_data = voices[voice]
            model = voice_data["model"]
            speaker = voice_data["speaker"]

            # Handle emotion properly - ensure it's a tensor or None
            emotion = voice_data.get("emotion")
            if emotion is None:
                # Create default emotion tensor with neutral emphasis
                emotion = torch.tensor([
                    0.0,  # happy
                    0.0,  # sad
                    0.0,  # disgust
                    0.0,  # fear
                    0.0,  # surprise
                    0.0,  # anger
                    0.0,  # other
                    1.0   # neutral
                ], device=device, dtype=torch.float32)
            elif not isinstance(emotion, torch.Tensor):
                print("Warning: Invalid emotion format, using neutral")
                emotion = None

            # Add emotion to conditioning if provided
            cond_dict = make_cond_dict(
                text=gen_text,
                speaker=speaker,
                language=voice_data.get("language", language),
                emotion=emotion,
                speaker_noised=voice_data.get("speaker_noised", False)
            )
            
            conditioning = model.prepare_conditioning(cond_dict)

            estimated_generation_duration = 30 * len(text) / 400
            estimated_total_steps = int(estimated_generation_duration * 86)
            pbar = ProgressBar(estimated_total_steps)

            def update_progress(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
                pbar.update(1)
                return True

            # Generate audio with the parameters
            codes = model.generate(
                prefix_conditioning=conditioning,
                audio_prefix_codes=audio_prefix_codes,
                cfg_scale=cfg_scale,
                sampling_params=dict(min_p=min_p),
                callback=update_progress
            )
            wavs = model.autoencoder.decode(codes).cpu()
            
            generated_audio_segments.append(wavs[0])

        if generated_audio_segments:
            final_wave = torch.cat(generated_audio_segments, dim=-1)
            
        return final_wave, model.autoencoder.sampling_rate

    @classmethod
    def get_supported_models(cls):
        """Get list of supported model types"""
        # Always return both models since we handle downloading
        return cls.model_types

    @classmethod
    def INPUT_TYPES(s):
        """Define input types for the node"""
        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "sample_text": ("STRING", {
                    "multiline": True,
                    "default": "Text of sample_audio"
                }),
                "speech": ("STRING", {
                    "multiline": True,
                    "default": "This is what I want to say"
                }),
                "seed": ("INT", {
                    "display": "number", 
                    "step": 1,
                    "default": 1, 
                    "min": -1,
                    "tooltip": s.tooltip_seed,
                }),
                "model_type": (s.model_types,),
                "language": (supported_language_codes,),
                "cfg_scale": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "min_p": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "tooltip": s.tooltip_speed,
                    "step": 0.01
                }),
                "disable_compiler": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable PyTorch compiler for better compatibility"
                }),
            },
            "optional": {
                "prefix_audio": ("AUDIO", {
                    "tooltip": "Optional audio to continue from"
                }),
                "speaker_noised": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply denoising to speaker reference"
                }),
                "emotion": ("EMOTION",),
            }
        }

    CATEGORY = "audio"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "create_audio"

    def disable_torch_compiler(self):
        """Disable PyTorch's inductor compiler and configure C++ environment"""
        # Configure PyTorch to avoid C++ compilation issues
        torch._dynamo.config.suppress_errors = True
        torch._inductor.config.fallback_random = True
        torch._inductor.config.cpp.enable_kernel_profile = False
        torch._inductor.config.triton.unique_kernel_names = True
        torch.backends.cudnn.enabled = True
        torch._dynamo.config.automatic_dynamic_shapes = False
        torch.set_float32_matmul_precision('high')

    def create_audio(self, sample_audio, sample_text, speech, seed=-1, 
                    model_type="Zyphra/Zonos-v0.1-transformer", 
                    language="en-us", cfg_scale=2.0, min_p=0.15,
                    speed=1.0, disable_compiler=True, prefix_audio=None, 
                    speaker_noised=False, emotion=None):        
        try:
            # Only disable compiler if explicitly requested
            if disable_compiler:
                self.disable_torch_compiler()

            # Check espeak installation before proceeding
            espeak_path = check_espeak_installation()
            if not espeak_path:
                raise RuntimeError("espeak is not installed or not found. Please install espeak-ng:\n"
                                 "1. Download from https://github.com/espeak-ng/espeak-ng/releases\n"
                                 "2. Run the installer\n"
                                 "3. Restart your computer\n"
                                 "If already installed, ensure it's in your system PATH")
            
            # Set environment variable for phonemizer
            os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_path

            # Create temporary wav file
            wave_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wave_file_name = wave_file.name
            wave_file.close()

            # Process sample audio and save to temp file
            hasAudio = False
            for (batch_number, waveform) in enumerate(sample_audio["waveform"].cpu()):
                # Compute hash of audio content
                audio_hash = self.hash_audio(waveform, sample_audio["sample_rate"])
                
                buff = io.BytesIO()
                torchaudio.save(buff, waveform, sample_audio["sample_rate"], format="WAV")
                with open(wave_file_name, 'wb') as f:
                    f.write(buff.getbuffer())
                hasAudio = True
                break
                
            if not hasAudio:
                raise FileNotFoundError("No audio input")

            # Get model and config paths
            model_path, config_path = self.get_model_path(model_type)
            
            # Cache model loading
            if self.CURRENT_MODEL_TYPE != model_type:
                # TODO: Use model = mm.load_model(model, device)

                if self.CURRENT_MODEL is not None:
                    del self.CURRENT_MODEL
                    torch.cuda.empty_cache()
                print(f"Loading {model_type} model...")
                self.CURRENT_MODEL = Zonos.from_local(
                    model_path=model_path,
                    config_path=config_path, 
                    device=device,
                    backbone="torch"
                )
                self.CURRENT_MODEL_TYPE = model_type
                print(f"{model_type} model loaded successfully!")
            
            model = self.CURRENT_MODEL

            # Improve speaker embedding caching logic using content hash
            if audio_hash != self.CURRENT_SPEAKER_HASH:
                print("Recomputing speaker embedding...")
                wav, sampling_rate = torchaudio.load(wave_file_name)
                self.CURRENT_SPEAKER_EMBEDDING = model.make_speaker_embedding(wav, sampling_rate)
                self.CURRENT_SPEAKER_EMBEDDING = self.CURRENT_SPEAKER_EMBEDDING.to(device, dtype=torch.bfloat16)
                self.CURRENT_SPEAKER_HASH = audio_hash
            
            speaker = self.CURRENT_SPEAKER_EMBEDDING
            
            main_voice = {
                "speaker": speaker,
                "language": language,
                "model": model,
                "emotion": emotion,
                "speaker_noised": speaker_noised
            }
            
            voices = {'main': main_voice}
            chunks = self.split_text(speech)

            # Process prefix audio if provided
            audio_prefix_codes = None
            if prefix_audio is not None:
                wav_prefix = prefix_audio["waveform"].mean(0, keepdim=True)
                wav_prefix = model.autoencoder.preprocess(wav_prefix, prefix_audio["sample_rate"])
                wav_prefix = wav_prefix.to(device, dtype=torch.float32)
                audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))

            # Generate audio with all parameters
            waveform, sample_rate = self.generate_audio(
                voices, chunks, seed, language,
                cfg_scale=cfg_scale,
                min_p=min_p,
                audio_prefix_codes=audio_prefix_codes
            )

            # Apply speed adjustment if needed
            if speed != 1:
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
                audio = self.time_shift(audio, speed)
            else:
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}

        finally:
            if wave_file_name is not None:
                try:
                    os.unlink(wave_file_name)
                except Exception as e:
                    print("Zonos: Cannot remove? "+wave_file_name)
                    print(e)

        return (audio,)

    @classmethod
    def IS_CHANGED(s, sample_audio, sample_text, speech, seed, model_type, language, 
                   cfg_scale, min_p, speed, disable_compiler=True, prefix_audio=None, 
                   speaker_noised=False, emotion=None):
        """Calculate hash for caching based on all input parameters"""
        m = hashlib.sha256()
        m.update(sample_text.encode())
        m.update(str(sample_audio).encode())
        m.update(speech.encode())
        m.update(str(seed).encode())
        m.update(model_type.encode())
        m.update(language.encode())
        m.update(str(cfg_scale).encode())
        m.update(str(min_p).encode())
        m.update(str(speed).encode())
        m.update(str(disable_compiler).encode())
        if prefix_audio is not None:
            m.update(str(prefix_audio).encode())
        m.update(str(speaker_noised).encode())
        if emotion is not None:
            m.update(str(emotion).encode())
        return m.digest().hex()

    def time_shift(self, audio, speed):
        import torch.nn.functional as F
        rate = audio['sample_rate']
        waveform = audio['waveform']
        
        # Calculate new length
        old_length = waveform.shape[-1]
        new_length = int(old_length / speed)
        
        # Handle waveform dimensions properly
        # Reshape to [batch, channel, time] format if needed
        if waveform.dim() == 4:  # [batch, extra_dim, channels, time]
            b, e, c, t = waveform.shape
            waveform = waveform.reshape(b * e, c, t)
        
        # Resample audio
        new_waveform = F.interpolate(
            waveform,
            size=new_length,
            mode='linear',
            align_corners=False
        )
        
        # Restore original shape if it was 4D
        if audio['waveform'].dim() == 4:
            b, e, c, _ = audio['waveform'].shape
            new_waveform = new_waveform.reshape(b, e, c, new_length)

        return {"waveform": new_waveform, "sample_rate": rate}

class ZonosEmotion:
    """Node for creating emotion vectors for Zonos TTS"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "happy": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Happiness intensity"
                }),
                "sad": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Sadness intensity"
                }),
                "disgust": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Disgust intensity"
                }),
                "fear": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Fear intensity"
                }),
                "surprise": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Surprise intensity"
                }),
                "anger": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Anger intensity"
                }),
                "other": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Other emotion intensity"
                }),
                "neutral": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Neutral intensity"
                }),
            }
        }

    CATEGORY = "audio"
    RETURN_TYPES = ("EMOTION",)
    FUNCTION = "create_emotion"

    def create_emotion(self, happy, sad, disgust, fear, surprise, anger, other, neutral):
        # Normalize values to ensure they sum to 1.0
        total = happy + sad + disgust + fear + surprise + anger + other + neutral
        if total > 0:
            happy = happy / total
            sad = sad / total
            disgust = disgust / total
            fear = fear / total
            surprise = surprise / total
            anger = anger / total
            other = other / total
            neutral = neutral / total

        # Create tensor in exact same order as gradio interface
        emotion_tensor = torch.tensor([
            happy,      # e1 - Happiness
            sad,        # e2 - Sadness
            disgust,    # e3 - Disgust
            fear,       # e4 - Fear
            surprise,   # e5 - Surprise
            anger,      # e6 - Anger
            other,      # e7 - Other
            neutral     # e8 - Neutral
        ], device=device, dtype=torch.float32)  # Add explicit dtype to match gradio
        
        return (emotion_tensor,)

    @classmethod
    def IS_CHANGED(s, happy, sad, disgust, fear, surprise, anger, other, neutral):
        m = hashlib.sha256()
        for val in [happy, sad, disgust, fear, surprise, anger, other, neutral]:
            m.update(str(val).encode())
        return m.digest().hex()
    
# EOF
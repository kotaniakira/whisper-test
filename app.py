import gradio as gr
import os
import torch
import gc
import whisper
import librosa
import soundfile as sf
import numpy as np
import time
import threading
import datetime

# キャッシュディレクトリ
CACHE_DIR = "./models"
os.makedirs(CACHE_DIR, exist_ok=True)

# 排他制御用のロック
processing_lock = threading.Lock()

# グローバル変数
current_model = None
current_model_name = ""
current_engine = ""
processor = None

def release_memory():
    """現在のモデルをメモリから解放する"""
    global current_model, processor, current_model_name, current_engine
    if current_model is not None:
        del current_model
        current_model = None
    if processor is not None:
        del processor
        processor = None
    current_model_name = ""
    current_engine = ""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory released.")

def save_transcription(text):
    """文字起こし結果をテキストファイルに保存する"""
    if not text:
        return None
    
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("outputs", f"transcription_{timestamp}.txt")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    
    return filepath

# --- Whisper Engine ---
def load_whisper(model_name):
    release_memory()
    print(f"Loading Whisper model '{model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device, download_root=CACHE_DIR)
    return model, "whisper"

def transcribe_whisper(model, audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

# --- NVIDIA NeMo Engine ---
def load_nemo(model_name):
    import nemo.collections.asr as nemo_asr
    release_memory()
    print(f"Loading NeMo model '{model_name}'...")
    try:
        if "canary" in model_name:
            model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=model_name)
        elif "rnnt" in model_name:
            model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)
        else:
            model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)
        if torch.cuda.is_available():
            model = model.cuda()
        return model, "nemo"
    except Exception as e:
        print(f"Failed to load NeMo model: {e}")
        raise e

def transcribe_nemo(model, audio_path):
    print(f"Transcribing with NeMo: {audio_path}")
    try:
        is_canary = False
        if hasattr(model, 'cfg') and hasattr(model.cfg, 'target'):
             if "canary" in str(model.cfg.target):
                 is_canary = True
        if is_canary:
            predicted_text = model.transcribe(paths2audio_files=[audio_path])[0]
        elif hasattr(model, 'transcribe'):
            try:
                predicted_text = model.transcribe(paths2audio_files=[audio_path])[0]
            except TypeError:
                predicted_text = model.transcribe([audio_path])[0]
        else:
            return "Error: Model does not support transcribe method."
    except Exception as e:
        return f"NeMo Error: {str(e)}"
    if isinstance(predicted_text, list):
        return predicted_text[0]
    return predicted_text

# --- Hugging Face Transformers ---
def load_transformers(model_name):
    from transformers import pipeline
    release_memory()
    print(f"Loading Transformers model '{model_name}'...")
    device = 0 if torch.cuda.is_available() else -1
    try:
        pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)
        return pipe, "transformers"
    except Exception as e:
        raise e

def transcribe_transformers(pipe, audio_path):
    print(f"Transcribing with Transformers: {audio_path}")
    result = pipe(audio_path)
    return result["text"]

# --- Meta Seamless M4T ---
def load_seamless(model_name):
    from transformers import AutoProcessor, SeamlessM4TModel, SeamlessM4Tv2Model
    release_memory()
    print(f"Loading Seamless M4T model '{model_name}'...")
    global processor
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        if "v2" in model_name:
            model = SeamlessM4Tv2Model.from_pretrained(model_name)
        else:
            model = SeamlessM4TModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return model, "seamless"
    except Exception as e:
        print(f"Failed to load Seamless M4T: {e}")
        raise e

def transcribe_seamless(model, audio_path):
    print(f"Transcribing with Seamless M4T: {audio_path}")
    global processor
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audios=audio, return_tensors="pt", sampling_rate=16000).to(model.device)
    output_tokens = model.generate(**inputs, tgt_lang="jpn", generate_speech=False)
    transcription = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    return transcription

# --- Main Logic ---
def process_audio(audio, model_selection):
    global current_model, current_model_name, current_engine, processor
    if audio is None:
        return "No audio provided.", ""
    
    if processing_lock.locked():
        print("!!! Another user is currently processing. Waiting for lock... !!!")
        
    with processing_lock:
        print(f"Start processing for: {model_selection}")
        start_time = time.time()
        if not model_selection:
            model_selection = "whisper-base"
        try:
            if current_model_name != model_selection:
                if model_selection.startswith("whisper-"):
                    w_name = model_selection.replace("whisper-", "")
                    current_model, current_engine = load_whisper(w_name)
                elif model_selection.startswith("nvidia/"):
                    current_model, current_engine = load_nemo(model_selection)
                elif "wav2vec2" in model_selection:
                    current_model, current_engine = load_transformers(model_selection)
                elif "seamless-m4t" in model_selection:
                    current_model, current_engine = load_seamless(model_selection)
                current_model_name = model_selection

            result_text = ""
            if current_engine == "whisper":
                result_text = transcribe_whisper(current_model, audio)
            elif current_engine == "nemo":
                result_text = transcribe_nemo(current_model, audio)
            elif current_engine == "transformers":
                result_text = transcribe_transformers(current_model, audio)
            elif current_engine == "seamless":
                result_text = transcribe_seamless(current_model, audio)
            else:
                result_text = "Unknown engine error."
            
            elapsed_time = time.time() - start_time
            time_info = f"⏱️ Time: {elapsed_time:.2f} sec"
            print(f"Processing finished. Time: {elapsed_time:.2f} sec")
            return result_text, time_info
        except Exception as e:
            import traceback
            traceback.print_exc()
            elapsed_time = time.time() - start_time
            return f"Error: {str(e)}", f"⏱️ Time: {elapsed_time:.2f} sec (Failed)"

# --- UI Definition ---
model_choices = [
    "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3",
    "nvidia/parakeet-rnnt-1.1b", "nvidia/parakeet-ctc-1.1b", "nvidia/parakeet-tdt_ctc-0.6b-ja", "nvidia/canary-1b",
    "facebook/seamless-m4t-v2-large", "facebook/seamless-m4t-medium",
    "facebook/wav2vec2-large-960h", "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
]

with gr.Blocks(title="Universal Speech Recognition Web UI") as demo:
    gr.Markdown("# Universal Speech Recognition Web UI")
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Input")
            model_dropdown = gr.Dropdown(choices=model_choices, value="whisper-base", label="Select Model")
            submit_btn = gr.Button("Transcribe", variant="primary")
        with gr.Column(scale=1):
            output_text = gr.Textbox(lines=15, label="Transcription Result")
            time_output = gr.Label(label="Processing Time")
            download_btn = gr.Button("Download Text Result")
            download_file = gr.File(label="Download File")
            download_btn.click(save_transcription, inputs=output_text, outputs=download_file)
    submit_btn.click(fn=process_audio, inputs=[audio_input, model_dropdown], outputs=[output_text, time_output])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(share=True)
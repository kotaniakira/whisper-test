import gradio as gr
import os
import torch
import gc
import whisper
import librosa
import soundfile as sf
import numpy as np
import time

# キャッシュディレクトリ
CACHE_DIR = "./models"
os.makedirs(CACHE_DIR, exist_ok=True)

# グローバル変数
current_model = None
current_model_name = ""
current_engine = ""  # "whisper", "nemo", "transformers", "phi4", "seamless"
processor = None     # Transformers / Phi4 / Seamless 用

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
        # NeMoのバージョン互換性対応
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

# --- Hugging Face Transformers (Wav2Vec2) ---
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

# --- Microsoft Phi-4 Multimodal ---
def load_phi4(model_name):
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    release_memory()
    print(f"Loading Phi-4 model '{model_name}'...")
    
    global processor
    
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.float16, 
            device_map="auto",
            _attn_implementation='eager'
        )
        return model, "phi4"
    except Exception as e:
        print(f"Failed to load Phi-4: {e}")
        raise e

def transcribe_phi4(model, audio_path):
    print(f"Transcribing with Phi-4: {audio_path}")
    global processor
    
    audio, sr = librosa.load(audio_path, sr=16000)
    prompt = "<|user|>\n<|audio_1|>Transcribe this audio to text.<|end|>\n<|assistant|>"
    inputs = processor(text=prompt, audios=audio, return_tensors="pt").to(model.device)
    
    generate_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    generate_ids = generate_ids[:, inputs.input_ids.shape[1]:] 
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response

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
    
    # 16kHz リサンプリング
    audio, sr = librosa.load(audio_path, sr=16000)
    
    inputs = processor(audios=audio, return_tensors="pt", sampling_rate=16000).to(model.device)
    
    # tgt_lang="jpn" で日本語として出力させる (ASR)
    # 英語にしたい場合は "eng"
    output_tokens = model.generate(**inputs, tgt_lang="jpn", generate_speech=False)
    transcription = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    
    return transcription


# --- Main Logic ---

def process_audio(audio, model_selection):
    global current_model, current_model_name, current_engine, processor
    
    if audio is None:
        return "No audio provided.", ""
    
    start_time = time.time()
    
    if not model_selection:
        model_selection = "whisper-base"

    try:
        # モデルロード判定
        if current_model_name != model_selection:
            
            if model_selection.startswith("whisper-"):
                w_name = model_selection.replace("whisper-", "")
                current_model, current_engine = load_whisper(w_name)
                
            elif model_selection.startswith("nvidia/"):
                current_model, current_engine = load_nemo(model_selection)
                
            elif "wav2vec2" in model_selection:
                current_model, current_engine = load_transformers(model_selection)
                
            elif "Phi-4" in model_selection:
                current_model, current_engine = load_phi4(model_selection)
            
            elif "seamless-m4t" in model_selection:
                current_model, current_engine = load_seamless(model_selection)
                
            current_model_name = model_selection

        # 推論実行
        result_text = ""
        if current_engine == "whisper":
            result_text = transcribe_whisper(current_model, audio)
        elif current_engine == "nemo":
            result_text = transcribe_nemo(current_model, audio)
        elif current_engine == "transformers":
            result_text = transcribe_transformers(current_model, audio)
        elif current_engine == "phi4":
            result_text = transcribe_phi4(current_model, audio)
        elif current_engine == "seamless":
            result_text = transcribe_seamless(current_model, audio)
        else:
            result_text = "Unknown engine error."
            
        elapsed_time = time.time() - start_time
        time_info = f"⏱️ Time: {elapsed_time:.2f} sec"
        
        return result_text, time_info
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        elapsed_time = time.time() - start_time
        return f"Error: {str(e)}", f"⏱️ Time: {elapsed_time:.2f} sec (Failed)"

# --- UI Definition ---

model_choices = [
    # Whisper
    "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3",
    # NVIDIA NeMo
    "nvidia/parakeet-rnnt-1.1b",
    "nvidia/parakeet-ctc-1.1b",
    "nvidia/parakeet-tdt_ctc-0.6b-ja",
    "nvidia/canary-1b",
    # Microsoft Phi-4
    "microsoft/Phi-4-multimodal-instruct",
    # Meta Seamless M4T
    "facebook/seamless-m4t-v2-large",
    "facebook/seamless-m4t-medium",
    # Transformers
    "facebook/wav2vec2-large-960h",
    "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
]

with gr.Blocks(title="Universal Speech Recognition Web UI") as demo:
    gr.Markdown("# Universal Speech Recognition Web UI")
    gr.Markdown("OpenAI Whisper, NVIDIA NeMo, Microsoft Phi-4, Meta Seamless M4T, Wav2Vec2 を切り替えて試せます。")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Input")
            model_dropdown = gr.Dropdown(choices=model_choices, value="whisper-base", label="Select Model")
            submit_btn = gr.Button("Transcribe", variant="primary")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(lines=15, label="Transcription Result")
            time_output = gr.Label(label="Processing Time")
            
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, model_dropdown],
        outputs=[output_text, time_output]
    )

if __name__ == "__main__":
    # concurrency_count=1 に設定することで、同時に1つの処理しか走らないようにする（他はキュー待ち）
    demo.queue(default_concurrency_limit=1).launch(share=True)

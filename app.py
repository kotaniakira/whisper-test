import gradio as gr
import os
import torch
import gc
import whisper
import librosa
import soundfile as sf
import numpy as np

# キャッシュディレクトリ
CACHE_DIR = "./models"
os.makedirs(CACHE_DIR, exist_ok=True)

# グローバル変数
current_model = None
current_model_name = ""
current_engine = ""  # "whisper", "nemo", "transformers", "phi4"
processor = None     # Transformers / Phi4 用

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
        # device_map="auto" には accelerate が必要
        # fp16 でロードしてメモリ節約
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.float16, 
            device_map="auto",
            _attn_implementation='eager' # 環境によっては flash_attn がないので eager 指定
        )
        return model, "phi4"
    except Exception as e:
        print(f"Failed to load Phi-4: {e}")
        raise e

def transcribe_phi4(model, audio_path):
    print(f"Transcribing with Phi-4: {audio_path}")
    global processor
    
    # 音声の読み込みとリサンプリング (16kHz)
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # プロンプト作成 (一行で記述)
    prompt = "<|user|>\n<|audio_1|>Transcribe this audio to text.<|end|>\n<|assistant|>"
    
    inputs = processor(text=prompt, audios=audio, return_tensors="pt").to(model.device)
    
    # 生成
    generate_ids = model.generate(
        **inputs, 
        max_new_tokens=500, 
        do_sample=False  # 決定論的にする
    )
    
    # デコード (入力プロンプト部分は除外)
    # Phi-4 の出力フォーマットに合わせて調整が必要な場合があるが、まずは標準的にデコード
    generate_ids = generate_ids[:, inputs.input_ids.shape[1]:] 
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response

# --- Main Logic ---

def process_audio(audio, model_selection):
    global current_model, current_model_name, current_engine, processor
    
    if audio is None:
        return "No audio provided."
    
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
                
            current_model_name = model_selection

        # 推論実行
        if current_engine == "whisper":
            return transcribe_whisper(current_model, audio)
        elif current_engine == "nemo":
            return transcribe_nemo(current_model, audio)
        elif current_engine == "transformers":
            return transcribe_transformers(current_model, audio)
        elif current_engine == "phi4":
            return transcribe_phi4(current_model, audio)
        else:
            return "Unknown engine error."
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

# --- UI Definition ---

model_choices = [
    # Whisper
    "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3",
    # NVIDIA NeMo
    "nvidia/parakeet-rnnt-1.1b",
    "nvidia/parakeet-ctc-1.1b",
    "nvidia/parakeet-tdt_ctc-0.6b-ja",
    "nvidia/canary-1b",
    # Microsoft Phi-4 (Multimodal)
    "microsoft/Phi-4-multimodal-instruct",
    # Transformers
    "facebook/wav2vec2-large-960h",
    "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
]

demo = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Input"),
        gr.Dropdown(choices=model_choices, value="whisper-base", label="Select Model")
    ],
    outputs=gr.Textbox(lines=15, label="Transcription Result"),
    title="Universal Speech Recognition Web UI",
    description="""
    OpenAI Whisper, NVIDIA NeMo, Microsoft Phi-4, Meta Wav2Vec2 を切り替えて試せます。
    
    - **Whisper**: 多言語対応、高精度。
    - **NVIDIA Parakeet/Canary**: 高速・高精度。
    - **Microsoft Phi-4**: マルチモーダル指示モデル。"Transcribe..."と指示して動作させています。
    - **Wav2Vec2**: 従来型モデル。
    """
)

if __name__ == "__main__":
    demo.launch(share=True)

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
import shutil
import uuid
import math
from sudachipy import dictionary
from sudachipy import tokenizer

# キャッシュディレクトリ
CACHE_DIR = "./models"
os.makedirs(CACHE_DIR, exist_ok=True)
TEMP_DIR = "./temp_chunks"
os.makedirs(TEMP_DIR, exist_ok=True)

# Sudachiの初期化
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

# 排他制御用のロック
processing_lock = threading.Lock()

# グローバル変数
current_model = None
current_model_name = ""
current_engine = ""
processor = None

# --- Helper Utilities ---

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

def force_garbage_collection():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_transcription(text):
    if not text:
        return None
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("outputs", f"transcription_{timestamp}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath

def get_reading(text):
    if not text:
        return ""
    tokens = tokenizer_obj.tokenize(text, mode)
    reading = "".join([m.reading_form() for m in tokens])
    return reading

def save_reading(text):
    if not text:
        return None
    reading_text = get_reading(text)
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("outputs", f"reading_{timestamp}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(reading_text)
    return filepath

class AudioChunker:
    """
    音声を読み込み、指定した秒数（chunk_sec）ごとに分割して一時ファイルとして保存するクラス。
    """
    def __init__(self, audio_path, chunk_sec=30, sr=16000):
        self.audio_path = audio_path
        self.chunk_sec = chunk_sec
        self.sr = sr
        self.temp_files = []
        self.y = None
        self.total_duration = 0

    def load_and_split(self):
        print(f"Loading and splitting audio: {self.audio_path} (chunk_sec={self.chunk_sec})")
        # リサンプリングして読み込み
        y, sr = librosa.load(self.audio_path, sr=self.sr, mono=True)
        self.y = y
        self.total_duration = len(y) / sr
        
        # 分割
        chunk_length = int(self.chunk_sec * sr)
        total_samples = len(y)
        num_chunks = math.ceil(total_samples / chunk_length)
        
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(TEMP_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        for i in range(num_chunks):
            start = i * chunk_length
            end = min((i + 1) * chunk_length, total_samples)
            chunk_data = y[start:end]
            
            chunk_filename = os.path.join(session_dir, f"chunk_{i:04d}.wav")
            sf.write(chunk_filename, chunk_data, sr)
            self.temp_files.append(chunk_filename)
            
        print(f"Split into {len(self.temp_files)} chunks. Total duration: {self.total_duration:.2f}s")
        return self.temp_files

    def cleanup(self):
        print("Cleaning up temp files...")
        for f in self.temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        # ディレクトリも削除（親ディレクトリが TEMP_DIR/uuid なのでその uuid ディレクトリを消す）
        if self.temp_files:
            parent_dir = os.path.dirname(self.temp_files[0])
            if os.path.exists(parent_dir):
                try:
                    shutil.rmtree(parent_dir)
                except:
                    pass
        self.temp_files = []
        self.y = None
        force_garbage_collection()

# --- Whisper Engine ---
def load_whisper(model_name):
    release_memory()
    print(f"Loading Whisper model '{model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.eval() は load_model 内部で設定されるが、念のため
    model = whisper.load_model(model_name, device=device, download_root=CACHE_DIR)
    return model, "whisper"

def transcribe_whisper(model, audio_path):
    print("Running Whisper with chunking...")
    chunker = AudioChunker(audio_path, chunk_sec=30) # Whisper works well with 30s
    chunks = chunker.load_and_split()
    
    full_text = ""
    try:
        for i, chunk_file in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            with torch.inference_mode():
                result = model.transcribe(chunk_file)
                text = result["text"]
                full_text += text + " "
            force_garbage_collection()
    except Exception as e:
        print(f"Whisper Error: {e}")
        raise e
    finally:
        chunker.cleanup()
        
    return full_text.strip()

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
        
        model.eval()
        return model, "nemo"
    except Exception as e:
        print(f"Failed to load NeMo model: {e}")
        raise e

def transcribe_nemo_chunk(model, chunk_path):
    """単一チャンクの推論（再試行ロジック用）"""
    is_canary = False
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'target'):
            if "canary" in str(model.cfg.target):
                is_canary = True
    
    with torch.inference_mode():
        if is_canary:
            predicted_text = model.transcribe(paths2audio_files=[chunk_path])[0]
        elif hasattr(model, 'transcribe'):
            try:
                predicted_text = model.transcribe(paths2audio_files=[chunk_path])[0]
            except TypeError:
                predicted_text = model.transcribe([chunk_path])[0]
        else:
            raise ValueError("Model does not support transcribe method.")
            
    if isinstance(predicted_text, list):
        predicted_text = predicted_text[0]
    
    # Hypothesisオブジェクトなどが返ってくる場合への対応
    if hasattr(predicted_text, 'text'):
        return predicted_text.text
    return str(predicted_text)

def transcribe_nemo(model, audio_path):
    print("Running NeMo with dynamic chunking...")
    # 初期チャンク長
    current_chunk_sec = 30
    
    # 全体を読み込み済みAudioChunkerを使わずに、失敗したら再分割する戦略をとる
    # しかし、AudioChunkerは分割済みファイルをリストで持つ仕様。
    # OOMが出たら、「残りの音声」をさらに細かく刻む...というのは複雑になる。
    # ここでは、「チャンクサイズを決めて分割 -> 失敗したらチャンクサイズを半分にして 最初からやり直し」
    # というシンプルなリトライ戦略をとる（整合性のため）。
    
    full_text = ""
    
    while current_chunk_sec >= 5:
        print(f"Trying chunk_sec = {current_chunk_sec}...")
        chunker = AudioChunker(audio_path, chunk_sec=current_chunk_sec)
        chunks = chunker.load_and_split()
        current_text_accum = ""
        failed = False
        
        try:
            for i, chunk_file in enumerate(chunks):
                print(f"NeMo Processing chunk {i+1}/{len(chunks)} (len={current_chunk_sec}s)...")
                try:
                    text = transcribe_nemo_chunk(model, chunk_file)
                    current_text_accum += text + " "
                    force_garbage_collection()
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"OOM detected at chunk {i+1}!")
                        failed = True
                        break # 内側のループを抜ける
                    else:
                        raise e
        except Exception as e:
            print(f"NeMo Critical Error: {e}")
            chunker.cleanup()
            raise e
        
        chunker.cleanup()
        
        if failed:
            print("Retrying with smaller chunk size...")
            current_chunk_sec = int(current_chunk_sec / 2)
            force_garbage_collection()
            continue
        else:
            # 成功
            return current_text_accum.strip()

    return "Error: CUDA OOM even with minimal chunk size."

# --- Hugging Face Transformers ---
def load_transformers(model_name):
    from transformers import pipeline
    release_memory()
    print(f"Loading Transformers model '{model_name}'...")
    device = 0 if torch.cuda.is_available() else -1
    try:
        # chunk_length_s と stride_length_s は pipeline 実行時に指定するが、
        # ここでパラメータとして持っておくわけではない。
        # torch_dtype=torch.float16 は GPU 利用時に有効
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = pipeline(
            "automatic-speech-recognition", 
            model=model_name, 
            device=device, 
            torch_dtype=dtype
        )
        return pipe, "transformers"
    except Exception as e:
        raise e

def transcribe_transformers(pipe, audio_path):
    print(f"Transcribing with Transformers (Pipeline Chunking): {audio_path}")
    # Pipeline の chunking 機能を利用
    # batch_size=1 でメモリ節約
    with torch.inference_mode():
        result = pipe(
            audio_path, 
            chunk_length_s=30, 
            stride_length_s=5, 
            batch_size=1,
            return_timestamps="word"
        )
    force_garbage_collection()
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
        model.eval()
        return model, "seamless"
    except Exception as e:
        print(f"Failed to load Seamless M4T: {e}")
        raise e

def transcribe_seamless_chunk(model, chunk_path):
    global processor
    # 読み込み時に 16k になっているが、念のため再指定
    audio, sr = librosa.load(chunk_path, sr=16000)
    
    inputs = processor(audios=audio, return_tensors="pt", sampling_rate=16000).to(model.device)
    
    with torch.inference_mode():
        output_tokens = model.generate(**inputs, tgt_lang="jpn", generate_speech=False)
        transcription = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
        
    del inputs, output_tokens
    return transcription

def transcribe_seamless(model, audio_path):
    print("Running Seamless M4T with chunking...")
    chunker = AudioChunker(audio_path, chunk_sec=30)
    chunks = chunker.load_and_split()
    
    full_text = ""
    try:
        for i, chunk_file in enumerate(chunks):
            print(f"Seamless Processing chunk {i+1}/{len(chunks)}...")
            text = transcribe_seamless_chunk(model, chunk_file)
            full_text += text + " "
            force_garbage_collection()
    except Exception as e:
        print(f"Seamless Error: {e}")
        raise e
    finally:
        chunker.cleanup()
        
    return full_text.strip()

# --- Main Logic ---
def process_audio(audio, model_selection):
    global current_model, current_model_name, current_engine, processor
    
    if audio is None:
        return "No audio provided.", ""
    
    # 簡易的なファイルサイズチェック（目安）
    # Gradioは一時ファイルパスを渡す
    file_size_mb = os.path.getsize(audio) / (1024 * 1024)
    print(f"Input Audio Size: {file_size_mb:.2f} MB")
    
    # 30分以上の警告などは、ここで duration をチェックしてもよいが、
    # 処理自体は通す（chunkingがあるため）
    
    if processing_lock.locked():
        print("!!! Another user is currently processing. Waiting for lock... !!!")
        
    with processing_lock:
        print(f"Start processing for: {model_selection}")
        start_time = time.time()
        
        if not model_selection:
            model_selection = "whisper-base"
            
        try:
            # モデルロード
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
            
            # 推論実行
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
            
            force_garbage_collection()
            return result_text, time_info
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            elapsed_time = time.time() - start_time
            force_garbage_collection()
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
            with gr.Row():
                download_btn = gr.Button("Download Text")
                reading_btn = gr.Button("Download Reading (Katakana)")
            download_file = gr.File(label="Download File")
            download_btn.click(save_transcription, inputs=output_text, outputs=download_file)
            reading_btn.click(save_reading, inputs=output_text, outputs=download_file)
    submit_btn.click(fn=process_audio, inputs=[audio_input, model_dropdown], outputs=[output_text, time_output])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(share=True)

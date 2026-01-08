import gradio as gr
import os
import torch
import gc

# 各ライブラリは使用時にインポートするか、ここでまとめてインポート
# インストールされていない場合のハンドリングのため、トップレベルではtry-exceptしても良いが
# 今回はrequirementsでインストール前提とする
import whisper
import librosa
import soundfile as sf
import numpy as np

# キャッシュディレクトリ
CACHE_DIR = "./models"
os.makedirs(CACHE_DIR, exist_ok=True)

# グローバル変数：現在ロードされているモデルとそのエンジンタイプ
current_model = None
current_model_name = ""
current_engine = ""  # "whisper", "nemo", "transformers"
processor = None     # Transformers用

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

# --- NVIDIA NeMo Engine (Parakeet, Canary) ---
def load_nemo(model_name):
    # nemoはimportが重いのでここでimport
    import nemo.collections.asr as nemo_asr
    
    release_memory()
    print(f"Loading NeMo model '{model_name}'...")
    
    # モデル名のマッピング（メニュー表示名 -> NeMo正式名）
    # Canary: nvidia/canary-1b
    # Parakeet CTC: nvidia/parakeet-ctc-1.1b
    # Parakeet RNNT: nvidia/parakeet-rnnt-1.1b
    
    # 自動的にダウンロード・ロードされる
    # EncDecRNNTBPEModel か EncDecCTCModel か判断が必要だが
    # from_pretrained はクラスメソッドなので、汎用的な ASRModel を使うか、トライ＆エラー
    
    try:
        # まずは汎用的にロードを試みる
        if "canary" in model_name:
            model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=model_name)
        elif "rnnt" in model_name:
            model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)
        else:
            # CTC models (Parakeet CTC etc)
            model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)
            
        if torch.cuda.is_available():
            model = model.cuda()
            
        return model, "nemo"
    except Exception as e:
        print(f"Failed to load NeMo model: {e}")
        raise e

def transcribe_nemo(model, audio_path):
    # NeMoはファイルパスのリストを受け取る
    print(f"Transcribing with NeMo: {audio_path}")
    
    try:
        # 1. まず標準的な呼び出し (paths2audio_files) を試す
        # 多くのASRモデル (EncDecCTCModelなど) はこれに対応しているはずだが
        # バージョンによってはキーワード引数を受け付けない場合がある
        
        # CanaryなどのMultiTaskModelかどうか
        is_canary = False
        if hasattr(model, 'cfg') and hasattr(model.cfg, 'target'):
             if "canary" in str(model.cfg.target):
                 is_canary = True

        if is_canary:
            # Canary specific: 
            # 多くのバージョンで model.transcribe(audio=[audio_path]) もしくは paths2audio_files
            # 最新のNeMoでは単純にリストを渡すだけの場合もある
            predicted_text = model.transcribe(paths2audio_files=[audio_path])[0]
            
        elif hasattr(model, 'transcribe'):
            # 一般的なCTC/RNNTモデル
            # キーワード引数 'paths2audio_files' がダメなら、位置引数としてリストを渡してみる
            # または 'audio' 引数の場合もある
            
            try:
                predicted_text = model.transcribe(paths2audio_files=[audio_path])[0]
            except TypeError:
                # キーワード引数がダメだった場合、リストを直接渡してみる (古いAPIや一部のモデル)
                predicted_text = model.transcribe([audio_path])[0]
                
        else:
            return "Error: Model does not support transcribe method."

    except Exception as e:
        print(f"NeMo transcribe error: {e}")
        # 最後の手段：直接 forward などを呼ぶのは複雑すぎるため、エラーを詳細に返す
        return f"NeMo Error: {str(e)}"
        
    # 結果がリストやタプルで返る場合があるので調整
    if isinstance(predicted_text, list):
        return predicted_text[0]
    return predicted_text

# --- Hugging Face Transformers Engine (Wav2Vec2) ---
def load_transformers(model_name):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
    
    release_memory()
    print(f"Loading Transformers model '{model_name}'...")
    
    # 簡易化のため pipeline を使用
    device = 0 if torch.cuda.is_available() else -1
    
    # pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)
    # 日本語モデルなどで tokenizer が必要な場合があるため pipeline が楽
    
    try:
        pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)
        return pipe, "transformers"
    except Exception as e:
        print(f"Failed to load Transformers model: {e}")
        raise e

def transcribe_transformers(pipe, audio_path):
    print(f"Transcribing with Transformers: {audio_path}")
    # pipelineはパスを受け取れるが、サンプリングレートに注意が必要な場合がある
    # pipelineは自動的に読み込んでくれる
    result = pipe(audio_path)
    return result["text"]


# --- Main Logic ---

def process_audio(audio, model_selection):
    global current_model, current_model_name, current_engine
    
    if audio is None:
        return "No audio provided."
    
    if not model_selection:
        model_selection = "whisper-base"

    try:
        # モデルのロードが必要かチェック
        if current_model_name != model_selection:
            
            if model_selection.startswith("whisper-"):
                # "whisper-base" -> "base"
                w_name = model_selection.replace("whisper-", "")
                current_model, current_engine = load_whisper(w_name)
                
            elif model_selection.startswith("nvidia/"):
                current_model, current_engine = load_nemo(model_selection)
                
            elif "wav2vec2" in model_selection:
                current_model, current_engine = load_transformers(model_selection)
                
            current_model_name = model_selection

        # 推論実行
        if current_engine == "whisper":
            return transcribe_whisper(current_model, audio)
        elif current_engine == "nemo":
            return transcribe_nemo(current_model, audio)
        elif current_engine == "transformers":
            return transcribe_transformers(current_model, audio)
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
    # NVIDIA NeMo (Parakeet / Canary)
    "nvidia/parakeet-rnnt-1.1b",   # 高精度, 英語メイン
    "nvidia/parakeet-ctc-1.1b",    # 高速, 英語メイン
    "nvidia/canary-1b",            # 多言語対応 (日本語含む)
    # Transformers (Wav2Vec2)
    "facebook/wav2vec2-large-960h",             # 英語 (Standard)
    "jonatasgrosman/wav2vec2-large-xlsr-53-japanese", # 日本語 (Community)
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
    OpenAI Whisper, NVIDIA Parakeet/Canary, Meta Wav2Vec2 を切り替えて試せます。
    
    - **Whisper**: 高精度、多言語対応。
    - **Parakeet**: NVIDIAの最新モデル。高速・高精度（主に英語）。
    - **Canary**: NVIDIAの多言語対応モデル。
    - **Wav2Vec2**: Metaのモデル（日本語版はコミュニティモデルを使用）。
    
    ※ 初回選択時はモデルのダウンロードに時間がかかります。
    ※ NeMoモデル(Parakeet/Canary)のロードには数分かかる場合があります。
    """
)

if __name__ == "__main__":
    demo.launch(share=True)
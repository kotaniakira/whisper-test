# Universal Speech Recognition Tool

Google Cloud Workbench (JupyterLab / Linux) 用の多機能音声認識 Web アプリです。
Whisper, NVIDIA Parakeet/Canary, Wav2Vec 2.0 を切り替えて試すことができます。

## 1. 準備 (Terminal で実行)

必要なシステムライブラリと Python パッケージをインストールします。
NeMo は依存関係が多いため、インストールに少し時間がかかります。

```bash
# 1. システムライブラリのインストール (音声処理に必須)
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# 2. Python ライブラリのインストール
# NeMo (ASR) を含めてインストールします
pip install -r requirements.txt

# ※ エラーが出る場合: Cython が必要なことがあるため、先に Cython を入れてから再試行してください
# pip install Cython && pip install -r requirements.txt
```

## 2. 起動方法

```bash
python app.py
```

起動後、ターミナルに表示される **`https://xxxx.gradio.live`** という URL をクリックしてブラウザで開いてください。

## 3. 対応モデルについて

画面上のドロップダウンリストからモデルを選択すると、自動的にダウンロード・ロードされます（初回は時間がかかります）。

- **Whisper (tiny ~ large-v3)**: OpenAI の標準モデル。安定して高精度。
- **NVIDIA Parakeet (RNNT/CTC)**: 非常に高速で高精度な NVIDIA のモデル（主に英語）。
- **NVIDIA Canary**: NVIDIA の多言語対応モデル（日本語も可）。
- **Wav2Vec 2.0**: Meta のモデル。ここでは日本語対応版 (`jonatasgrosman/wav2vec2-large-xlsr-53-japanese`) も選択可能です。

## 4. 注意点

- **メモリ消費**: NeMo のモデル（Canary等）は巨大です。GPU メモリ不足で落ちる場合は、インスタンスの GPU を強化するか、Whisper `small` など軽量なモデルに戻してください。
- **再起動**: モデルを切り替える際、前のモデルをメモリから消す処理を入れていますが、完全には消えないことがあります。動作が重くなった場合は `app.py` を再起動してください。

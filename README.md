# Universal Speech Recognition Tool

Google Cloud Workbench (JupyterLab / Linux) 等で動作する、多機能音声認識 Web アプリケーションです。
**OpenAI Whisper** をはじめ、**NVIDIA NeMo**, **Microsoft Phi-4**, **Meta Seamless M4T**, **Wav2Vec 2.0** などの最新モデルを、一つの画面で切り替えて比較・検証することができます。

## 🚀 特徴

- **Web UI (Gradio)**: ブラウザからマイク録音、または音声ファイルをアップロードして手軽に試せます。
- **マルチモデル対応**: プルダウンからモデルを選ぶだけで、自動的にダウンロード・ロードして推論を実行します。
- **GPU 自動利用**: CUDA 対応 GPU があれば自動的に高速化します。
- **メモリ管理**: モデル切り替え時に、前のモデルを GPU メモリから解放する処理を内蔵しています。

## 📦 対応モデル一覧

| カテゴリ | モデル名 (ID) | 特徴 |
| --- | --- | --- |
| **OpenAI** | `whisper-tiny` ~ `large-v3` | デファクトスタンダード。安定して高精度。 |
| **NVIDIA** | `parakeet-rnnt-1.1b` | 非常に高速で高精度（英語に強い）。 |
| **NVIDIA** | `parakeet-ctc-1.1b` | 高速な CTC モデル。 |
| **NVIDIA** | `parakeet-tdt_ctc-0.6b-ja` | **日本語対応** の軽量・高速モデル。 |
| **NVIDIA** | `canary-1b` | 10億パラメータの多言語・マルチタスクモデル。 |
| **Microsoft** | `Phi-4-multimodal-instruct` | 音声も理解できる最新のマルチモーダル LLM。指示に従って動作します。 |
| **Meta** | `seamless-m4t-v2-large` | 翻訳もこなす巨大モデル。ここでは日本語 ASR として動作します。 |
| **Meta** | `seamless-m4t-medium` | 上記の中規模版。 |
| **Meta** | `wav2vec2-large-960h` | 従来型の Transformer ASR（英語）。 |
| **Community**| `wav2vec2-large-xlsr-53-japanese` | 日本語対応 Wav2Vec 2.0 モデル。 |

## 🛠️ インストール手順 (Google Cloud Workbench / Linux)

### 1. システムライブラリの準備
音声処理に必要なライブラリをインストールします。

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
```

### 2. Python ライブラリのインストール
依存関係を一括インストールします。
※ NeMo や Phi-4, Seamless M4T は依存関係が多いため、インストールに数分かかります。

```bash
pip install -r requirements.txt
```

#### エラーが出る場合
Cython が必要な場合があります。その際は以下を実行してから再試行してください。
```bash
pip install Cython && pip install -r requirements.txt
```

## ▶️ 使い方

### Web アプリの起動

```bash
python app.py
```

起動すると、ターミナルに以下のような URL が表示されます。

```text
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxxxxx.gradio.live
```

**`https://xxxxxxxx.gradio.live`** の方をクリックして、ブラウザでアクセスしてください。

### 操作方法
1.  **Audio Input**: マイクで録音するか、ファイルをアップロードします。
2.  **Select Model**: 試したいモデルを選択します。
    *   初回選択時はモデルのダウンロードが行われるため、完了まで時間がかかります。
    *   GPU メモリ（VRAM）が小さい環境では、Large モデルや Phi-4 は動作しない場合があります。
3.  **Submit**: 音声認識が実行され、結果がテキストボックスに表示されます。

## ⚠️ 注意点・トラブルシューティング

- **GPU メモリ不足 (OOM)**: 
  `Phi-4` や `Seamless M4T Large` は 16GB 程度の VRAM を推奨します。T4 GPU (16GB) ではギリギリ動作しますが、他のプロセスが動いていると落ちることがあります。エラーが出たら `whisper-small` など軽量なモデルに切り替えてください。
  
- **モデルの切り替え**: 
  モデルを変更する際、前のモデルをメモリから消去しようとしますが、PyTorch の仕様上完全に消えないことがあります。動作が重くなったりエラーが出たりした場合は、一度 `Ctrl+C` でプログラムを停止し、再起動してください。

- **NeMo の警告**: 
  NVIDIA NeMo の読み込み時に多くの警告が出ることがありますが、基本的には無視して問題ありません。
import argparse
import os
import torch
import whisper

def main():
    parser = argparse.ArgumentParser(description="Whisper Speech Recognition Tool")
    parser.add_argument("audio_file", help="Path to the input audio file (mp3, wav, m4a, etc.)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], help="Model size to use")
    parser.add_argument("--output", default="result.txt", help="Path to save the text output")
    parser.add_argument("--cache-dir", default="./models", help="Directory to cache the downloaded models")
    
    args = parser.parse_args()

    # Create model cache directory if it doesn't exist
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"--- Settings ---")
    print(f"Audio File: {args.audio_file}")
    print(f"Model: {args.model}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Cache Dir: {args.cache_dir}")
    print(f"----------------")

    # Load model
    print(f"Loading model '{args.model}'...")
    try:
        # download_root specifies where to save/load the model
        model = whisper.load_model(args.model, download_root=args.cache_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Transcribe
    print("Transcribing... (This may take time depending on audio length and model size)")
    try:
        result = model.transcribe(args.audio_file)
        
        text = result["text"]
        
        # Output to console
        print("\n--- Result ---")
        print(text)
        print("--------------\n")

        # Save to file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved result to: {args.output}")

    except Exception as e:
        print(f"Error during transcription: {e}")

if __name__ == "__main__":
    main()

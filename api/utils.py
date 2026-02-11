import librosa

def load_audio(file_path: str, target_sr: int = 16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

import torch
import config
import datetime
import inspect
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa.display
import os

segment_samples = config.segment_length * config.sample_rate

def get_audio_segments(audio_path: str) -> list:
    audio, _ = librosa.load(audio_path, sr=config.sample_rate)
    return [audio[i * segment_samples:(i + 1) * segment_samples] for i in range(-(-len(audio) // segment_samples))]


def get_audio_segments_from_ndarray(audio: np.ndarray) -> list:
    return [audio[i * segment_samples:(i + 1) * segment_samples] for i in range(-(-len(audio) // segment_samples))]


def process_audio_segment(segment, device):
    if len(segment) < segment_samples:
        segment = np.pad(segment, (0, segment_samples - len(segment)), "constant")

    mel_spec = librosa.feature.melspectrogram(y=segment, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels)
    audio_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)

    return audio_tensor

def tensor_to_wav(tensor):
    denoised_output = tensor.squeeze().cpu().detach().numpy()
    return librosa.feature.inverse.mel_to_audio(denoised_output, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length)

def merge_and_save_audio_segments(audio_segments, file_path):
    sf.write(file_path, np.concatenate(audio_segments), config.sample_rate)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def debug_print(*args):
    if config.DEBUG_MODE:
        frame = inspect.currentframe().f_back
        debug_message = f"[DEBUG] [{get_timestamp()}] [{Path(frame.f_code.co_filename).name}:{frame.f_lineno}]"
        print(debug_message, *args)

def create_unique_folder(base_folder_name="debug"):
    if config.DEBUG_MODE:
        unique_folder_path = Path(__file__).parent.absolute().parent / base_folder_name / get_timestamp().replace(":", "-")
        unique_folder_path.mkdir(parents=True, exist_ok=True)
        return str(unique_folder_path)

def create_spectrogram(input_data, base_path, title, filename):
    if config.DEBUG_MODE and config.SAVE_SPECTROGRAMS:
        # Определение типа входных данных
        if isinstance(input_data, str):  # Путь к файлу
            y, sr = librosa.load(input_data, sr=config.sample_rate)
        elif isinstance(input_data, np.ndarray):  # Загруженный аудиофрагмент
            y = input_data
            sr = config.sample_rate
        elif isinstance(input_data, torch.Tensor):  # Тензор
            y = input_data.numpy()
            sr = config.sample_rate
        else:
            raise TypeError("Неподдерживаемый тип входных данных для create_spectrogram")

        # Преобразование в мел-спектрограмму
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels)

        # Создание и сохранение спектрограммы
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(Path(base_path) / filename)
        plt.close()

def save_audio_samples(clean_audio_tensor, noisy_audio_tensor, outputs, unique_folder, epoch, segment_index):
    if config.DEBUG_MODE and config.SAVE_INTERMEDIATE_AUDIO:
        for audio_tensor, prefix in zip([clean_audio_tensor, noisy_audio_tensor, outputs], ['clean', 'noisy', 'output']):
            sf.write(f"{unique_folder}/{prefix}_epoch{epoch+1}_segment{segment_index}.wav", tensor_to_wav(audio_tensor), config.sample_rate)


def mix_audio_files(file_1_path, file_2_path):
     # Загрузка первого аудиофайла
    audio1, _ = librosa.load(file_1_path, sr=config.sample_rate)
    # Загрузка второго аудиофайла
    audio2, _ = librosa.load(file_2_path, sr=config.sample_rate)

    # Убедитесь, что оба аудио имеют одинаковую длину
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]

    # Смешивание аудио
    mixed_audio = (audio1 + audio2) / 2
    return mixed_audio


def mix_audios(audio_1, audio_2):
    # Убедитесь, что оба аудио имеют одинаковую длину
    min_len = min(len(audio_1), len(audio_2))
    audio_1 = audio_1[:min_len]
    audio_2 = audio_2[:min_len]

    # Смешивание аудио
    mixed_audio = (audio_1 + audio_2) / 2
    return mixed_audio


def merge_filenames(file1, file2):
    # Извлечение названий файлов без расширений
    name1, ext = os.path.splitext(os.path.basename(file1))
    name2, _ = os.path.splitext(os.path.basename(file2))

    # Соединение названий файлов и добавление расширений
    merged_name = name1 + '_' + name2 + '.' + ext
    return merged_name


def save_noisy_audios(clean_audio_paths: list, noise_paths: list, target_directory: str):
    for clean_audio_path in clean_audio_paths:
        for noise_path in noise_paths:
            noisy_audio = mix_audio_files(clean_audio_path, noise_path)
            sf.write(target_directory + '/' + merge_filenames(clean_audio_path, noise_path), noisy_audio, config.sample_rate)
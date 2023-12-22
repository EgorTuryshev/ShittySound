from math import ceil
from os import name
import pathlib
from re import S
import numpy as np
import soundfile as sf

import librosa
import librosa.display

import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import config
from utils import debug_print, create_unique_folder, save_spectrogram

segment_length = config.segment_length * config.sample_rate # Рассчитанная длина сегмента в сэмплах

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Слой 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Слой 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Слой 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Слой 4
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Слой 5
        self.conv5 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        # Выходной слой
        self.conv6 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        # Применяем свертки и функции активации
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Выходной слой
        x = self.conv6(x)
        return x
    
def save(model):
    torch.save(model, config.model_filename)


def load():
    return torch.load(config.model_filename)

def tensor_to_wav(tensor):
    denoised_output = tensor.squeeze().cpu().detach().numpy()  # Предполагая, что outputs имеет размер [1, 1, config.n_mels, config.time_frames]
    denoised_audio = librosa.feature.inverse.mel_to_audio(denoised_output, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length)
    return denoised_audio


def merge_and_save_audio_segments(audio_segments, file_path):
    merged_audio = np.concatenate(audio_segments)
    sf.write(file_path, merged_audio, config.sample_rate)

def prepare_dataset(clean_audios: list, noisy_audios: list):
    clean_audio_segments = []
    noisy_audio_segments = []
    for i in range(len(clean_audios)):
        clean_audio_segments = clean_audio_segments + get_audio_segments(clean_audios[i])
        noisy_audio_segments = noisy_audio_segments + get_audio_segments(noisy_audios[i])
    
    return clean_audio_segments, noisy_audio_segments

def get_audio_segments(audio_path: str) -> list:
    audio, _ = librosa.load(audio_path, sr=config.sample_rate)

    num_segments = -(-len(audio) // segment_length)  # Округление вверх

    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        audio_segment = audio[start:end]

        if len(audio_segment) < segment_length:
            audio_segment = np.pad(audio_segment, (0, segment_length - len(audio_segment)), "constant")

        segments.append(audio_segment)

    return segments

def train(model, clean_audios: list, noisy_audios: list, criterion, optimizer, device):
    clean_audio_segments, noisy_audio_segments = prepare_dataset(clean_audios, noisy_audios)
    # merge_and_save_audio_segments(clean_audio_segments, "./clean_merged.wav")
    # merge_and_save_audio_segments(noisy_audio_segments, "./noisy_merged.wav")

    unique_folder = create_unique_folder()
    for epoch in range(config.epochs):
        running_loss = 0.0
        # denoised_segments = []

        for i in range(len(clean_audio_segments)):
            mel_spec_clean = librosa.feature.melspectrogram(y=clean_audio_segments[i], sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels)
            mel_spec_noisy = librosa.feature.melspectrogram(y=noisy_audio_segments[i], sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels)
            if epoch == 900:
                save_spectrogram(mel_spec_clean, unique_folder, "Clean Mel Spectrogram", epoch+1, "clean", i)
                save_spectrogram(mel_spec_noisy, unique_folder, "Noisy Mel Spectrogram", epoch+1, "noisy", i)

            clean_audio_tensor = torch.tensor(mel_spec_clean, dtype=torch.float32).unsqueeze(0)
            noisy_audio_tensor = torch.tensor(mel_spec_noisy, dtype=torch.float32).unsqueeze(0)

            clean_target = clean_audio_tensor.unsqueeze(1).to(device)
            noisy_input = noisy_audio_tensor.unsqueeze(1).to(device)
            if config.DEBUG_MODE and epoch == 900:
                audio = tensor_to_wav(clean_target)
                sf.write(f"{unique_folder}/clean_epoch{epoch+1}_segment{i}.wav", audio, config.sample_rate)
            if config.DEBUG_MODE and epoch == 900:
                audio = tensor_to_wav(noisy_input)
                sf.write(f"{unique_folder}/noisy_epoch{epoch+1}_segment{i}.wav", audio, config.sample_rate)    

            optimizer.zero_grad()  # Обнуляем градиенты

            # Передаем шумные входные данные в модель
            outputs = model(noisy_input)
            # denoised_segments.append(tensor_to_wav(outputs))
            if config.DEBUG_MODE and epoch == 900:
                audio = tensor_to_wav(outputs)
                sf.write(f"{unique_folder}/output_epoch{epoch+1}_segment{i}.wav", audio, config.sample_rate)    

            # Вычисляем функцию потерь, сравнивая выход модели с чистым аудио
            loss = criterion(outputs, clean_target)

            # Вычисляем градиенты и делаем шаг оптимизации
            loss.backward()
            optimizer.step()

            # Накапливаем значение функции потерь
            running_loss += loss.item()

        # merge_and_save_audio_segments(denoised_segments, f'./epoch_outputs/denoised_epoch{epoch+1}.wav')
        # Выводим статистику после каждой эпохи
        print(f'Epoch {epoch + 1}/{config.epochs}, Loss: {running_loss}')

def denoise_audio(input_audio_path: str, output_audio_path: str, model, device):
    # Разделение аудио на сегменты
    segments = get_audio_segments(input_audio_path)
    denoised_audio = []

    for i in range(len(segments)):
        audio_segment = segments[i]

        # Если сегмент слишком короткий, дополняем нулями
        if len(audio_segment) < segment_length:
            audio_segment = np.pad(audio_segment, (0, max(segment_length - len(audio_segment), 0)), "constant")

        # Преобразование в MEL-спектрограмму и в тензор
        S = librosa.feature.melspectrogram(y=audio_segment, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels)
        S_DB_tensor = torch.tensor(S, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
        
        # Применение модели
        model.eval()
        with torch.no_grad():
            denoised_output = model(S_DB_tensor)

        audio_output = tensor_to_wav(denoised_output)
        denoised_audio.append(audio_output)

    merge_and_save_audio_segments(denoised_audio, output_audio_path)

def main():
    print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Segment length:", segment_length)

    clean_audios = [config.clean_audio_data]
    noisy_audios = [config.noisy_audio_data]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file = pathlib.Path(config.model_filename)
    if (not file.exists()):
        model = CNN().to(device)

        criterion = nn.HuberLoss()  # Выбираем функцию потерь
        optimizer = Adam(model.parameters(), lr=0.001)  # Выбираем оптимизатор

        train(model, clean_audios, noisy_audios, criterion, optimizer, device)
        save(model)

    model = load()
    denoise_audio(config.noisy_audio_data, './denoised_speech.wav', model, device)

if __name__ == "__main__":
    main()
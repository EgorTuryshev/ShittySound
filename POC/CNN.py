from math import ceil
from os import name
import pathlib
from re import S
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import librosa
import librosa.display

import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# Глобальные переменные
model_filename = "model.pth"

# Гиперпараметры
epochs = 10
segment_length = 48000  # Пример длины сегмента, например 1 секунда при 48 kHz
sample_rate = 48000 # Частота дискретизации
hop_length = 512
n_fft = 2048
n_mels = 128
time_frames = 94 # Этот параметр можно задавать динамически, узнав размер спектрограммы


class DTLN(nn.Module):
    def __init__(self, input_channels):
        super(DTLN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        # Вычислите размер, учитывая сверточные и пулинг слои
        self.flattened_size = 256 * (n_mels // 8) * (time_frames // 8)  # Примерное вычисление, уточните в зависимости от ваших данных

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, n_mels * time_frames)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Преобразование входных данных из матрицы в вектор
        x = x.view(-1, self.flattened_size)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Восстановление размерности до размера спектрограммы
        x = x.view(-1, 1, n_mels, time_frames)

        return x


def save(model):
    torch.save(model, model_filename)


def load():
    return torch.load(model_filename)


def merge_and_save_audio_segments(audio_segments, file_path):
    merged_audio = np.concatenate(audio_segments)
    S = librosa.feature.melspectrogram(y=merged_audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    audio = librosa.feature.inverse.mel_to_audio(S, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    sf.write(file_path, audio, sample_rate)


def get_audio_segments(audio_path):
    audio, _ = librosa.load(audio_path, sr=sample_rate)

    # Разделение аудио на сегменты
    num_segments = int(np.ceil(len(audio) / segment_length))
    segments = []

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        audio_segment = audio[start:end]

        # Если сегмент слишком короткий, дополняем нулями
        if len(audio_segment) < segment_length:
            audio_segment = np.pad(audio_segment, (0, max(segment_length - len(audio_segment), 0)), "constant")

        segments.append(audio_segment)

    return segments


def prepare_dataset(clean_audios, noisy_audios):
    clean_audio_segments = []
    noisy_audio_segments = []
    for i in range(len(clean_audios)):
        clean_audio_segments = clean_audio_segments + get_audio_segments(clean_audios[i])
        noisy_audio_segments = noisy_audio_segments + get_audio_segments(noisy_audios[i])
    
    return clean_audio_segments, noisy_audio_segments


def tensor_to_wav(tensor):
    denoised_output = tensor.squeeze().cpu().detach().numpy()  # Предполагая, что outputs имеет размер [1, 1, n_mels, time_frames]
    denoised_audio = librosa.feature.inverse.mel_to_audio(denoised_output, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    return denoised_audio


def train(model, clean_audios, noisy_audios, criterion, optimizer, device):
    clean_audio_segments, noisy_audio_segments = prepare_dataset(clean_audios, noisy_audios)
    merge_and_save_audio_segments(clean_audio_segments, "./clean_merged.wav")
    merge_and_save_audio_segments(noisy_audio_segments, "./noisy_merged.wav")

    for epoch in range(epochs):
        running_loss = 0.0
        # denoised_segments = []

        for i in range(len(clean_audio_segments)):
            mel_spec_clean = librosa.feature.melspectrogram(y=clean_audio_segments[i], sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            mel_spec_noisy = librosa.feature.melspectrogram(y=noisy_audio_segments[i], sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

            clean_audio_tensor = torch.tensor(mel_spec_clean, dtype=torch.float32).unsqueeze(0)
            noisy_audio_tensor = torch.tensor(mel_spec_noisy, dtype=torch.float32).unsqueeze(0)

            clean_target = clean_audio_tensor.unsqueeze(1).to(device)
            noisy_input = noisy_audio_tensor.unsqueeze(1).to(device)

            optimizer.zero_grad()  # Обнуляем градиенты

            # Передаем шумные входные данные в модель
            outputs = model(noisy_input)
            # denoised_segments.append(tensor_to_wav(outputs))

            # Вычисляем функцию потерь, сравнивая выход модели с чистым аудио
            loss = criterion(outputs, clean_target)

            # Вычисляем градиенты и делаем шаг оптимизации
            loss.backward()
            optimizer.step()

            # Накапливаем значение функции потерь
            running_loss += loss.item()

        # merge_and_save_audio_segments(denoised_segments, f'./epoch_outputs/denoised_epoch{epoch+1}.wav')
        # Выводим статистику после каждой эпохи
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss}')


def denoise_audio(input_audio_path, output_audio_path, model, device):
    # Загрузка аудиофайла
    audio, _ = librosa.load(input_audio_path, sr=sample_rate)

    # Разделение аудио на сегменты
    num_segments = int(np.ceil(len(audio) / segment_length))
    denoised_audio = []

    for i in range(num_segments):
        # Вырезаем сегмент
        start = i * segment_length
        end = start + segment_length
        audio_segment = audio[start:end]

        # Если сегмент слишком короткий, дополняем нулями
        if len(audio_segment) < segment_length:
            audio_segment = np.pad(audio_segment, (0, max(segment_length - len(audio_segment), 0)), "constant")

        # Преобразование в MEL-спектрограмму
        S = librosa.feature.melspectrogram(y=audio_segment, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        # print(S.shape)
        S_DB = librosa.power_to_db(S, ref=np.max)

        # Преобразование в тензор
        S_DB_tensor = torch.tensor(S_DB, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Применение модели
        model.eval()
        with torch.no_grad():
            denoised_output = model(S_DB_tensor)

        # Обратное преобразование
        denoised_output = denoised_output.squeeze().cpu().numpy()

        denoised_output = np.clip(denoised_output, -80, 0)  # Нормализация значений
        denoised_mel = librosa.db_to_power(denoised_output)
        try:
            denoised_segment = librosa.feature.inverse.mel_to_audio(denoised_mel, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        except ValueError as e:
            print("Ошибка при обратном преобразовании:", e)
            denoised_segment = np.zeros_like(audio)  # Заполнение нулями в случае ошибки
        
        denoised_audio.append(denoised_segment)

    # Объединение обработанных сегментов
    denoised_audio = np.concatenate(denoised_audio)

    # Сохранение результата в аудиофайл
    sf.write(output_audio_path, denoised_audio, sample_rate)


def main():
    clean_audios = ['./source_audio/speech.wav']
    noisy_audios = ['./source_audio/noisy_speech.wav']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file = pathlib.Path(model_filename)
    if (not file.exists()):
        model = DTLN(input_channels=1).to(device)

        criterion = nn.MSELoss()  # Выбираем функцию потерь (MSE подходит для задачи регрессии)
        optimizer = Adam(model.parameters(), lr=0.001)  # Выбираем оптимизатор

        train(model, clean_audios, noisy_audios, criterion, optimizer, device)
        save(model)

    model = load()
    denoise_audio('./source_audio/noisy_speech.wav', './denoised_speech.wav', model, device)

main()
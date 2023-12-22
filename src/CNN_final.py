import pathlib
import soundfile as sf
import librosa

import os

import torch
from torch import nn
import torch.nn as nn
from torch.optim import AdamW

import config
import utils

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        # Слой 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)

        # Слой 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)

        # Слой 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01)

        # Слой 4
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.LeakyReLU(negative_slope=0.01)

        # Слой 5
        self.conv5 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.LeakyReLU(negative_slope=0.01)

        # Skip Connection Layer
        self.skip_conv = nn.Conv2d(1, 32, kernel_size=(1, 1))  # для подстройки размерности

        # Выходной слой
        self.conv6 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        # Сохраняем вход для skip connection
        identity = x

        # Применяем свертки и функции активации
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))

        # Skip Connection
        identity = self.skip_conv(identity)
        x += identity  # объединяем с исходными данными

        # Выходной слой
        x = self.conv6(x)
        return x
    
def save(model):
    torch.save(model, config.model_filename)

def load():
    return torch.load(config.model_filename)

def get_segments_from_files(audio_paths: list):
    audio_segments = []
    for audio_path in audio_paths:
        segments = utils.get_audio_segments(audio_path)
        audio_segments.extend(segments)
    return audio_segments

def train(model, criteria, optimizer, device):
    for epoch in range(config.epochs):
        running_loss = 0.0
        k = 0

        for clean_file in os.listdir(config.clean_dir):
            k += 1
            if (k > config.max_clean_files):
                break

            clean_path = os.path.join(config.clean_dir, clean_file)
            clean_audio, _ = librosa.load(clean_path, sr=config.sample_rate)

            for noise_file in os.listdir(config.noise_dir):
                noise_path = os.path.join(config.noise_dir, noise_file)
                noise_audio, _ = librosa.load(noise_path, sr=config.sample_rate)

                min_len = min(len(clean_audio), len(noise_audio))
                clean_audio_cut = clean_audio[:min_len]
                noise_audio = noise_audio[:min_len]

                mixed_audio = (clean_audio_cut + noise_audio) / 2

                clean_audio_segments = utils.get_audio_segments_from_ndarray(clean_audio_cut)
                mixed_audio_segments = utils.get_audio_segments_from_ndarray(mixed_audio)

                segment_pairs = list(zip(clean_audio_segments, mixed_audio_segments))
                for i in range(0, len(segment_pairs), 2):
                    segments_to_process = segment_pairs[i:i + 2]

                    clean_tensors = [utils.process_audio_segment(pair[0], device).squeeze(0) for pair in segments_to_process]
                    mixed_tensors = [utils.process_audio_segment(pair[1], device).squeeze(0) for pair in segments_to_process]

                    clean_batch = torch.stack(clean_tensors)
                    mixed_batch = torch.stack(mixed_tensors)

                    # Обнуляем градиенты
                    optimizer.zero_grad()
                    # Передаем шумные входные данные в модель
                    outputs = model(mixed_batch)
                    # Вычисляем функцию потерь, сравнивая выход модели с чистым аудио
                    loss = criteria(outputs, clean_batch)
                    # Вычисляем градиенты и делаем шаг оптимизации
                    loss.backward()
                    optimizer.step()
                    # Накапливаем значение функции потерь
                    running_loss += loss.item()

        print(f'Epoch: {epoch}, Loss: {running_loss}')

def denoise_audio(input_audio_path: str, output_audio_path: str, model, device):
    # Разделение аудио на сегменты
    segments = utils.get_audio_segments(input_audio_path)
    # print(len(segments))
    denoised_audio = []

    for segment in segments:
        tensor = utils.process_audio_segment(segment, device)
        
        # Применение модели
        model.eval()
        with torch.no_grad():
            denoised_output = model(tensor)

        audio_output = utils.tensor_to_wav(denoised_output)
        denoised_audio.append(audio_output)

    utils.merge_and_save_audio_segments(denoised_audio, output_audio_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file = pathlib.Path(config.model_filename)
    if (not file.exists()):
        model = EnhancedCNN().to(device)

        criteria = nn.HuberLoss()  # Выбираем функцию потерь
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
        
        train(model, criteria, optimizer, device)
        save(model)

    model = load()
    denoise_audio('./noisy_speech.wav', './denoised_speech.wav', model, device)

if __name__ == "__main__":
    main()
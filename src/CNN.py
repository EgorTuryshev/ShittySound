import pathlib
import soundfile as sf

import librosa
import librosa.display

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

def train(model, clean_audios: list, noisy_audios: list, criteria, optimizer, device):
    clean_audio_segments = get_segments_from_files(clean_audios)
    noisy_audio_segments = get_segments_from_files(noisy_audios)
    unique_folder = utils.create_unique_folder()
    
    for epoch in range(config.epochs):
        running_loss = 0.0

        for i, (clean_segment, noisy_segment) in enumerate(zip(clean_audio_segments, noisy_audio_segments)):
            clean_tensor = utils.process_audio_segment(clean_segment, device)
            noisy_tensor = utils.process_audio_segment(noisy_segment, device)
            utils.create_spectrogram(clean_tensor, unique_folder, f'Clean Mel Spectrogram - Epoch {epoch}', f"Clean-{epoch}_segment{i}.png")
            utils.create_spectrogram(noisy_tensor, unique_folder, f'Noisy Mel Spectrogram - Epoch {epoch}', f"Noisy-{epoch}_segment{i}.png")

            # Обнуляем градиенты
            optimizer.zero_grad()
            # Передаем шумные входные данные в модель
            outputs = model(noisy_tensor)
            # Вычисляем функцию потерь, сравнивая выход модели с чистым аудио
            loss = criteria(outputs, clean_tensor)
            # Вычисляем градиенты и делаем шаг оптимизации
            loss.backward()
            optimizer.step()
            # Накапливаем значение функции потерь
            running_loss += loss.item()

            utils.save_audio_samples(clean_tensor, noisy_tensor, outputs, unique_folder, epoch, i)

        print(f'Epoch {epoch + 1}/{config.epochs}, Loss: {running_loss}')

def denoise_audio(input_audio_path: str, output_audio_path: str, model, device):
    # Разделение аудио на сегменты
    segments = utils.get_audio_segments(input_audio_path)
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
    clean_audios = [config.clean_audio_data]
    noisy_audios = [config.noisy_audio_data]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file = pathlib.Path(config.model_filename)
    if (not file.exists()):
        model = EnhancedCNN().to(device)

        criteria = nn.HuberLoss()  # Выбираем функцию потерь
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
        
        train(model, clean_audios, noisy_audios, criteria, optimizer, device)
        save(model)

        model = load()
        denoise_audio(config.noisy_audio_data, './denoised_speech.wav', model, device)

if __name__ == "__main__":
    main()
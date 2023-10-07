# TO-DO:
# Убрать dc offset
# Нормализация
# Обучающая выборка

from math import ceil
import pathlib
from re import S
import torch
import torchaudio
import soundfile as sf
from torch import nn
from torchaudio.transforms import Spectrogram, GriffinLim, InverseSpectrogram

import matplotlib.pyplot as plt

# print(torch.version.cuda)
# print(torch.cuda.is_available())

# Гиперпараметры
sample_rate = 48000 # Частота дискретизации
epochs = 1 # Эпохи
window_len = 2048
window_hop = 1024
split_len = 0.5
n_fft = 2048

# Определение модели DTLN (пока почти заглушка)
class DTLN(nn.Module):
    def __init__(self, input_size):
        super(DTLN, self).__init__()
        # self.block1 = nn.Sequential(
        #     nn.Conv1d(1, 256, kernel_size=3, stride=1, padding=1)
        # )
        self.block2 = nn.Sequential(
            nn.LSTM(input_size, 256, 2, batch_first=True)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid()
        )
        self.final_layer = nn.Conv1d(256, 1, kernel_size=1)

    def forward(self, x):
        print(x.shape)
        batch_size, num_channels, num_frames, num_features = x.shape
        x = x.view(batch_size, num_frames, num_channels * num_features)
        # x = self.block1(x)
        # x = x.transpose(1, 2)
        x,_ = self.block2(x) # Выбираем только выходные данные LSTM
        # x = self.block3(x[:, -1, :])
        # x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.final_layer(x)
        return x


class StftPipeline(torch.nn.Module):
    def __init__(
        self,
        n_fft=n_fft
    ):
        super().__init__()
        self.spec = Spectrogram(n_fft=n_fft, power=None, normalized=True, win_length=window_len, hop_length=window_hop)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Convert to power spectrogram
        spec = self.spec(waveform)

        return spec
    

class IfftPipeline(torch.nn.Module):
    def __init__(
        self,
        n_fft=n_fft
    ):
        super().__init__()
        #self.griffinlim = GriffinLim(n_fft=n_fft, power=power, momentum=0.99, win_length=window_len, hop_length=window_hop, n_iter=64)
        self.inverse = InverseSpectrogram(n_fft=n_fft, win_length=window_len, hop_length=window_hop)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        inverse = self.inverse(waveform)

        return inverse


# Функция для выведения спектра
def plot_waveform(waveform, title=""):
    plt.figure(figsize=(20,4))
    plt.plot(waveform.t().numpy())
    plt.title(title)
    plt.show()


def get_audio_segments(filename, length):
    data, sr = torchaudio.load(filename, normalize=True) # Нормализация под вопросом

    duration = len(data) / sr
    audio_segments = []

    for i in range(ceil(duration / length)):
        start_sample = int(i * length * sr)
        end_sample = int((i * length + length) * sr)
        audio_segments.append(data[start_sample:end_sample])

    return audio_segments


def main():
    signal, _ = torchaudio.load('nine/signal.wav')
    stftPip = StftPipeline()
    stftPip.to(device='cpu', dtype=torch.float32)
    ifftPip = IfftPipeline()
    ifftPip.to(device='cpu', dtype=torch.float32)
    torchaudio.save('stft.wav', ifftPip(stftPip(signal)), sample_rate)
    return

    # Создание модели и перенос на GPU
    device = torch.device('cuda')

    filename = "model.pth"
    file = pathlib.Path(filename)
    if (not file.exists()):
        model = DTLN(419).to(device)
        train(model, device)
        save(model)
    else:
        model = load(filename)
    evaluate(model, device, 'before.wav')


def train(model, device):
    voice_files = ['nine/signal.wav']
    noise_files = ['nine/noise.wav']

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    total_steps = len(voice_files) * epochs
    current_step = 0

    stftPip = StftPipeline()
    stftPip.to(device=device, dtype=torch.float32)

    for epoch in range(epochs):
        for voice_file, noise_file in zip(voice_files, noise_files):
            voice_segments = get_audio_segments(voice_file, split_len)
            noise_segments = get_audio_segments(noise_file, split_len)
            for i in range(len(voice_segments)):
                voice_waveform = voice_segments[i]
                noise_waveform = noise_segments[i]

                # Обрезка до минимальной длины для синхронизации файлов голоса и шума по времени (потом лучше дополнять шум по времени)
                min_length = min(voice_waveform.size(1), noise_waveform.size(1))
                voice_waveform = voice_waveform[:, :min_length]
                noise_waveform = noise_waveform[:, :min_length]

                # Теперь можно безопасно сложить два тензора
                noisy_waveform = voice_waveform + noise_waveform

                # Перенос данных на GPU и добавление размерности канала
                # Преобразование стерео в моно путем усреднения по размерности канала (звук с микрофона - всегда в моно)
                voice_waveform = voice_waveform.mean(dim=0).unsqueeze(0).unsqueeze(0).to(device)
                noisy_waveform = noisy_waveform.mean(dim=0).unsqueeze(0).unsqueeze(0).to(device)

                # Обучение модели
                model.train()
                optimizer.zero_grad()

                output = model(stftPip(noisy_waveform))
                # loss = criterion(output, stftPip(voice_waveform))

                # loss.backward()
                optimizer.step()
                    
            current_step += 1
            # print(f"Потери: {loss}")
            print(f"Прогресс обучения: {current_step / total_steps * 100:.2f}%")

            if (epoch % 2 == 0):
                torch.cuda.empty_cache()
    
    print('Обучение завершено')


def evaluate(model, device, filename):
    model.eval()

    stftPip = StftPipeline()
    stftPip.to(device=device, dtype=torch.float32)
    ifftPip = IfftPipeline()
    ifftPip.to(device=device, dtype=torch.float32)

    input_waveform, _ = torchaudio.load(filename)
    input_waveform = input_waveform.mean(dim=0).unsqueeze(0).unsqueeze(0).to(device)
    output = ifftPip(model(stftPip(input_waveform)))
    # Переводим тензор обратно на CPU и удаляем размерность канала
    output_waveform = output.detach().cpu().squeeze().unsqueeze(0)
    # Сохраняем в файл
    torchaudio.save('after.wav', output_waveform, sample_rate)

    plot_waveform(output_waveform, title="Output Waveform")


def save(model):
    torch.save(model, "model.pth")


def load(filename):
    return torch.load(filename)


main()
import torch
import torch_directml
import torchaudio
from torch import nn

import matplotlib.pyplot as plt

#Гиперпараметры
sample_rate = 48000 #Частота дискретизации
epochs = 50 #Эпохи
batch_size = 5 # Размер пакета (не работает)

# Определение модели DTLN (пока почти заглушка)
class DTLN(nn.Module):
    def __init__(self):
        super(DTLN, self).__init__()
        self.conv1 = nn.Conv1d(1, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.final_layer = nn.Conv1d(512, 1, kernel_size=1)
# Определение движения по слоям модели
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_layer(x)
        return x

#Функция для выведения спектра
def plot_waveform(waveform, title=""):
    plt.figure(figsize=(20,4))
    plt.plot(waveform.t().numpy())
    plt.title(title)
    plt.show()

# Создание модели и перенос на GPU
device = torch_directml.device()
model = DTLN().to(device)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss() # Критерий - средняя квадратическая функция
optimizer = torch.optim.Adam(model.parameters()) # Просто норм оптимизатор для подобных сеток

# Загрузка данных
# Предполагается, что есть два списка файлов: один для чистого голоса, другой для шума
voice_files = ['nine/signal.wav']
noise_files = ['nine/noise.wav']

total_steps = len(voice_files) * epochs
current_step = 0

for epoch in range(epochs):
    for voice_file, noise_file in zip(voice_files, noise_files):
        voice_waveform, _ = torchaudio.load(voice_file)
        noise_waveform, _ = torchaudio.load(noise_file)
        
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

        output = model(noisy_waveform)
        
        loss = criterion(output, voice_waveform)
        
        print(f"Потери: {loss}")

        loss.backward()
        optimizer.step()
        
        print(f"Потери: {loss}")
        current_step += 1
        print(f"Прогресс обучения: {current_step / total_steps * 100:.2f}%")
        
        # Освобождение памяти
        #del voice_waveform
        #del noise_waveform
        #del noisy_waveform
        #if(epoch % 2==0):
        # torch.cuda.empty_cache()

# Переводим тензор обратно на CPU и удаляем размерность канала
output_waveform = output.detach().cpu().squeeze().unsqueeze(0)
noisy_waveform = noisy_waveform.detach().cpu().squeeze().unsqueeze(0)    
# Сохраняем в файл
torchaudio.save('после.wav', output_waveform, sample_rate=48000)
torchaudio.save('до.wav', noisy_waveform, sample_rate=48000)
        
#plot_waveform(output_waveform, title="Output Waveform")    

print('Обучение завершено')

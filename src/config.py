DEBUG_MODE = False
SAVE_INTERMEDIATE_AUDIO = False
SAVE_SPECTROGRAMS = False

model_filename = 'model.pth'
clean_dir = './../data/source_audio/clean'
noise_dir = './../data/source_audio/noise'

max_clean_files = 100
epochs = 60
segment_length = 2 # Длина сегмента в секундах
sample_rate = 48000 # Частота дискретизации
hop_length = 2048
n_fft = 4096
n_mels = 512
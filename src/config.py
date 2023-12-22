DEBUG_MODE = True
SAVE_INTERMEDIATE_AUDIO = False
SAVE_SPECTROGRAMS = False

model_filename = "model.pth"
clean_audio_data = './../data/source_audio/speech.wav'
noisy_audio_data = './../data/source_audio/noisy_speech.wav'

epochs = 100
segment_length = 2 # Длина сегмента в секундах
sample_rate = 48000 # Частота дискретизации
hop_length = 2048
n_fft = 4096
n_mels = 512
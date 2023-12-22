# ShittySound
## Python 3.10.6
### Делает плохой звук хуже
### Не стоит благодарности

Запускать файл CNN.py

Для вывода промежуточных результатов создать папку epoch_outputs в POC и раскомментить следующие строки:
1. denoised_segments = [] 
2. denoised_segments.append(tensor_to_wav(outputs))
3. merge_and_save_audio_segments(denoised_segments, f'./epoch_outputs/denoised_epoch{epoch+1}.wav')
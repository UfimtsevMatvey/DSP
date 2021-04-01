import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io.wavfile

def DTFT(x, M):
    """
    Функция вычисляет значения ДВПФ в M точках на отрезке 
    по оси нормированных частот [-0.5; 0.5]

    Parameters
    ----------
    x : complex
        входная последовательность отсчетов, первый отсчет при k=0
    M : int
        колличество точек на отрезке [-0.5; 0.5]

    Returns
    -------
    float
        значения оси нормированных частот, 
        соотвествующих вычисленным значениям
        спектральной функции  
    complex
        значения спектральной функции      
    """
    return (-0.5+(np.arange(M)/M), np.fft.fftshift(np.fft.fft(x, M)))
    
from IPython.display import Audio



fs, x1 = scipy.io.wavfile.read('flute.wav')
print("fs = ",fs)
print(len(x1))

x = x1
f, t, Zxx = signal.stft(x, fs=fs, window=('blackman'), nperseg=20000, noverlap=None, nfft=2**16)
#f, t, Zxx = signal.stft(x, fs=fs, window=('boxcar'), nperseg=20000, noverlap=None, nfft=2**15)
#amp = 2 * np.sqrt(2)
fig, ax = plt.subplots(figsize=[8, 4])
ax.pcolormesh(t, f,  20*np.log10(np.abs(Zxx)), vmin=0, cmap=plt.get_cmap('inferno'))

#ax.set_yscale('log')
ax.set_ylim((0, 1500))


plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout()
plt.show()
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

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

#Постройте графики для трех окон длины  N=64  из таблицы во временной области ( wi[k] ,  i=1,2,3 ) 
#и в частотной области (ДВПФ в линейном масштабе и в дБ). Для каждого из окон графически определите следующие параметры.
# Окно Прямоугольное
N = 64
w=signal.windows.boxcar(M=N, sym=False)
w

plt.figure(figsize=[8, 3], dpi=120)

plt.subplot(1, 3, 1)
plt.title('Прямоугольное окно')
k=np.arange(N)
plt.stem(k, w, '-', '.')
plt.xlabel('$k$')
plt.ylabel('$w[k]$')

nu, Wn = DTFT(w, M=8*2048)
print(abs(Wn))
Max_ = max(abs(Wn))

plt.subplot(1, 3, 2)
k=np.arange(N)
plt.plot(nu, abs(Wn))
plt.xlabel('$\\nu$')
plt.ylabel('$|W(\\nu)|$')
plt.grid()

plt.subplot(1, 3, 3)
k=np.arange(N)
plt.plot(nu, 20*np.log10(abs(Wn)/max(abs(Wn))))
plt.grid()
plt.ylim(ymin=-80)
plt.xlabel('$\\nu$')
plt.ylabel('$20 \lg \; |W(\\nu)\; / \;W(0)|$, дБ')
plt.tight_layout()
#plt.show()

# Окно Ханна
w=signal.windows.hann(M=N, sym=False)
w

plt.figure(figsize=[8, 3], dpi=120)
plt.subplot(1, 3, 1)
plt.title('Окно Ханна')
k=np.arange(N)
plt.stem(k, w, '-', '.')
plt.xlabel('$k$')
plt.ylabel('$w[k]$')

nu, Wn = DTFT(w, M=8*2048)

plt.subplot(1, 3, 2)
k=np.arange(N)
plt.plot(nu, abs(Wn))
plt.xlabel('$\\nu$')
plt.ylabel('$|W(\\nu)|$')
plt.grid()

plt.subplot(1, 3, 3)
k=np.arange(N)
plt.plot(nu, 20*np.log10(abs(Wn)/max(abs(Wn))))
plt.grid()
plt.ylim(ymin=-120)
plt.xlabel('$\\nu$')
plt.ylabel('$20 \lg \; |W(\\nu)\; / \;W(0)|$, дБ')
plt.tight_layout()
#plt.show()

# Окно Блэкмана
w=signal.windows.blackman(M=N, sym=False)
w

plt.figure(figsize=[8, 3], dpi=120)

plt.subplot(1, 3, 1)
plt.title('Окно Блэкмана')
k=np.arange(N)
plt.stem(k, w, '-', '.')
plt.xlabel('$k$')
plt.ylabel('$w[k]$')

nu, Wn = DTFT(w, M=8*2048)

plt.subplot(1, 3, 2)
k=np.arange(N)
plt.plot(nu, abs(Wn))
plt.xlabel('$\\nu$')
plt.ylabel('$|W(\\nu)|$')
plt.grid()

plt.subplot(1, 3, 3)
k=np.arange(N)
plt.plot(nu, 20*np.log10(abs(Wn)/max(abs(Wn))))
plt.grid()
plt.ylim(ymin=-120)
plt.xlabel('$\\nu$')
plt.ylabel('$20 \lg \; |W(\\nu)\; / \;W(0)|$, дБ')
plt.tight_layout()
#plt.show()

#########################################################
#Произведите спектральный анализ с помощью ДПФ размерности  M=2048  последовательности
N = 64
n0 = 25
k=np.arange(N)
x=np.cos(2*np.pi*k*n0/N)+np.cos(2*np.pi*k*(n0 + 2)/N)

nu, Xn = DTFT(x, M=2048)

plt.figure(figsize=[8, 4], dpi=120)
plt.title('Взвешивание прямоугольным окном')
plt.plot(nu, abs(Xn))
plt.xlabel('$\\nu$')
plt.ylabel('$|X(\\nu)|$')
plt.xlim([-0.5, 0.5])
plt.tight_layout()

nu, Xn = DTFT(x*signal.windows.hann(M=N, sym=False), M=2048)
plt.figure(figsize=[8, 4], dpi=120)
plt.title('Взвешивание окном Ханна')
plt.plot(nu, abs(Xn))
plt.xlabel('$\\nu$')
plt.ylabel('$|X(\\nu)|$')
plt.xlim([-0.5, 0.5])
plt.tight_layout()

nu, Xn = DTFT(x*signal.windows.blackman(M=N, sym=False), M=2048)
plt.figure(figsize=[8, 4], dpi=120)
plt.title('Взвешивание окном Блэкмана')
plt.plot(nu, abs(Xn))
plt.xlabel('$\\nu$')
plt.ylabel('$|X(\\nu)|$')
plt.xlim([-0.5, 0.5])
plt.tight_layout()

plt.show()
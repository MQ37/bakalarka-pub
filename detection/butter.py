import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

colors = ['tab:blue', 'tab:orange', 'tab:green']


# generate signal
x = np.linspace(0, 1, 500)
y = np.sin(2 * np.pi * 5 * x) + 0.5 * np.sin(2 * np.pi * 50 * x)

plt.rcParams["figure.figsize"] = (10, 6)
plt.subplot(4, 2, 1)
plt.plot(x, y, color='tab:red', label='Původní signál')
plt.legend()

# plot filtered signals
for i, order in enumerate([1, 3, 5]):
    plt.subplot(4, 2, 3 + i*2)
    # filter signal with order of filter
    b, a = signal.butter(order, 0.1)
    y_filt = signal.lfilter(b, a, y)

    # plot filtered signal
    color = colors[i]
    plt.plot(x, y_filt, color=color, label=f'Řád filtru = {order}')
    plt.legend()


# plot frequency response
plt.subplot(1, 2, 2)
for order in [1, 3, 5]:
    b, a = signal.butter(order, 100, 'low', analog=True)
    w, h = signal.freqs(b, a)

    plt.semilogx(w, 20 * np.log10(abs(h)), label=f'Řád filtru = {order}')

plt.xlabel('Frekvence (rad / s)')
plt.ylabel('Amplituda (dB)')
plt.margins(0, 0.1)
# cutoff frequency
plt.axvline(100, color='red')
plt.legend()

plt.show()


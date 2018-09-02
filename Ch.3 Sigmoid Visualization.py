import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

z = np.arange(-7, 7, 0.1)

phi_z = sigmoid(z)

plt.plot(z, phi_z)
# Add a vertical line across the axis
plt.axvline(0.0, color='k')
# Add a horizontal span(rectangle) across the axis
# plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
# Add a hotizontal line across the axis
plt.axhline(y=1.0, ls='dotted', color='k')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.axhline(y=0.0, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()
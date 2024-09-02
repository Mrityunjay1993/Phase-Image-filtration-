import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Select video files
baseFileName = []  # Replace with a method to select files, e.g., using tkinter
folder = os.getcwd()
full1 = os.path.join(folder, baseFileName[0])  # Assuming baseFileName is a list of selected files

video = cv2.VideoCapture(full1)
T = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

I = []
f0 = []

for t in range(T):
    ret, frame = video.read()
    if ret:
        f0.append(frame)
        f1 = f0[t]
        f2 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        I.append(f2)

I = np.array(I)

# Uncomment and repeat for additional videos if needed
# video2 = cv2.VideoCapture(full2)
# T2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
# ...

# Calculate average
avg1 = np.mean(I.reshape(-1, I.shape[2]), axis=0)

# Plotting
plt.subplot(2, 3, 3)
plt.plot(avg1)

# FFT and reconstruction
f = np.fft.fft(I, axis=2)
phase = np.abs(f)
reconstruct = np.fft.ifftn(phase, axes=(0, 1, 2))

for m in range(reconstruct.shape[2]):
    reconstruct[:, :, m] = np.abs(reconstruct[:, :, m])

# Uncomment to save video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# VidObj = cv2.VideoWriter('reconstruct.avi', fourcc, 1, (frameWidth, frameHeight))
# for f in range(reconstruct.shape[2]):
#     VidObj.write(cv2.normalize(reconstruct[:, :, f], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
# VidObj.release()

# Example for plotting a specific pixel's intensity over time
a = I[800, 300, :]
plt.plot(a)
plt.show()


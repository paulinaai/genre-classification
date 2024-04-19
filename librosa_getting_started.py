from time import sleep
import librosa
import numpy as np
import matplotlib.pyplot as plt


violin, sample_rate = librosa.load('violin.wav')
fft_violin = np.abs(librosa.stft(violin))
# cq_violin = np.abs(librosa.cqt(violin, sr = sample_rate))

fig, axes = plt.subplots()
img = librosa.display.specshow(
        librosa.amplitude_to_db(
            # ref=np.max means that the brightest color will be assigned to the
            #    max value presented.  Just makes for better color in the 
            #    spectrogram.
            fft_violin, ref=np.max), 
            # cq_violin, ref=np.max),
        # These axis labels are not just text label, they are telling the 
        #    librosa.display function what values to plot.
        sr = sample_rate, x_axis="time", y_axis="fft", ax=axes
        # sr = sample_rate, x_axis="time", y_axis="cqt_hz", ax=axes
    )
fig.colorbar(img, ax=axes, format="%+2.0f dB")
plt.show()

# Represent the spectrogram as a 2D array of floats.  You'd probably want
#    to scale/shift this to work with it.
print(img.get_array())
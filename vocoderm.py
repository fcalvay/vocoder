
from random import randint
# conda install -c skmad simpleaudio
# conda install -c conda-forge pydub
#import ffmpeg
import pydub
from pydub import AudioSegment
#conda install -c anaconda pyaudio
#conda install -c conda-forge python-sounddevice
import matplotlib.pyplot as plt
import numpy as np
import simpleaudio as sa
from scipy.signal import argrelextrema
import sounddevice as sd
import math

from scipy import signal
from scipy.signal import decimate


from math import log10

import os
os.environ["PATH"] += ":/sw/bin/"

audiolist = AudioSegment.from_mp3('./myfile.mp3')
fs=len(audiolist)/1000

# from http://stackoverflow.com/questions/2226853/interpreting-wav-data/2227174#2227174
spf=audiolist.raw_data


samplerate=audiolist.frame_rate


soundi=np.frombuffer(spf,dtype=np.int16)
sound = soundi*1.0

#notes=decimate(note, 10, n=None, ftype='iir', axis=-1, zero_phase=True)
#notes=decimate(notes, 10, n=None, ftype='iir', axis=-1, zero_phase=True)



duration = 0.002  # Note duration ```
channels=37
lowf=50
highf=8000

a=5
b=3
u=0
lu=[]
correct=[]
test=[]
inv=[]

for n in range(1,channels):
    u=np.mod(a*u+b,channels)
    print (u)
    lu.append(u)
    correct.append(n)
inv=correct
    
for n in range(1,channels):
    for m in range(1,channels):
        if(lu[m-1]==n):
            inv[n-1]=m
print(inv)
        
#ab=0
#bb=0
#for at in range(0,channels):
#    for bt in range(0,channels):
#        u=0
#        test=[]
#  
#        for n in range(1,channels):
#            u=np.mod(at*u+bt,channels)
#            test.append(u)
#        if(test==lu):
#            ab=at
#            bb=bt
#        
#print(ab)
#print(bb)







fsi=int(fs/duration)
fsi=int(fsi/30)
finaloutput=np.linspace(0, 1, duration * samplerate, False)

first=0
length=int(duration*samplerate)
t = np.linspace(first*samplerate, duration, length, False)
output = t*0.0

u=0

for nt in range(20*fsi,24*fsi):
    first=nt*length
   # print(nt)
    last=first+length
    soundext=sound[first:last]

    first=last
    t = np.linspace(first*samplerate,first*samplerate+ duration, length, False)
    output = t*0.0



    frequency=lowf

    for n in range(1,channels):
        u=np.mod(a*u+b,channels)
        frequency=lowf+(highf-lowf)/channels*n
        testnoteinput=np.sin(frequency * t*2 * np.pi)
        frequencyoutput=lowf+(highf-lowf)/channels*u
        testnoteoutput=np.sin(frequencyoutput * t*2 * np.pi)
        corr=np.dot(soundext,testnoteinput)/length*duration
        output=output+corr*testnoteoutput
        
        testnoteinput=np.cos(frequency * t*2 * np.pi)
        frequencyoutput=lowf+(highf-lowf)/channels*u
        testnoteoutput=np.cos(frequencyoutput * t*2 * np.pi)
        corr=np.dot(soundext,testnoteinput)/length*duration
        output=output+corr*testnoteoutput
        



    finaloutput=np.concatenate([finaloutput,output])

output=finaloutput
 # Ensure that highest value is in 16-bit range
audio = output * (2**15 - 1) / np.max(np.abs(output))
# Convert to 16-bit data
audio = audio.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(audio, 2, 2, samplerate)
#
## Wait for playback to finish before exiting
play_obj.wait_done()

decodeoutput=np.linspace(0, 1, duration * samplerate, False)

for nt in range(0*fsi,4*fsi):
    first=nt*length
 #   print(nt)
    last=first+length
    soundext=finaloutput[first:last]

    first=last
    t = np.linspace(first*samplerate,first*samplerate+ duration, length, False)
    output = t*0.0



    frequency=lowf

    for n in range(1,channels):
        u=np.mod(a*u+b,channels)
        frequency=lowf+(highf-lowf)/channels*u
        testnoteinput=np.sin(frequency * t*2 * np.pi)
        frequencyoutput=lowf+(highf-lowf)/channels*n
        testnoteoutput=np.sin(frequencyoutput * t*2 * np.pi)
        corr=np.dot(soundext,testnoteinput)/length*duration
        output=output+corr*testnoteoutput
        
        testnoteinput=np.cos(frequency * t*2 * np.pi)
        frequencyoutput=lowf+(highf-lowf)/channels*n
        testnoteoutput=np.cos(frequencyoutput * t*2 * np.pi)
        corr=np.dot(soundext,testnoteinput)/length*duration
        output=output+corr*testnoteoutput
        



    decodeoutput=np.concatenate([decodeoutput,output])

output=decodeoutput
 # Ensure that highest value is in 16-bit range
audio = output * (2**15 - 1) / np.max(np.abs(output))
# Convert to 16-bit data
audio = audio.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(audio, 2, 2, samplerate)
#
## Wait for playback to finish before exiting
play_obj.wait_done()










#play_obj = sa.play_buffer(audio, 1, 2, fs)

# =============================================================================
#sound = AudioSegment.from_mp3('./myfile.mp3')
#play(sound)
##
##
##
#import sounddevice as sd
#from scipy.io.wavfile import write
#
#fs = 44100  # Sample rate
#seconds = 3  # Duration of recording
#
#myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
#sd.wait()  # Wait until recording is finished
#write('output.wav', fs, myrecording)  # Save as WAV file
# # =============================================================================
## #
## #
## # =============================================================================
#
# import pyaudio
# import wave
#
# chunk = 1024  # Record in chunks of 1024 samples
# sample_format = pyaudio.paInt16  # 16 bits per sample
# channels = 1
# fs = 8000  # Record at 44100 samples per second
# seconds = 3
# filename = "output.wav"
#
# p = pyaudio.PyAudio()  # Create an interface to PortAudio
#
# print('Recording')
#
# stream = p.open(format=sample_format,
#                 channels=channels,
#                 rate=fs,
#                 frames_per_buffer=chunk,
#                 input=True)
#
# frames = []  # Initialize array to store frames
#
# # Store data in chunks for 3 seconds
# for i in range(0, int(fs / chunk * seconds)):
#     data = stream.read(chunk)
#     frames.append(data)
#
# # Stop and close the stream
# stream.stop_stream()
# stream.close()
# # Terminate the PortAudio interface
# p.terminate()
#
# print('Finished recording')
#
# # Save the recorded data as a WAV file
# wf = wave.open(filename, 'wb')
# wf.setnchannels(channels)
# wf.setsampwidth(p.get_sample_size(sample_format))
# wf.setframerate(fs)
# wf.writeframes(b''.join(frames))
# wf.close()
#
# =============================================================================
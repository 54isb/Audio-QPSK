#import pyaudio
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pyaudio
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
#Default parameters *****************************************
warnings.filterwarnings('ignore')
Fs = 44100 #SAMPING FREQ. [Hz] fs = 44.1KHz
fc = 440.0 #center frequency [Hz] : Default: 440.0Hz (=A)
Ts = 0.01 #Symbol rate [sec]
DTs = (int)(Fs*Ts) #Number of samples for one symbol
K = 8 #number of sidelobes for pulse shaping [pulses]
alpha = 0.4 #Rolloff factor : Default 0.4
NUM_SYM = 100 #number of symbols
Gain = 100.0 #Volume / Gain for Audio output : Default 25.0
SNR = 15 #Signal-to-Noise power Ratio (SNR)[dB] : Default 15dB
#************************************************************

def rrc_shaping(t): #shaping function
    condlist = [t == 0.0, t == Ts/(4.0*alpha), t == - Ts/(4.0*alpha), (t!=0.0)&(t!=Ts/(4.0*alpha))&(t!=- Ts/(4.0*alpha))]
    funclist = [lambda t: 1.0 - alpha + 4.0 * alpha/np.pi, lambda t: alpha/np.sqrt(2) * ((1+2.0/np.pi) * np.sin(np.pi/(4.0*alpha)) +  (1.0 - 2.0/np.pi) * np.cos(np.pi/(4.0*alpha))), lambda t: alpha/np.sqrt(2) * ((1+2.0/np.pi) * np.sin(np.pi/(4.0*alpha)) +  (1.0 - 2.0/np.pi) * np.cos(np.pi/(4.0*alpha))), lambda t:1.0/(np.pi * t/Ts * (1.0 - np.power(4.0*alpha*t/Ts,2))) * (np.sin(np.pi * (1.0-alpha) * t/Ts) + 4.0*alpha*t/Ts*np.cos(np.pi*(1.0+alpha)*t/Ts))]
    return np.piecewise(t.astype(dtype=np.float64), condlist, funclist)

def mapping(x): #NOTE Gray-labeled QPSK
    condlist = [x ==0, x==1, x==2, x==3]
    funclist = [lambda x: 1.0/np.sqrt(2.0) + 1j * 1.0/np.sqrt(2.0),lambda x: -1.0/np.sqrt(2.0) + 1j*1.0/np.sqrt(2.0),lambda x:1.0/np.sqrt(2.0) - 1j * 1.0/np.sqrt(2.0),lambda x:-1.0/np.sqrt(2.0) - 1j * 1.0/np.sqrt(2.0)]
    return np.piecewise(x.astype(dtype=np.complex), condlist, funclist)

def demapping(x): #NOTE Demapper for Gray-labeled QPSK
    condlist = [(np.real(x)>0.0)&(np.imag(x)>0.0),(np.real(x)<=0.0)&(np.imag(x)>0.0),(np.real(x)>0.0)&(np.imag(x)<=0.0),(np.real(x)<=0.0)&(np.imag(x)<=0.0)]
    funclist = [lambda x: 0,lambda x: 1,lambda x: 2,lambda x: 3]
    return np.piecewise(x.astype(dtype=np.complex), condlist, funclist)

def comparebit(a,b):
    return np.bitwise_and(1, np.bitwise_xor(a,b)) + np.bitwise_and(1, np.right_shift(np.bitwise_xor(a,b),1))

#Open Audio output
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=Fs,
                frames_per_buffer=1024,
                output=True)

#Transmission Main*****************************************

#generating filter response
pulse = rrc_shaping(np.arange(-DTs*K,DTs*K) * 1/Fs)
#Generate Discrete Signals
data = np.random.randint(0,4,(NUM_SYM,1))
#Mapping Data onto QPSK symbols
qpsk_sym = mapping(data)

#generating observation samples in time domain
time_seq = np.reshape(np.zeros(NUM_SYM*DTs),[-1,1]) + 1j * np.reshape(np.zeros(NUM_SYM*DTs),[-1,1])
time_seq[::DTs] = qpsk_sym
#generating pulse-shaped baseband signal
trans_sig = np.convolve(np.real(time_seq).flatten(),pulse) + 1j * np.convolve(np.imag(time_seq).flatten(),pulse)

#generate waveform in RF domain
time_axis = np.arange(np.size(trans_sig)) * 1/Fs
tr_waveform = np.real(trans_sig)*np.cos(2.0*np.pi*fc*time_axis) - np.imag(trans_sig)*np.sin(2.0*np.pi*fc*time_axis)

#Playing Sound of Transmitted Signals
print("Play - Trans")
out_wave = Gain * tr_waveform
stream.write(out_wave.astype(np.float32).tostring())

#transmission over sonic wave
sigma = np.sqrt(1.0/pow(10.0,SNR/10.0) * Ts * Fs);
waveform = tr_waveform + np.random.normal(0.0,sigma,np.size(tr_waveform))
out_wave = waveform

#Playing Sound of Received Signals
print("Play - Receive")
stream.write(out_wave.astype(np.float32).tostring())
stream.close()

#Down Convert: RF -> Baseband
waveform_re = waveform*np.cos(2.0*np.pi*fc*time_axis)
rec_signal_re = np.convolve(waveform_re,pulse)
waveform_im = - waveform*np.sin(2.0*np.pi*fc*time_axis)
rec_signal_im = np.convolve(waveform_im,pulse)

#Demodulation***************************
baseband_rec = rec_signal_re[2*DTs*K:NUM_SYM*DTs+2*DTs*K-2:DTs] + 1j * rec_signal_im[2*DTs*K:NUM_SYM*DTs+2*DTs*K-2:DTs]
ans = demapping(baseband_rec).astype(dtype=np.int)
error = comparebit(ans.reshape(NUM_SYM,1),data).sum()

#Showing results************************
print('Center Frequency: ', fc, '[Hz], Bandwidth: ', 1.0/Ts,'[Hz]')
print('SNR:',SNR,'[dB]')
print('Number of Transmitted Bits:',NUM_SYM*2)
print('Number of Errors:',error)

#Plotting signal waveforms*************************
plt.figure('Signal waveforms', figsize=(15,7))
plt.subplot(2,3,1,xlabel='[sec]')
time_axis = np.arange(np.size(pulse)) * 1/Fs
plt.plot(time_axis,pulse,label="Filter response")
plt.subplot(2,3,2,xlabel='[sec]')
time_axis = np.arange(np.size(time_seq)) * 1/Fs
plt.plot(time_axis,np.real(time_seq),label="Real Part (Modulated)")
plt.plot(time_axis,np.imag(time_seq),label="Imag Part (Modulated)")
plt.legend()
plt.subplot(2,3,3,xlabel='[sec]')
time_axis = np.arange(np.size(trans_sig)) * 1/Fs
plt.plot(time_axis,np.real(trans_sig),label="Real Part (Transmitted)")
plt.plot(time_axis,np.imag(trans_sig),label="Imag Part (Transmitted)")
plt.legend()
plt.subplot(2,3,4,xlabel='[sec]')
time_axis = np.arange(np.size(tr_waveform)) * 1/Fs
plt.plot(time_axis,tr_waveform,label="Received signals (before Noise)")
plt.legend()
plt.subplot(2,3,5,xlabel='[sec]')
time_axis = np.arange(np.size(waveform)) * 1/Fs
plt.plot(time_axis,waveform,label="Received signals (after Noise)")
plt.legend()
plt.subplot(2,3,6,xlabel='[sec]')
time_axis = np.arange(np.size(rec_signal_im)) * 1/Fs
plt.plot(time_axis,rec_signal_re,label="Real part (Received)")
plt.plot(time_axis,rec_signal_im,label="Imag part (Received)")
plt.legend()
plt.show()

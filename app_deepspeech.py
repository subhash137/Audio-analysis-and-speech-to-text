import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import io
from scipy import signal
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load Whisper model

# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

import whisper



def remove_noise(y, sr):
    S = librosa.stft(y)
    noise_spectrum = np.mean(np.abs(S[:, :int(S.shape[1] * 0.1)]), axis=1)
    S_clean = S - noise_spectrum[:, np.newaxis]
    S_clean = np.maximum(S_clean, 0)
    y_clean = librosa.istft(S_clean)
    
    nyquist = 0.5 * sr
    cutoff = 100 / nyquist
    b, a = signal.butter(5, cutoff, btype='highpass', analog=False)
    y_clean = signal.filtfilt(b, a, y_clean)
    st.write(y_clean.shape)
    y_clean = librosa.util.normalize(y_clean)
    y_clean = y_clean.astype(np.float32)
    
    return y_clean
def save_audio(audio, sample_rate):
    # Create a directory to store recordings if it doesn't exist
    # os.makedirs("recordings", exist_ok=True)
    
    # # Generate a filename with current timestamp
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "audio.wav"
    
    # Save the audio file
    sf.write(filename, audio, sample_rate)
    
    return filename

def process_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)

    
    duration = librosa.get_duration(y=y, sr=sr)
    st.write(duration)
    st.write(y.shape)
    channels = 1 if y.ndim == 1 else y.shape[0]
    
    y_clean = remove_noise(y, sr)
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 20))
    
    librosa.display.waveshow(y, sr=sr, ax=axs[0])
    axs[0].set_title('Original Waveform')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    
    librosa.display.waveshow(y_clean, sr=sr, ax=axs[1])
    axs[1].set_title('Cleaned Waveform')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amplitude')
    
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axs[2])
    fig.colorbar(img, ax=axs[2], format='%+2.0f dB')
    axs[2].set_title('Spectrum after STFT')
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axs[3])
    fig.colorbar(img, ax=axs[3], format='%+2.0f dB')
    axs[3].set_title('Mel Spectrogram')
    
    plt.tight_layout()
    return fig, duration, channels, sr, y_clean

def record_audio(duration=5, sample_rate=16000):
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def transcribe_audio(audio, sample_rate):
    model = whisper.load_model("base")
    result = model.transcribe(audio)
    return result["text"]
    # input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
    # predicted_ids = model.generate(input_features)
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    # return transcription

st.title('Audio Analysis and Speech-to-Text')

tab1, tab2 = st.tabs(["Real-time Speech-to-Text", "Audio File Analysis"])

with tab1:
    st.header("Real-time Speech-to-Text")
    if st.button("Start Recording"):
        st.write("Recording...")
        audio = record_audio()
        st.write("Recording complete. Processing...")
        saved_file = save_audio(audio, 16000)
        st.write(f"Audio saved as: {saved_file}")
        
        # Play the recorded audio
        st.audio(saved_file, format='audio/wav')
        
        transcription = transcribe_audio(audio, 16000)
        
        st.write("Transcription:", transcription)

    st.write("Click 'Start Recording' to begin.")

with tab2:
    st.header("Audio File Analysis")
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        fig, duration, channels, sr, y_clean = process_audio(uploaded_file)
        
        st.pyplot(fig)
        
        st.write(f"Audio Properties:")
        st.write(f"- Length: {duration:.2f} seconds")
        st.write(f"- Channels: {channels}")
        st.write(f"- Sampling Rate: {sr} Hz")
        
        cleaned_audio_buffer = io.BytesIO()
        sf.write(cleaned_audio_buffer, y_clean, sr, format='wav')
        cleaned_audio_buffer.seek(0)
        
        st.write("Cleaned Audio:")
        st.audio(cleaned_audio_buffer, format='audio/wav')
        
        transcribed_text = transcribe_audio(y_clean, sr)
        
        st.write("Transcribed Text:")
        st.write(transcribed_text)
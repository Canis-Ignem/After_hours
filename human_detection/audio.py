import pyaudio
import wave
 
def alarm(sound_file):
    
    # Defines a chunk size of 1024 samples per data frame.
    chunk = 1024 
    
    # Open sound file  in read binary form.
    file = wave.open(sound_file, 'rb')
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Creates a Stream to which the wav file is written to.
    # Setting output to "True" makes the sound be "played" rather than recorded
    stream = p.open(format = p.get_format_from_width(file.getsampwidth()),
                    channels = file.getnchannels(),
                    rate = file.getframerate(),
                    output = True)
    
    # Read data in chunks
    data = file.readframes(chunk)
    
    # Play the sound by writing the audio data to the stream
    while data != '':
        stream.write(data)
        data = file.readframes(chunk)
    
    # Stop, Close and terminate the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
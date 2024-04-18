import os
import time
import wave
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment


def find_wav_files(directory, include_metadata=False):
    all_paths = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".wav"):
                file_path = os.path.join(root, filename)
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    all_paths.append(file_path)
    return all_paths

def signal_csv_n_create_folder(base,wav_file_path):
    target_dirs = []
    for wav in wav_file_path:
        # print(wav)
        # Get the directory and file name without extension
        directory,filename = os.path.split( wav )
        file_stem = os.path.splitext( filename )[0]
        # print(directory,filename,file_stem)
        base = os.path.normpath(base)  # Normalize the base path
        target_directory = os.path.join( "Signals_csv",base,file_stem )
        target_dirs.append(target_directory)
        # # Create target directory if it does not exist
        os.makedirs( target_directory,exist_ok=True )
    return target_dirs

def split_wav_n_create_folder(wav_file_path):
    # Get the directory and file name without extension
    directory,filename = os.path.split( wav_file_path )
    file_stem = os.path.splitext( filename )[0]
    target_directory = os.path.join( "Splitted_wavs",file_stem )

    # Create target directory if it does not exist
    os.makedirs( target_directory,exist_ok=True )

    # Load the WAV file
    audio = AudioSegment.from_wav( wav_file_path )

    # Split the audio into 1-second chunks (1000 milliseconds)
    chunk_length_ms = 1000
    chunks = [audio[i:i+chunk_length_ms] for i in range( 0,len( audio ),chunk_length_ms )]

    # Save each chunk as a new file
    chucked = []
    chucked_names = []
    for i,chunk in enumerate( chunks ):
        chunk_name = f"{file_stem}_chunk{i+1}.wav"
        chucked_names.append(chunk_name)
        chunk_path = os.path.join( target_directory,chunk_name )
        chunk.export( chunk_path,format="wav" )
        chucked.append( chunk_path )
    return chucked,chucked_names


def RealTime_generate_mfcc_image(frame_duration,wav_path,png_path,dir_signal):
    # Load audio file
    sr = 16000

    audio,_ = librosa.load( wav_path,sr=sr )
    frame_length = int( frame_duration * sr )
    normalized_audio = np.array( [] )

    #Normalization
    for i in range( 0,len( audio ),frame_length ):
        frame = audio[i:i+frame_length]
        max_amp = np.max( np.abs( frame ) )
        normalized_frame = frame / max_amp if max_amp > 0 else frame
        normalized_audio = np.concatenate( (normalized_audio,normalized_frame) )

    Real_time_save(normalized_audio,dir_signal)

    mfcc= librosa.feature.mfcc( y=normalized_audio,n_mfcc=40,fmax=8000,sr=sr )
    # delta_mfcc = librosa.feature.delta( mfcc_ )
    # delta2_mfcc = librosa.feature.delta( mfcc_,order=2 )
    # mfcc = np.concatenate( (mfcc_,delta_mfcc,delta2_mfcc),axis=0 )

    # Plotting
    plt.figure( figsize=(10,4) )
    librosa.display.specshow( mfcc,vmax=220,vmin=-150 )
    plt.colorbar( )
    plt.title( f'MFCC of {os.path.basename( wav_path )}' )
    plt.tight_layout( )
    plt.savefig( png_path )
    plt.close( )
def Real_time_save(normalized_audio,dir_signal):
    real_time_dir = os.path.join(dir_signal, "Real_time")
    os.makedirs(real_time_dir, exist_ok=True)  # This creates the directory if it does not exist

    # Define the file path for the CSV file where the data will be saved
    file_path = os.path.join(real_time_dir, "normalized_audio.csv")

    # Save the normalized_audio to a CSV file with comma-separated values, horizontally
    np.savetxt(file_path, [normalized_audio], delimiter=',', fmt='%f')  # fmt='%f' for floating point format

    # Optional: Print the path of the saved file for confirmation
    print(f"Real-time audio data saved at: {file_path}")

def Training_generate_mfcc_image(wav_path,png_path,dir_signal):
    def load_audio_file(file_path):
        # Load an audio file as a floating point time series.
        audio,sr = librosa.load( file_path,sr=16000 )
        return audio,sr
    def frequency_warp(audio,sr,alpha):
        # Apply the resampling to warp the frequencies of the audio signal.
        return librosa.resample( audio,orig_sr=sr,target_sr=int( sr * alpha ) )
    def peak_normalize(audio):
        # Normalize the audio signal to the peak amplitude within the frame.
        max_amplitude = np.max( np.abs( audio ) )
        return audio / max_amplitude if max_amplitude > 0 else audio
    def extract_mfcc_features(audio,sr,n_mfcc=40,fmax=None):
        # Make sure n_fft is not larger than the length of the audio
        n_fft = min( 2048,len( audio ) )

        # If the audio is shorter than expected, pad it with zeros
        if len( audio ) < n_fft:
            padding = n_fft-len( audio )
            audio = np.pad( audio,(0,padding),'constant' )

        # Ensure fmax is set to the Nyquist frequency if not provided
        if fmax is None:
            fmax = sr // 2

        # Compute MFCC features from the audio signal
        mfccs = librosa.feature.mfcc( y=audio,sr=sr,n_mfcc=n_mfcc,n_fft=n_fft,fmax=fmax )
        return mfccs

    original_audio,sr = load_audio_file( wav_path )
    # Augment and process the audio file
    alpha_values = np.arange( 0.8,1.21,0.02 )

    last_mfcc_features,last_alpha,last_frame_index = (None,) * 3
    normalized_frame = None
    for alpha in alpha_values:
        # Frequency Warping
        warped_audio = frequency_warp(original_audio, sr, alpha)
        # Calculate frame length in samples
        frame_length_samples = int(0.2 * sr)
        # Iterate over frames
        for frame_index in range(0, len(warped_audio), frame_length_samples):
            frame = warped_audio[frame_index:frame_index+frame_length_samples]

            # If the last frame is shorter than the frame length, pad it
            if len(frame) < frame_length_samples:
                frame = np.pad(frame, (0, frame_length_samples - len(frame)), 'constant')

            # Normalize the current frame
            normalized_frame = peak_normalize(frame)

            # Extract MFCC features from the normalized frame
            mfcc_features = extract_mfcc_features(normalized_frame, sr)

            # Store the last features
            last_mfcc_features = mfcc_features ; last_alpha = alpha ; last_frame_index = frame_index
            # Plotting

    Training_save(normalized_frame,dir_signal)

    plt.figure( figsize=(10,4) )
    librosa.display.specshow( last_mfcc_features,vmax=220,vmin=-150 )
    plt.colorbar( )
    plt.title( f'MFCC of {os.path.basename( wav_path )}' )
    plt.tight_layout( )
    plt.savefig( png_path )
    plt.close( )
def Training_save(normalized_frame,dir_signal):
    training_dir = os.path.join(dir_signal, "Training")
    os.makedirs(training_dir, exist_ok=True)  # This creates the directory if it does not exist

    # Define the file path for the text file where the frame will be saved
    file_path = os.path.join(training_dir, "normalized_frame.txt")

    # Save the normalized_frame to a text file with space-separated values
    np.savetxt(file_path, normalized_frame, delimiter=' ')

    # Optional: Print the path of the saved file for confirmation
    print(f"Training frame saved at: {file_path}")


if __name__ == '__main__':
    directory = '.'
    # 1. find wavs
    wavs_directories = find_wav_files( directory )
    print( wavs_directories )
    # 2 Split them (in duration 1sec) and create new folder for them
    splitted = []
    signals_directory_to_save=[]
    for wavv in wavs_directories:
        now,names= split_wav_n_create_folder( wavv )
        splitted.append(now)
        signals_directory_to_save.append(signal_csv_n_create_folder(wavv,names))
    print(signals_directory_to_save)
    print( splitted )

    # 3 Generate MFCC images for each split WAV
    for wav_paths, signal_dir in zip(splitted, signals_directory_to_save):
        for path_wav,dir_signal in zip(wav_paths,signal_dir):
            print(dir_signal)
            print(path_wav)
            png_filename = os.path.splitext( os.path.basename( path_wav ) )[0]+'.png'

            png_directory_mine = os.path.join( 'mfcc_Real_Time',os.path.dirname( path_wav ) )
            png_directory_article = os.path.join( 'mfcc_Training',os.path.dirname( path_wav ) )

            png_path_mine = os.path.join( png_directory_mine,png_filename )
            png_path_article = os.path.join( png_directory_article,png_filename )

            # Ensure the directory exists
            os.makedirs( png_directory_mine,exist_ok=True )
            os.makedirs( png_directory_article,exist_ok=True )

            # Generate and save the MFCC image
            # A. mine
            frame_duration = 0.02
            RealTime_generate_mfcc_image( frame_duration , path_wav,png_path_mine ,dir_signal)
            print( f"Generated MFCC image at {png_path_mine}" )
            #B . Article Chat
            """Training_generate_mfcc_image( path_wav,png_path_article,dir_signal )

            print( f"Generated MFCC image at {png_path_article}" )"""

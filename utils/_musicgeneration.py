import numpy as np
from scipy.signal import resample, butter, lfilter, savgol_filter
import librosa
import numpy as np
from scipy.signal import lfilter
from scipy import interpolate

def create_musical_waveform(latent_tensor, target_length, sample_rate):
    """
    Creates a musical waveform from latent representations.
    """
    # Analyze and prepare latent vector dimensions
    batch_size, quantizers, code_len = latent_tensor.shape
    latent_avg = latent_tensor[0].cpu().detach().numpy()  # Use the first example

    # Create an initial silent audio signal
    audio = np.zeros(target_length)
    total_dur = target_length / sample_rate  # Total duration in seconds
    t = np.linspace(0, total_dur, target_length)  # Time vector

    # Base frequency and harmonic settings
    base_freq = 220  # Approximately A3 in Hz
    harmonics = [1, 2, 3, 4, 5, 6]  # Fundamental plus 5 harmonics

    # For each quantizer, add a distinct sound component
    for q in range(quantizers):
        # Resize and normalize latent codes
        latent_q = latent_avg[q]
        latent_q = (latent_q - np.mean(latent_q)) / (np.std(latent_q) + 1e-6)
        latent_q = savgol_filter(latent_q, 51, 3)  # Apply smoothing filter

        # Set amplitude factor for frequency modulation
        amp = 0.8 / quantizers

        # Resample latent code for time-varying frequency modulation
        mod_signal = np.interp(
            np.linspace(0, len(latent_q) - 1, target_length),
            np.arange(len(latent_q)),
            latent_q
        )

        # Frequency modulation based on the harmonic structure
        freq = base_freq * ((q % len(harmonics)) + 1) * harmonics[q % len(harmonics)]
        mod_depth = 20 + 10 * (q + 1)  # Modulation depth

        # Add a sinusoidal carrier wave with frequency modulation
        carrier = np.sin(2 * np.pi * freq * t + mod_depth * np.cumsum(mod_signal) / sample_rate)
        audio += amp * carrier * np.exp(-0.5 * t)  # Applying an envelope for decaying amplitude

    # Apply a low-pass filter (allow frequencies below 1500 Hz)
    b, a = butter(4, 1500 / (sample_rate / 2), 'low')
    audio = lfilter(b, a, audio)

    # Normalize the final audio signal
    audio = 0.95 * audio / np.max(np.abs(audio))
    return audio

def create_melodic_music(latent_tensor, target_length, sample_rate, valence, arousal):
    """
    Advanced function to create melodic music using latent representations.
    
    Emotional settings:
      - Happy: High valence, High arousal -> Fast tempo, major chord structure
      - Sad: Low valence, Low arousal -> Slow tempo, minor chord structure
      - Calm: High valence, Low arousal -> Medium tempo, soft major tonality
      - Angry: Low valence, High arousal -> Fast, noisy, dissonant sound
    """
    # Analyze latent dimensions
    batch_size, quantizers, code_len = latent_tensor.shape
    latent_avg = latent_tensor[0].cpu().detach().numpy()  # Take the first example

    # Create an empty audio signal
    audio = np.zeros(target_length)
    total_dur = target_length / sample_rate  # Total duration in seconds
    t = np.linspace(0, total_dur, target_length)  # Time vector

    # Define musical parameters based on emotional inputs
    tempo_factor = 0.5 + arousal * 1.5       # Higher arousal results in faster tempo
    harmonicity = 0.3 + valence * 0.7        # Higher valence yields a more harmonic sound
    major_factor = valence                   # Higher valence favors major tonality
    noise_factor = (1.0 - valence) * arousal * 0.3  # More noise for anger

    # Define major and minor chord intervals
    major_notes = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]  # Major intervals
    minor_notes = [1.0, 1.2, 1.4, 1.8, 2.25, 2.7]   # Minor intervals

    # Blend chords based on the valence value
    notes = [m * major_factor + n * (1 - major_factor) for m, n in zip(major_notes, minor_notes)]

    # Define a base frequency around which notes are generated
    base_freq = 110 + valence * 110  # Lower valence results in lower tonality

    # Create a musical layer for each quantizer
    for q in range(quantizers):
        # Normalize and smooth latent codes
        latent_q = latent_avg[q]
        latent_q = (latent_q - np.mean(latent_q)) / (np.std(latent_q) + 1e-6)
        latent_q = savgol_filter(latent_q, 51, 3)  # Apply smoothing

        # Interpolate latent codes over time
        mod_signal = np.interp(
            np.linspace(0, len(latent_q) - 1, target_length),
            np.arange(len(latent_q)),
            latent_q
        )

        # Set modulation depth based on arousal (higher arousal = deeper modulation)
        mod_depth = 10 + arousal * 40

        # Choose a different note for each quantizer
        note_idx = q % len(notes)
        freq_multiplier = notes[note_idx]
        freq = base_freq * freq_multiplier

        # Create a sine carrier wave with frequency modulation
        carrier = np.sin(2 * np.pi * freq * t + mod_depth * np.cumsum(mod_signal) / sample_rate)

        # Set amplitude; reduce amplitude for higher frequencies for a natural sound
        amp = (0.8 / quantizers) * (1.0 / (1.0 + 0.2 * note_idx))

        # Create an envelope based on arousal (attack and decay phases)
        attack = 0.1 * (1.0 - arousal * 0.7)  # Lower arousal yields a slower attack
        decay = 0.2 * (1.0 - arousal * 0.3)   # Lower arousal yields a slower decay

        envelope = np.ones_like(t)
        attack_samples = int(attack * sample_rate)
        decay_samples = int(decay * sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if decay_samples > 0:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

        # Mix the carrier with its envelope into the audio signal
        audio += amp * carrier * envelope

        # Add noise if conditions indicate (more noise for lower valence and higher arousal)
        if q % 3 == 0 and noise_factor > 0.1:
            noise = np.random.randn(target_length) * noise_factor * amp * 0.2
            audio += noise * envelope

    # Apply a low-pass filter based on valence (higher valence = brighter sound)
    cutoff_freq = 800 + valence * 3000
    b, a = butter(4, cutoff_freq / (sample_rate / 2), 'low')
    audio = lfilter(b, a, audio)

    # Impose a rhythmic structure by emphasizing beats (dependent on tempo)
    rhythm_length = int(sample_rate / tempo_factor)  # Beat spacing based on tempo factor
    beats = np.zeros_like(audio)
    for i in range(0, target_length, rhythm_length):
        if i + 500 < target_length:
            beats[i:i+500] = np.linspace(1.2, 0.8, 500)

    # Modulate the audio with the rhythm
    audio = audio * (0.8 + 0.2 * beats)

    # Final normalization of the output audio
    audio = 0.95 * audio / np.max(np.abs(audio))
    return audio

def create_piano_melody(latent_tensor, target_length, sample_rate, valence, arousal):
    """
    Function to synthesize emotional piano-like music.
    """
    # Prepare latent vectors
    batch_size, quantizers, code_len = latent_tensor.shape
    latent_avg = latent_tensor[0].cpu().detach().numpy()
    
    # Create an empty audio signal
    audio = np.zeros(target_length)
    total_dur = target_length / sample_rate
    t = np.linspace(0, total_dur, target_length)
    
    # Musical settings based on emotional parameters
    tempo_bpm = 40 + int(arousal * 160)  # BPM between 40 and 200
    beat_duration = 60 / tempo_bpm         # Duration of one beat (in seconds)
    
    # Major-minor tonality ratio (0: full minor, 1: full major)
    major_scale_ratio = 0.2 + valence * 0.8
    
    # Major and minor scales (for Western music)
    major_scale = [0, 2, 4, 5, 7, 9, 11, 12]   # Intervals of C major scale
    minor_scale = [0, 2, 3, 5, 7, 8, 10, 12]    # Intervals of A minor scale
    
    # Decide the scale based on the valence value
    if valence > 0.6:
        scale = major_scale  # Happy/calm -> Major
    elif valence < 0.4:
        scale = minor_scale  # Sad/angry -> Minor
    else:
        # Mix major and minor scales
        scale = [m * major_scale_ratio + n * (1 - major_scale_ratio) 
                 for m, n in zip(major_scale, minor_scale)]
        scale = [round(s) for s in scale]
    
    # Note release factor for legato (long notes)
    note_release = 0.95
    
    # Base note frequency (A4 = 440Hz, adjusted by valence)
    base_freq = 220 * (1.0 + 0.2 * valence)
    
    # Piano note synthesis function (softened version)
    def piano_note(freq, duration, amplitude=0.5, decay_factor=5.0):
        """
        Produces a piano-like note with a softened timbre.
        """
        n_samples = int(duration * sample_rate)
        note_t = np.linspace(0, duration, n_samples)
        
        # Fundamental frequency and harmonics with a softer balance
        harmonics = [1.0, 0.7, 0.4, 0.25, 0.15, 0.1, 0.07]
        wave = np.zeros_like(note_t)
        
        for i, h in enumerate(harmonics):
            h_freq = freq * (i + 1)
            # Ensure the harmonic is below the Nyquist limit
            if h_freq < sample_rate / 2:
                wave += h * np.sin(2 * np.pi * h_freq * note_t)
        
        # ADSR envelope - softer attack and longer release for gentle music
        attack_time = 0.005   # 5ms attack for a soft start
        decay_time = 0.02     # 20ms decay
        sustain_level = 0.8   # Higher sustain
        release_time = 0.5    # 500ms release for a smooth fade-out
        
        # Natural piano envelope (using an exponential decay for a slow drop)
        decay_factor = max(2.0, decay_factor)  # Ensure a slower decay
        envelope = np.exp(-decay_factor * note_t)
        
        # Attack phase: ramp up the envelope
        attack_samples = int(attack_time * sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Release phase: ramp down the envelope at the tail end
        release_start = int((duration - release_time) * sample_rate)
        if release_start > 0 and release_start < n_samples:
            release_env = np.linspace(envelope[release_start], 0, n_samples - release_start)
            envelope[release_start:] = release_env
        
        return amplitude * envelope * wave
    
    # Convert latent codes to a melody
    # Each quantizer will behave like an instrumental voice
    melody_length = int(total_dur / beat_duration) + 1
    
    # Use a higher smoothing factor for more consistent note selection
    smoothing_factor = 100
    
    # Create a rhythmic melody pattern – fewer layers for a calm sound
    for q in range(min(2, quantizers)):
        # Smooth and normalize latent codes
        latent_q = latent_avg[q].copy()
        latent_smooth = np.convolve(latent_q, np.ones(smoothing_factor)/smoothing_factor, mode='same')
        latent_norm = (latent_smooth - np.mean(latent_smooth)) / (np.std(latent_smooth) + 1e-10)
        
        # Map latent values to musical note indices (using a reduced dynamic range)
        note_indices = np.interp(latent_norm, (-1.5, 1.5), (0, len(scale)-1))
        note_indices = np.clip(note_indices, 0, len(scale)-1)
        
        # Apply an octave shift for different layers (first layer is mid-octave)
        octave_shift = q - 1
        
        # Interpolate note indices over the melody length (using quadratic interpolation for smoothness)
        f_interp = interpolate.interp1d(np.linspace(0, melody_length, len(note_indices)), 
                                        note_indices, kind='quadratic')
        melody_points = np.arange(melody_length)
        melody_curve = f_interp(melody_points)
        
        # Note density: lower density for calmer music
        note_density = 0.3 + arousal * 0.4
        
        # Track previous note positions to impose a consistent rhythm
        last_note_pos = -4  # No note at the beginning
        play_pattern = np.zeros(melody_length, dtype=bool)
        
        # Determine whether to play a note at each potential beat position
        for i in range(melody_length):
            if i - last_note_pos >= 2:  # At least 2 beats apart
                if i % 2 == 0:  # More likely to play on even beats
                    if np.random.random() < (0.6 + 0.2 * arousal):
                        play_pattern[i] = True
                        last_note_pos = i
                elif i % 4 == 1:  # Less likely on off-beats
                    if np.random.random() < (0.3 + 0.1 * arousal):
                        play_pattern[i] = True
                        last_note_pos = i
        
        # For each potential note position, synthesize and add the note to the audio signal
        for i in range(melody_length):
            if play_pattern[i]:
                # Convert the note index to a scale degree
                note_idx = int(round(melody_curve[i]))
                semitones = scale[note_idx % len(scale)]
                octave = octave_shift + int(note_idx / len(scale))
                
                # Convert MIDI note (A4=69, 440Hz) to frequency
                midi_note = 12 * (octave + 4) + semitones
                freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
                
                # Note duration – a fraction of a beat determined by the release factor
                note_time = beat_duration * note_release
                amplitude = 0.3 + 0.1 * np.random.random()
                decay = 3.0  # Decay factor for the note envelope
                
                note = piano_note(freq, note_time, amplitude, decay)
                
                # Determine the note's start time in the audio signal
                start_time = i * beat_duration
                start_sample = int(start_time * sample_rate)
                end_sample = min(start_sample + len(note), len(audio))
                audio_slice = audio[start_sample:end_sample]
                note_slice = note[:len(audio_slice)]
                
                # Mix the note into the main audio signal
                audio[start_sample:end_sample] = audio_slice + note_slice
    
    # Add extra reverb for a calm atmosphere
    def rich_reverb(audio_signal, decay=0.7, delays=[int(sample_rate*0.1), int(sample_rate*0.2)]):
        """
        Adds a richer reverb effect.
        """
        reverb_audio = audio_signal.copy()
        for delay in delays:
            for i in range(delay, len(audio_signal)):
                reverb_audio[i] += decay * (0.6 * audio_signal[i-delay])
        return reverb_audio
    
    # Enhance resonance by applying multiple reverb delays
    audio = rich_reverb(audio, 0.6, [int(sample_rate*0.1), int(sample_rate*0.2), int(sample_rate*0.3)])
    
    # Final normalization to prevent clipping
    audio = 0.9 * audio / (np.max(np.abs(audio)) + 1e-8)
    
    return audio
import subprocess
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import os
from datetime import datetime
import numpy as np
import noisereduce as nr
from audio_separator.separator import Separator
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline

class SpeechProcessor:
    """Process speech by separating vocals and focusing on the loudest speaker"""
    
    def __init__(self):
        print("Loading Audio Separator...")
        import sys
        if sys.platform == "win32":
            os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
        
        self.separator = Separator()
        
        print("Loading Speaker Diarization model...")
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
            
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
    def process_speech(self, audio_path):
        """Process speech by separating vocals and focusing on the loudest speaker"""
        try:
            print("Step 1: Separating vocals from background noise...")
            vocals_file_1 = self._separate_vocals(audio_path, 'UVR-MDX-NET-Inst_HQ_3.onnx')
            
            print("Step 2: Further cleaning vocals...")
            vocals_file_2 = self._separate_vocals(vocals_file_1, 'UVR_MDXNET_KARA_2.onnx')
            
            print("Step 3: Identifying speakers and processing their speech...")
            return self._process_speakers(vocals_file_2, audio_path)
            
        except Exception as e:
            print(f"Speech processing failed: {e}")
            return {'primary_speaker': None, 'speaker_files': {}, 'primary_file': audio_path}

    def _separate_vocals(self, audio_path, model_name):
        """Separate vocals using the specified model"""
        self.separator.load_model(model_filename=model_name)
        output_files = self.separator.separate([audio_path])
        
        vocals_file = None
        for file in output_files:
            if 'vocal' in file.lower() and 'instrumental' not in file.lower():
                vocals_file = file
                break
        
        return vocals_file if vocals_file else output_files[0]

    def _process_speakers(self, vocals_file, original_audio_path):
        """Process speakers from the vocals file"""
        audio, sr = librosa.load(vocals_file, sr=None)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        diarization = self.diarization_pipeline(
            vocals_file,
            min_speakers=1,
            max_speakers=5
        )
        
        speaker_energies = {}
        speaker_segments = {}
        
        # Process each speaker turn and calculate energy levels
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            
            segment = audio[start_sample:end_sample]
            energy = np.sqrt(np.mean(segment**2))
            
            if speaker not in speaker_energies:
                speaker_energies[speaker] = []
                speaker_segments[speaker] = []
            speaker_energies[speaker].append(energy)
            speaker_segments[speaker].append((start_sample, end_sample))
        
        # Find the speaker with highest average energy
        loudest_speaker = max(speaker_energies.items(), 
                            key=lambda x: np.mean(x[1]))[0]
        
        print(f"\nIdentified {len(speaker_energies)} speakers:")
        for speaker in speaker_energies:
            avg_energy = np.mean(speaker_energies[speaker])
            print(f"Speaker {speaker}: Average energy = {avg_energy:.4f}")
        print(f"\nPrimary speaker: {loudest_speaker}")
        
        speaker_audios = self._save_speaker_audios(
            audio, sr, speaker_segments, original_audio_path
        )
        
        return {
            'primary_speaker': loudest_speaker,
            'speaker_files': speaker_audios,
            'primary_file': speaker_audios[loudest_speaker]
        }

    def _save_speaker_audios(self, audio, sr, speaker_segments, original_audio_path):
        """Save individual audio files for each speaker"""
        speaker_audios = {}
        
        for speaker, segments in speaker_segments.items():
            mask = np.zeros_like(audio)
            for start, end in segments:
                mask[start:end] = 1
            
            speaker_audio = audio * mask
            speaker_audio = librosa.util.normalize(speaker_audio)
            
            speaker_path = os.path.splitext(original_audio_path)[0] + f'_speaker_{speaker}.wav'
            sf.write(speaker_path, speaker_audio, sr)
            speaker_audios[speaker] = speaker_path
            
            print(f"Saved audio for speaker {speaker} to: {speaker_path}")
        
        return speaker_audios

class AcousticApproachTranscriber:
    """Direct acoustic to graphemes with speech processing"""
    
    def __init__(self):
        print("Loading Wav2Vec2 model...")
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        
        print("Initializing Speech Processor...")
        self.speech_processor = SpeechProcessor()
        
    def transcribe(self, audio_path, use_processing=True):
        """Manual greedy CTC decoding of audio with optional processing"""
        if use_processing:
            print("Processing speech to identify speakers...")
            result = self.speech_processor.process_speech(audio_path)
            print(f"Primary speaker: {result['primary_speaker']}")
            
            transcriptions = {}
            for speaker, file_path in result['speaker_files'].items():
                print(f"\nTranscribing speaker {speaker}...")
                audio, rate = librosa.load(file_path, sr=16000, mono=True)
                
                # Apply noise reduction and normalization for better transcription
                audio = nr.reduce_noise(y=audio, sr=rate)
                audio = librosa.util.normalize(audio)
                
                # Convert audio to model input format
                inputs = self.wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt")
                
                with torch.no_grad():
                    logits = self.model(inputs.input_values).logits
                
                predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
                
                # Map token IDs to characters
                vocab_dict = self.wav2vec_processor.tokenizer.get_vocab()
                id_to_char = {v: k.replace('|', ' ') for k, v in vocab_dict.items()}
                
                # CTC decoding: remove duplicates and blank tokens
                ctc_blank = self.wav2vec_processor.tokenizer.pad_token_id
                decoded = []
                prev_token = None
                
                for idx in predicted_ids:
                    if idx != prev_token and idx != ctc_blank:
                        decoded.append(id_to_char.get(idx, ''))
                    prev_token = idx
                
                transcription = ''.join(decoded).replace('<s>', '').replace('</s>', '').strip()
                transcriptions[speaker] = transcription
                print(f"Speaker {speaker}: {transcription}")
            
            return {
                'transcriptions': transcriptions,
                'primary_speaker': result['primary_speaker'],
                'primary_transcription': transcriptions.get(result['primary_speaker'], ''),
                'audio_file': audio_path
            }
        else:
            # Process audio without speaker separation
            audio, rate = librosa.load(audio_path, sr=16000, mono=True)
            
            audio = nr.reduce_noise(y=audio, sr=rate)
            audio = librosa.util.normalize(audio)
            
            inputs = self.wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
            
            # Map token IDs to characters
            vocab_dict = self.wav2vec_processor.tokenizer.get_vocab()
            id_to_char = {v: k.replace('|', ' ') for k, v in vocab_dict.items()}
            
            # CTC decoding: remove duplicates and blank tokens
            ctc_blank = self.wav2vec_processor.tokenizer.pad_token_id
            decoded = []
            prev_token = None
            
            for idx in predicted_ids:
                if idx != prev_token and idx != ctc_blank:
                    decoded.append(id_to_char.get(idx, ''))
                prev_token = idx
            
            raw_transcription = ''.join(decoded).replace('<s>', '').replace('</s>', '').strip()
            
            return {
                'text': raw_transcription,
                'audio_file': audio_path
            }

class PronunciationTester:
    """Test pronunciation with speech processing"""
    
    def __init__(self):
        self.transcriber = AcousticApproachTranscriber()
    
    def test_pronunciation(self, audio_path):
        """Test pronunciation on audio"""
        print("\n" + "="*60)
        print("TESTING PRONUNCIATION")
        print("="*60)
        
        result = self.transcriber.transcribe(audio_path, use_processing=True)
        
        print("\nSpeaker transcriptions:")
        for speaker, text in result['transcriptions'].items():
            print(f"   {speaker}: {text}")
        
        print(f"\nPrimary speaker ({result['primary_speaker']}): {result['primary_transcription']}")
        return result
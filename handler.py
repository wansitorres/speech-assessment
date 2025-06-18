import runpod
import base64
import tempfile
import os
from src.speech_assessment import PronunciationTester

# Global instance to avoid reloading models on each request
pronunciation_tester = None

def handler(event):
    """
    RunPod serverless handler function
    
    Expected input format:
    {
        "audio": "path/to/uploaded/audio/file.wav"
    }
    
    OR for base64 compatibility:
    {
        "audio_base64": "base64_encoded_audio_data"
    }
    
    Returns:
    {
        "transcriptions": {...},
        "primary_speaker": "SPEAKER_00", 
        "primary_transcription": "transcribed text"
    }
    """
    global pronunciation_tester
    
    try:
        # Initialize the pronunciation tester once (global instance)
        if pronunciation_tester is None:
            print("Initializing Speech Assessment Engine...")
            pronunciation_tester = PronunciationTester()
            print("Speech Assessment Engine ready!")
        
        # Get input data
        input_data = event["input"]
        temp_audio_path = None
        
        # Handle different input formats
        if "audio" in input_data:
            # Direct file path (for file uploads)
            audio_file_path = input_data["audio"]
            
            # Validate file exists
            if not os.path.exists(audio_file_path):
                return {"error": f"Audio file not found: {audio_file_path}"}
            
            temp_audio_path = audio_file_path
            print(f"Processing uploaded audio file: {temp_audio_path}")
            
        elif "audio_base64" in input_data:
            audio_base64 = input_data["audio_base64"]
            
            try:
                audio_data = base64.b64decode(audio_base64)
            except Exception as e:
                return {"error": f"Invalid base64 audio data: {str(e)}"}
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_audio_path = temp_file.name
            
            print(f"Processing base64 audio file: {temp_audio_path}")
            
        else:
            return {"error": "Missing required field: 'audio' (file path) or 'audio_base64'"}
        
        # Use your original test_pronunciation method
        result = pronunciation_tester.test_pronunciation(
            audio_path=temp_audio_path
        )
        
        # Clean up temporary file (only if we created it from base64)
        try:
            if "audio_base64" in input_data and temp_audio_path:
                os.unlink(temp_audio_path)
        except Exception as e:
            print(f"Cleanup warning: {e}")
        
        # Remove file paths from response (not needed for API)
        if 'audio_file' in result:
            del result['audio_file']
        
        print("Processing completed successfully!")
        return result
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        return {"error": f"Processing failed: {str(e)}"}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
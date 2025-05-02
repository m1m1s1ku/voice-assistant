import os
import sys
import time
import argparse
import sounddevice as sd
import soundfile as sf
import tempfile
import mlx_whisper
from mlx_audio.tts.generate import generate_audio
from mlx_lm import load, generate
import speech_recognition as sr
import numpy as np

model, tokenizer = load("mlx-community/Mistral-Small-24B-Instruct-2501-4bit")

def record_audio_with_silence_detection(timeout=10, phrase_time_limit=None, sample_rate=16000):
    """Record audio from the microphone with silence detection."""
    recognizer = sr.Recognizer()
    
    recognizer.pause_threshold = 2
    recognizer.energy_threshold = 100
    
    print("Waiting for speech... (Speech now)")
    
    with sr.Microphone(sample_rate=sample_rate) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0 # Normalize audio data [-1, 1]
            
            return audio_float, sample_rate
            
        except sr.WaitTimeoutError:
            return np.zeros(0, dtype=np.float32), sample_rate

def save_audio_to_file(audio, sample_rate, filename):
    """Save audio data to a WAV file."""
    sf.write(filename, audio, sample_rate)
    print(f"Audio saved in {filename}")

def transcribe_audio(audio_file):
    """Transcribe audio file to text using Whisper local model."""
    ref_text = mlx_whisper.transcribe(audio_file, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")[
                    "text"
                ]
    return ref_text

def get_llm_response(text, conversation_history):
    """Get response from an LLM based on the transcribed text and conversation history."""
    print(f"Getting LLM response for : {text}")
        
    # TODO : Make prompt configurable for multilanguage support
    messages = [
        {"role": "system", "content": "Tu es un assistant IA utile et concis. Réponds en français. Maximum une phrase à la fois. Ne répond surtout rien d'autre qu'une phrase, aucun exemple de code."}
    ]
    
    messages.extend(conversation_history)

    user_message = {"role": "user", "content": text}
    messages.append(user_message)
    
    # print(messages)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    response_content = generate(model, tokenizer, prompt=prompt, verbose=True)

    conversation_history.append(user_message)
    conversation_history.append({"role": "assistant", "content": response_content})
    
    return response_content

def text_to_speech(text, output_file="response.wav", voice=None):
    """Convert text to speech using generate_audio from mlx-audio."""
    print("Converting text in speech...")
    
    base_filename = os.path.splitext(output_file)[0]
    
    # Generate speech from text using generate_audio
    generate_audio(
        text=text,
        model_path="OuteAI/Llama-OuteTTS-1.0-1B",
        voice=voice,
        speed=1.0,
        lang_code="f",
        file_prefix=base_filename,
        audio_format="wav",
        sample_rate=24000,
        join_audio=True,
        verbose=True
    )
    
    actual_output_file = f"{base_filename}.wav"
    
    return actual_output_file

def play_audio(filename):
    """Play audio from a file."""
    data, fs = sf.read(filename)
    sd.play(data, fs)
    sd.wait()

def conversation_loop(args):
    """Run a continuous conversation loop."""
    print("Starting conversation, say something...")
    
    temp_dir = tempfile.gettempdir()
    conversation_active = True
    
    conversation_history = []
    
    while conversation_active:
        try:
            # Step 1: Record audio from microphone with silence detection
            audio, sample_rate = record_audio_with_silence_detection(
                timeout=args.timeout, 
                phrase_time_limit=args.max_duration, 
                sample_rate=args.sample_rate
            )
            
            if len(audio) == 0:
                print("No audio detected, starting for next interaction...")
                continue
            
            # Save the recorded audio to a temporary file
            input_file = os.path.join(temp_dir, "input.wav")
            save_audio_to_file(audio, sample_rate, input_file)
            
            # Step 2: Transcribe audio to text
            transcribed_text = transcribe_audio(input_file)
            print(f"Transcribed text: {transcribed_text}")
            
            # Vérifier si la transcription est vide
            if not transcribed_text.strip():
                print("Blank transcript, starting next interaction...")
                continue
            
            # Step 3: Get response from LLM with conversation history
            llm_response = get_llm_response(transcribed_text, conversation_history)
            print(f"LLM response : {llm_response}")
                        
            # Step 4: Convert LLM response to speech
            output_file = text_to_speech(llm_response, voice=args.voice)
            
            # Step 5: Play the response
            play_audio(output_file)
            
            print("\nReady for next interaction, say something...")
            
        except KeyboardInterrupt:
            conversation_active = False
        except Exception as e:
            print(f"Error while conversation: {e}")

def main():
    parser = argparse.ArgumentParser(description="MLX-Audio demo with microphone input and LLM response")
    parser.add_argument("--timeout", type=int, default=10, help="Délai d'attente maximum pour commencer à parler (en secondes)")
    parser.add_argument("--max-duration", type=int, default=None, help="Durée maximale d'enregistrement (en secondes)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Taux d'échantillonnage audio")
    parser.add_argument("--voice", type=str, default=None, help="Chemin vers le fichier de voix OuteTTS (ex: alexandra.json)")
    parser.add_argument("--model", type=str, default="OuteAI/Llama-OuteTTS-1.0-1B", help="Chemin du modèle pour TTS")
    args = parser.parse_args()
    
    # Start the conversation loop instead of a single interaction
    conversation_loop(args)

if __name__ == "__main__":
    main()
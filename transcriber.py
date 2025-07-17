import whisper
import sys
import os
from pathlib import Path
import argparse
import time

def transcribe_audio(audio_file_path, model_size="base", language=None, output_file=None):
    """
    Transcribe audio file using Whisper
    
    Args:
        audio_file_path (str): Path to the audio file
        model_size (str): Whisper model size (tiny, base, small, medium, large)
        language (str): Language code (e.g., 'es' for Spanish, 'en' for English)
        output_file (str): Path to save transcription (optional)
    
    Returns:
        dict: Transcription result
    """
    
    # Verificar si el archivo existe
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"El archivo {audio_file_path} no existe")
    
    # Verificar extensión del archivo
    supported_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac']
    file_extension = Path(audio_file_path).suffix.lower()
    
    if file_extension not in supported_formats:
        print(f"Advertencia: El formato {file_extension} puede no ser compatible")
        print(f"Formatos soportados: {', '.join(supported_formats)}")
    
    print(f"Cargando modelo Whisper '{model_size}'...")
    start_time = time.time()
    
    # Cargar el modelo
    model = whisper.load_model(model_size)
    
    load_time = time.time() - start_time
    print(f"Modelo cargado en {load_time:.2f} segundos")
    
    print(f"Transcribiendo archivo: {audio_file_path}")
    transcribe_start = time.time()
    
    # Transcribir el audio
    if language:
        result = model.transcribe(audio_file_path, language=language)
    else:
        result = model.transcribe(audio_file_path)
    
    transcribe_time = time.time() - transcribe_start
    print(f"Transcripción completada en {transcribe_time:.2f} segundos")
    
    # Guardar resultado si se especifica archivo de salida
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        print(f"Transcripción guardada en: {output_file}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Transcriptor de audio usando Whisper')
    parser.add_argument('audio_file', help='Ruta al archivo de audio')
    parser.add_argument('-m', '--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Tamaño del modelo Whisper (default: base)')
    parser.add_argument('-l', '--language', help='Código de idioma (ej: es, en, fr)')
    parser.add_argument('-o', '--output', help='Archivo de salida para la transcripción')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    try:
        # Transcribir el audio
        result = transcribe_audio(
            args.audio_file, 
            args.model, 
            args.language, 
            args.output
        )
        
        # Mostrar resultado
        print("\n" + "="*50)
        print("TRANSCRIPCIÓN:")
        print("="*50)
        print(result['text'])
        
        if args.verbose:
            print("\n" + "="*50)
            print("INFORMACIÓN DETALLADA:")
            print("="*50)
            print(f"Idioma detectado: {result['language']}")
            
            print("\nSegmentos:")
            for i, segment in enumerate(result['segments'], 1):
                start = segment['start']
                end = segment['end']
                text = segment['text']
                print(f"{i:2d}. [{start:6.2f}s - {end:6.2f}s]: {text}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
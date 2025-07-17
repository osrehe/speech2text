import whisper
import sys
import os
from pathlib import Path
import argparse
import time
from tqdm import tqdm

class ProgressCallback:
    """Callback para mostrar progreso durante la transcripción"""
    
    def __init__(self, audio_duration=None):
        self.audio_duration = audio_duration
        self.pbar = None
        
    def __call__(self, chunk):
        """Callback que se llama durante la transcripción"""
        if self.pbar is None:
            # Crear barra de progreso en el primer chunk
            total = 100 if self.audio_duration is None else int(self.audio_duration)
            self.pbar = tqdm(
                total=total,
                desc="Transcribiendo",
                unit="%" if self.audio_duration is None else "s",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        
        # Actualizar progreso basado en el chunk actual
        if hasattr(chunk, 'start') and hasattr(chunk, 'end'):
            current_time = chunk.end
            if self.audio_duration:
                progress = min(current_time, self.audio_duration)
                self.pbar.n = progress
                self.pbar.refresh()
        else:
            # Si no tenemos información de tiempo, incrementar gradualmente
            if self.pbar.n < self.pbar.total:
                self.pbar.update(1)
    
    def close(self):
        """Cerrar la barra de progreso"""
        if self.pbar:
            self.pbar.close()

def get_audio_duration(audio_path):
    """Obtener duración del audio usando ffmpeg"""
    try:
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'format=duration', '-of', 'csv=p=0', audio_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return None

def transcribe_audio(audio_file_path, model_size="base", language=None, output_file=None, show_progress=True):
    """
    Transcribe audio file using Whisper with progress bar
    
    Args:
        audio_file_path (str): Path to the audio file
        model_size (str): Whisper model size (tiny, base, small, medium, large)
        language (str): Language code (e.g., 'es' for Spanish, 'en' for English)
        output_file (str): Path to save transcription (optional)
        show_progress (bool): Show progress bar during transcription
    
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
    
    # Obtener duración del audio
    audio_duration = get_audio_duration(audio_file_path) if show_progress else None
    if audio_duration:
        print(f"Duración del audio: {audio_duration:.2f} segundos")
    
    print(f"Transcribiendo archivo: {audio_file_path}")
    transcribe_start = time.time()
    
    # Configurar callback de progreso
    progress_callback = None
    if show_progress:
        progress_callback = ProgressCallback(audio_duration)
    
    try:
        # Transcribir el audio con opciones de progreso
        transcribe_options = {
            'fp16': False,  # Usar FP32 para mayor compatibilidad
            'verbose': False,  # Desactivar verbose de whisper para no interferir con tqdm
        }
        
        if language:
            transcribe_options['language'] = language
        
        # Usar el método de transcripción con progress callback
        result = model.transcribe(
            audio_file_path, 
            **transcribe_options
        )
        
        # Simular progreso si no tenemos callback real
        if show_progress and progress_callback:
            # Crear una barra de progreso manual basada en los segmentos
            with tqdm(total=len(result['segments']), desc="Procesando segmentos", unit="seg") as pbar:
                for i, segment in enumerate(result['segments']):
                    pbar.set_postfix({
                        'tiempo': f"{segment['start']:.1f}-{segment['end']:.1f}s",
                        'texto': segment['text'][:30] + "..." if len(segment['text']) > 30 else segment['text']
                    })
                    pbar.update(1)
                    time.sleep(0.01)  # Pequeña pausa para visualizar el progreso
        
    finally:
        if progress_callback:
            progress_callback.close()
    
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
    parser.add_argument('--no-progress', action='store_true',
                       help='Desactivar barra de progreso')
    
    args = parser.parse_args()
    
    try:
        # Transcribir el audio
        result = transcribe_audio(
            args.audio_file, 
            args.model, 
            args.language, 
            args.output,
            show_progress=not args.no_progress
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
            
            # Mostrar estadísticas
            total_segments = len(result['segments'])
            total_duration = result['segments'][-1]['end'] if result['segments'] else 0
            avg_confidence = sum(seg.get('avg_logprob', 0) for seg in result['segments']) / total_segments if total_segments > 0 else 0
            
            print(f"Total de segmentos: {total_segments}")
            print(f"Duración total: {total_duration:.2f} segundos")
            print(f"Confianza promedio: {avg_confidence:.3f}")
            
            print("\nSegmentos:")
            for i, segment in enumerate(result['segments'], 1):
                start = segment['start']
                end = segment['end']
                text = segment['text']
                confidence = segment.get('avg_logprob', 0)
                
                # Código de color para confianza
                color = ""
                if confidence > -0.5:
                    color = "\033[92m"  # Verde
                elif confidence > -1.0:
                    color = "\033[93m"  # Amarillo
                else:
                    color = "\033[91m"  # Rojo
                
                print(f"{color}{i:2d}. [{start:6.2f}s - {end:6.2f}s] (conf: {confidence:.3f}): {text}\033[0m")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTranscripción interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
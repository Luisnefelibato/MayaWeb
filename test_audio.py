import requests
import pyaudio
import wave
import os
import time

def record_audio(filename="test_audio.wav", seconds=5, rate=44100, chunk=1024, channels=1):
    """
    Graba audio del micrófono y lo guarda en un archivo.
    
    Args:
        filename (str): Nombre del archivo donde se guardará la grabación.
        seconds (int): Duración de la grabación en segundos.
        rate (int): Tasa de muestreo.
        chunk (int): Tamaño del búfer de audio.
        channels (int): Número de canales (1 para mono, 2 para estéreo).
    """
    p = pyaudio.PyAudio()
    
    print(f"Grabando {seconds} segundos de audio...")
    
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    frames = []
    
    for i in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Grabación finalizada.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Guardar la grabación en un archivo
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Audio guardado como {filename}")
    return filename

def test_transcribe(audio_file, server_url="http://localhost:5000"):
    """
    Prueba el endpoint /transcribe enviando un archivo de audio.
    
    Args:
        audio_file (str): Ruta al archivo de audio a transcribir.
        server_url (str): URL del servidor.
    """
    endpoint = f"{server_url}/transcribe"
    
    with open(audio_file, 'rb') as f:
        files = {'audio': f}
        print(f"Enviando {audio_file} a {endpoint}...")
        response = requests.post(endpoint, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Transcripción: {result['text']}")
        return result['text']
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def test_voice_chat(audio_file, session_id="test_session", server_url="http://localhost:5000"):
    """
    Prueba el endpoint /voice-chat enviando un archivo de audio.
    
    Args:
        audio_file (str): Ruta al archivo de audio a enviar.
        session_id (str): ID de sesión para la conversación.
        server_url (str): URL del servidor.
    """
    endpoint = f"{server_url}/voice-chat"
    
    with open(audio_file, 'rb') as f:
        files = {'audio': f}
        data = {'session_id': session_id}
        print(f"Enviando {audio_file} a {endpoint}...")
        response = requests.post(endpoint, files=files, data=data)
    
    if response.status_code == 200:
        # Guardar el audio de respuesta
        output_file = "respuesta.mp3"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"Respuesta de audio guardada como {output_file}")
        return output_file
    else:
        try:
            error_info = response.json()
            print(f"Error: {response.status_code}")
            print(f"Detalle: {error_info}")
            if 'transcription' in error_info:
                print(f"Transcripción: {error_info['transcription']}")
            if 'response' in error_info:
                print(f"Respuesta textual: {error_info['response']}")
        except:
            print(f"Error: {response.status_code}")
            print(response.text)
        return None

def main():
    """Función principal para probar los endpoints de audio."""
    server_url = input("URL del servidor (dejar en blanco para localhost:5000): ").strip()
    if not server_url:
        server_url = "http://localhost:5000"
    
    while True:
        print("\nOpciones:")
        print("1. Grabar audio y transcribir")
        print("2. Grabar audio y enviar a voice-chat")
        print("3. Salir")
        
        opcion = input("Seleccione una opción (1-3): ").strip()
        
        if opcion == "1":
            # Grabar audio y transcribir
            filename = record_audio()
            test_transcribe(filename, server_url)
        
        elif opcion == "2":
            # Grabar audio y enviar a voice-chat
            filename = record_audio()
            test_voice_chat(filename, server_url=server_url)
        
        elif opcion == "3":
            # Salir
            print("¡Hasta luego!")
            break
        
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()
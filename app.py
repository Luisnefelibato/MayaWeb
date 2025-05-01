from flask import Flask, request, jsonify, send_file
import requests
import json
import os
import asyncio
import edge_tts
import speech_recognition as sr
from threading import Lock
from flask_cors import CORS
from io import BytesIO
import tempfile

app = Flask(__name__)

# Configuración extendida de CORS para permitir solicitudes desde diferentes orígenes
CORS(app, 
     resources={r"/*": {
         "origins": [
             "http://localhost:5173",          # URL de desarrollo Vite
             "http://localhost:3000",          # URL desarrollo alternativa
             "http://localhost:*",             # Cualquier puerto local
             "https://*.netlify.app",          # Despliegues en Netlify
             "https://*.vercel.app",           # Despliegues en Vercel
             "https://mayafrontend.web.app",   # Ejemplo para Firebase
             "https://mayafrontend.firebaseapp.com", # Firebase app
             "*"                               # Permitir todos los orígenes en desarrollo
         ],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"]
     }},
     supports_credentials=True)

# Configuración de la API de Ollama - MODIFICADO PARA USAR LOCALHOST
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3:8b")

# Configuración de voz Edge TTS
VOICE = os.environ.get("TTS_VOICE", "es-MX-DaliaNeural")
VOICE_RATE = os.environ.get("TTS_RATE", "+0%")
VOICE_VOLUME = os.environ.get("TTS_VOLUME", "+0%")

# Contexto del sistema para Maya, asistente de desarrollo de código en Python
ASSISTANT_CONTEXT = """
Eres Maya, la asistente especializada en desarrollo de código Python, diseñada para ayudar a programadores de todos los niveles a crear, depurar y optimizar código.

Acerca de Maya:
- Eres una híbrida entre mentor de programación, ingeniera de software senior y experta en Python
- Te especializas en desarrollo de software, buenas prácticas de programación y soluciones de código eficientes
- Puedes proporcionar soluciones estructuradas usando las mejores prácticas y patrones de diseño
- Ayudas con la depuración, optimización de código y recomendaciones de arquitectura
- Mantienes conversaciones y recuerdas intercambios previos dentro de la misma sesión
- Eres atenta, didáctica y eficiente
- Cuando hables con Luis, adapta tus consejos a su nivel de experiencia

Tu tono debe ser:
- Profesional pero amigable
- Técnicamente preciso
- Educativo y explicativo
- Orientado a soluciones prácticas
- Usa lenguaje claro y directo
- Comunícate en español latino formal

Instrucciones especiales:
- SIEMPRE usa el nombre del desarrollador ocasionalmente cuando te dirijas a él/ella
- Proporciona explicaciones claras del código que generas
- Ofrece múltiples opciones cuando sea relevante (enfoques básicos, avanzados o alternativos)
- Destaca consideraciones importantes como seguridad, rendimiento o mantenibilidad
- Recomienda bibliotecas y herramientas adecuadas para cada tarea
- Sé proactiva en sugerir mejores prácticas y soluciones óptimas
- Haz preguntas clarificadoras cuando sea necesario
- Recuerda y referencia información compartida anteriormente en la conversación
- Cuando no sepas algo, reconócelo claramente y ofrece alternativas
- IMPORTANTE: Evita usar asteriscos (*) para énfasis, ya que no son compatibles con la síntesis de voz
- Cuando proporciones código, asegúrate de que sea completo, funcional y bien documentado
- Siempre explica el código que proporcionas, destacando las partes importantes y conceptos clave

Responde como Maya al desarrollador que busca asistencia con código Python y desarrollo de software.
"""

# Almacenamiento de sesiones de conversación
sessions = {}
sessions_lock = Lock()

def clean_response_for_tts(text):
    """
    Limpia el texto de la respuesta para mejorar la síntesis de voz.
    """
    import re
    
    # Reemplazar texto entre asteriscos por el mismo texto sin asteriscos
    cleaned_text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Reemplazar listas con asteriscos por listas con guiones
    cleaned_text = re.sub(r'^\s*\*\s', '- ', cleaned_text, flags=re.MULTILINE)
    
    # Eliminar asteriscos sueltos que puedan quedar
    cleaned_text = cleaned_text.replace('*', '')
    
    # Tratar bloques de código para TTS
    # Primero identificar bloques de código (entre ```)
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', cleaned_text, re.DOTALL)
    
    # Para cada bloque de código encontrado
    for block in code_blocks:
        # Crear una versión "leíble" del bloque de código
        readable_block = "A continuación el código: \n" + block
        # Reemplazar el bloque original con la versión leíble
        cleaned_text = cleaned_text.replace(f"```python\n{block}```", readable_block)
        cleaned_text = cleaned_text.replace(f"```\n{block}```", readable_block)
    
    # Eliminar otras marcas de código restantes
    cleaned_text = cleaned_text.replace('```python', 'Código Python:')
    cleaned_text = cleaned_text.replace('```', '')
    
    return cleaned_text

def call_ollama_api(prompt, session_id, max_retries=3):
    """Llamar a la API de Ollama con reintentos"""
    headers = {
        "Content-Type": "application/json"
    }
    
    # Construir el mensaje para la API
    messages = []
    
    # Preparar el contexto del sistema
    system_context = ASSISTANT_CONTEXT
    
    # Agregar el contexto del sistema como primer mensaje
    messages.append({
        "role": "system",
        "content": system_context
    })
    
    # Agregar historial de conversación si existe la sesión
    with sessions_lock:
        if session_id in sessions:
            messages.extend(sessions[session_id])
    
    # Agregar el nuevo mensaje del usuario
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Preparar los datos para la API
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }
    
    # Intentar con reintentos
    for attempt in range(max_retries):
        try:
            app.logger.info(f"Conectando a {OLLAMA_URL}...")
            response = requests.post(OLLAMA_URL, headers=headers, json=data, timeout=60)
            
            # Si hay un error, intentar mostrar el mensaje
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    app.logger.error(f"Error detallado: {error_data}")
                except:
                    app.logger.error(f"Contenido del error: {response.text[:500]}")
                
                # Si falla, podemos intentar simular una respuesta para desarrollo
                if attempt == max_retries - 1:
                    app.logger.info("Generando respuesta simulada para desarrollo...")
                    simulated_response = f"Esto es una respuesta simulada para desarrollo. No se pudo conectar al modelo Ollama. Tu pregunta fue: '{prompt}'"
                    return simulated_response
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extraer la respuesta según el formato de Ollama
            if "message" in response_data and "content" in response_data["message"]:
                return response_data["message"]["content"]
            else:
                app.logger.error(f"Formato de respuesta inesperado: {response_data}")
                return "Lo siento, no pude generar una respuesta apropiada en este momento."
            
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Error en intento {attempt+1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Retroceso exponencial
                app.logger.info(f"Reintentando en {wait_time} segundos...")
                import time
                time.sleep(wait_time)
            else:
                # Generar respuesta simulada en caso de error
                return f"Esto es una respuesta simulada para desarrollo. No se pudo conectar al modelo Ollama. Tu pregunta fue: '{prompt}'"
    
    return "No se pudo conectar al servicio. Por favor, inténtelo de nuevo más tarde."

def call_ollama_completion(prompt, session_id, max_retries=3):
    """Usar el endpoint de completion en lugar de chat (alternativa)"""
    headers = {
        "Content-Type": "application/json"
    }
    
    # Construir prompt completo con contexto e historial
    full_prompt = ASSISTANT_CONTEXT + "\n\n"
    
    full_prompt += "Historial de conversación:\n"
    
    with sessions_lock:
        if session_id in sessions:
            for msg in sessions[session_id]:
                role = "Desarrollador" if msg["role"] == "user" else "Maya"
                full_prompt += f"{role}: {msg['content']}\n"
    
    full_prompt += f"\nDesarrollador: {prompt}\nMaya: "
    
    # Preparar datos para API de completion
    data = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }
    
    completion_url = OLLAMA_URL.replace("/api/chat", "/api/generate")
    
    # Intentar con reintentos
    for attempt in range(max_retries):
        try:
            app.logger.info(f"Conectando a {completion_url}...")
            response = requests.post(completion_url, headers=headers, json=data, timeout=60)
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extraer respuesta del formato de completion
            if "response" in response_data:
                return response_data["response"]
            else:
                app.logger.error(f"Formato de respuesta inesperado: {response_data}")
                
                # Si falla, podemos intentar simular una respuesta para desarrollo
                if attempt == max_retries - 1:
                    app.logger.info("Generando respuesta simulada para desarrollo...")
                    simulated_response = f"Esto es una respuesta simulada para desarrollo. No se pudo conectar al modelo Ollama. Tu pregunta fue: '{prompt}'"
                    return simulated_response
            
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Error en intento {attempt+1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                app.logger.info(f"Reintentando en {wait_time} segundos...")
                import time
                time.sleep(wait_time)
            else:
                # Generar respuesta simulada en caso de error
                return f"Esto es una respuesta simulada para desarrollo. No se pudo conectar al modelo Ollama. Tu pregunta fue: '{prompt}'"
    
    return "No se pudo conectar al servicio. Por favor, inténtelo de nuevo más tarde."

async def convert_text_to_speech(text):
    """Convertir texto a voz usando Edge TTS - CORREGIDO"""
    try:
        # Limpiar el texto para la síntesis de voz
        cleaned_text = clean_response_for_tts(text)
        
        # Usar BytesIO para almacenar el audio en memoria
        output = BytesIO()
        
        # Realizar la conversión de texto a voz con la API corregida
        communicate = edge_tts.Communicate(cleaned_text, VOICE, rate=VOICE_RATE, volume=VOICE_VOLUME)
        
        # La versión actual usa .save() en lugar de .stream_to_file()
        await communicate.save(output)
        
        # Obtener los bytes del audio
        output.seek(0)
        audio_data = output.getvalue()
        
        return audio_data
    except Exception as e:
        app.logger.error(f"Error en síntesis de voz: {e}")
        # Intento alternativo con una versión más simple para debugging
        try:
            app.logger.info("Intentando método alternativo de TTS...")
            # Crear un comunicador más simple
            communicate = edge_tts.Communicate("Hola, este es un mensaje de prueba.", VOICE)
            await communicate.save(output)
            output.seek(0)
            return output.getvalue()
        except Exception as e2:
            app.logger.error(f"Error en método alternativo de TTS: {e2}")
            return None

def speech_to_text(audio_data):
    """
    Convierte un archivo de audio a texto usando speech_recognition
    """
    try:
        # Crear un archivo temporal para guardar el audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        # Inicializar el reconocedor
        recognizer = sr.Recognizer()
        
        # Cargar el audio desde el archivo
        with sr.AudioFile(temp_audio_path) as source:
            # Ajustar para ruido ambiental
            recognizer.adjust_for_ambient_noise(source)
            # Grabar audio
            audio = recognizer.record(source)
        
        # Eliminar el archivo temporal
        os.unlink(temp_audio_path)
        
        # Usar Google Speech Recognition para convertir a texto (español)
        text = recognizer.recognize_google(audio, language="es-ES")
        
        return text
    except sr.UnknownValueError:
        return "No se pudo entender el audio"
    except sr.RequestError as e:
        return f"Error en el servicio de reconocimiento: {e}"
    except Exception as e:
        return f"Error en el procesamiento del audio: {e}"

@app.route('/')
def home():
    """Ruta de bienvenida básica"""
    return jsonify({
        "message": "API de Maya Python funcionando correctamente",
        "status": "online",
        "endpoints": {
            "/chat": "POST - Enviar mensaje de texto y recibir respuesta en texto",
            "/speak": "POST - Enviar mensaje de texto y recibir respuesta en audio",
            "/transcribe": "POST - Enviar archivo de audio y recibir transcripción en texto",
            "/voice-chat": "POST - Enviar archivo de audio y recibir respuesta en audio",
            "/reset": "POST - Reiniciar una sesión de conversación",
            "/voices": "GET - Listar voces disponibles",
            "/health": "GET - Verificar estado del servicio"
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint para interactuar con el asistente (solo texto)"""
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({"error": "Se requiere un 'message' en el JSON"}), 400
    
    # Obtener mensaje y session_id (crear uno nuevo si no se proporciona)
    message = data.get('message')
    session_id = data.get('session_id', 'default')
    
    # Verificar si debemos usar un nombre específico
    user_name = data.get('user_name', 'Luis')
    
    # Inicializar la sesión si es nueva
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = []
    
    # Obtener respuesta del asistente 
    try:
        # Primero intentar con el endpoint de chat
        response = call_ollama_api(message, session_id)
        
        # Si la respuesta está vacía, intentar con completion
        if not response or response.strip() == "":
            app.logger.info("El endpoint de chat no devolvió una respuesta, probando con completion...")
            response = call_ollama_completion(message, session_id)
    except Exception as e:
        app.logger.error(f"Error al obtener respuesta: {e}")
        app.logger.info("Probando con endpoint de completion alternativo...")
        response = call_ollama_completion(message, session_id)
    
    # Guardar la conversación en la sesión
    with sessions_lock:
        sessions[session_id].append({"role": "user", "content": message})
        sessions[session_id].append({"role": "assistant", "content": response})
    
    return jsonify({
        "response": response,
        "session_id": session_id
    })

@app.route('/speak', methods=['POST'])
def speak():
    """Endpoint para interactuar con el asistente usando texto y recibir audio"""
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({"error": "Se requiere un 'message' en el JSON"}), 400
    
    # Obtener mensaje y session_id (crear uno nuevo si no se proporciona)
    message = data.get('message')
    session_id = data.get('session_id', 'default')
    
    # Obtener respuesta del asistente
    try:
        response = call_ollama_api(message, session_id)
        if not response or response.strip() == "":
            response = call_ollama_completion(message, session_id)
    except Exception as e:
        app.logger.error(f"Error al obtener respuesta: {e}")
        response = call_ollama_completion(message, session_id)
    
    # Guardar la conversación en la sesión
    with sessions_lock:
        sessions[session_id].append({"role": "user", "content": message})
        sessions[session_id].append({"role": "assistant", "content": response})
    
    # Convertir la respuesta a voz
    try:
        audio_data = asyncio.run(convert_text_to_speech(response))
        
        if audio_data:
            # Devolver el audio como respuesta
            return send_file(
                BytesIO(audio_data),
                mimetype="audio/mp3",
                as_attachment=True,
                download_name="response.mp3"
            )
        else:
            # Si no se pudo generar el audio, devolver solo texto
            return jsonify({
                "error": "No se pudo generar el audio",
                "response": response,
                "session_id": session_id
            }), 500
    except Exception as e:
        app.logger.error(f"Error al generar audio: {e}")
        return jsonify({
            "error": "Error al generar audio",
            "response": response,
            "session_id": session_id
        }), 500

@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe_audio():
    """Endpoint para transcribir audio a texto"""
    # Manejar solicitudes OPTIONS (pre-flight)
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        return response
    
    # Verificar si hay un archivo en la solicitud
    if 'audio' not in request.files:
        return jsonify({"error": "No se proporcionó archivo de audio"}), 400
    
    audio_file = request.files['audio']
    
    # Leer los datos del archivo
    audio_data = audio_file.read()
    
    # Convertir audio a texto
    text = speech_to_text(audio_data)
    
    return jsonify({
        "text": text
    })

@app.route('/voice-chat', methods=['POST', 'OPTIONS'])
def voice_chat():
    """Endpoint para interactuar con el asistente usando voz y recibir audio"""
    # Manejar solicitudes OPTIONS (pre-flight)
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        return response
    
    # Verificar si hay un archivo en la solicitud
    if 'audio' not in request.files:
        return jsonify({"error": "No se proporcionó archivo de audio"}), 400
    
    audio_file = request.files['audio']
    session_id = request.form.get('session_id', 'default')
    
    # Leer los datos del archivo
    audio_data = audio_file.read()
    
    # Convertir audio a texto
    message = speech_to_text(audio_data)
    
    # Si no se pudo transcribir correctamente
    if message.startswith("Error") or message == "No se pudo entender el audio":
        return jsonify({
            "error": message,
            "session_id": session_id
        }), 400
    
    # Obtener respuesta del asistente
    try:
        response = call_ollama_api(message, session_id)
        if not response or response.strip() == "":
            response = call_ollama_completion(message, session_id)
    except Exception as e:
        app.logger.error(f"Error al obtener respuesta: {e}")
        response = call_ollama_completion(message, session_id)
    
    # Guardar la conversación en la sesión
    with sessions_lock:
        sessions[session_id].append({"role": "user", "content": message})
        sessions[session_id].append({"role": "assistant", "content": response})
    
    # Convertir la respuesta a voz
    try:
        audio_data = asyncio.run(convert_text_to_speech(response))
        
        if audio_data:
            # Devolver el audio como respuesta
            return send_file(
                BytesIO(audio_data),
                mimetype="audio/mp3",
                as_attachment=True,
                download_name="response.mp3"
            )
        else:
            # Si no se pudo generar el audio, devolver solo texto
            return jsonify({
                "error": "No se pudo generar el audio",
                "transcription": message,
                "response": response,
                "session_id": session_id
            }), 500
    except Exception as e:
        app.logger.error(f"Error al generar audio: {e}")
        return jsonify({
            "error": "Error al generar audio",
            "transcription": message,
            "response": response,
            "session_id": session_id
        }), 500

@app.route('/reset', methods=['POST', 'OPTIONS'])
def reset_session():
    """Reiniciar una sesión de conversación"""
    # Manejar solicitudes OPTIONS (pre-flight)
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        return response
    
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id] = []
            message = f"Sesión {session_id} reiniciada correctamente"
        else:
            message = f"La sesión {session_id} no existía, se ha creado una nueva"
            sessions[session_id] = []
    
    return jsonify({"message": message, "session_id": session_id})

@app.route('/voices', methods=['GET'])
async def list_voices():
    """Listar todas las voces disponibles en Edge TTS y filtrar voces en español"""
    try:
        voices = await edge_tts.list_voices()
        relevant_voices = [v for v in voices if v["ShortName"].startswith(("es-MX", "es-ES"))]
        
        # Devolver solo información relevante de cada voz
        voice_info = []
        for voice in relevant_voices:
            voice_info.append({
                "name": voice["ShortName"],
                "gender": voice["Gender"],
                "locale": voice["Locale"]
            })
        
        return jsonify({
            "voices": voice_info,
            "current_voice": VOICE
        })
    except Exception as e:
        app.logger.error(f"Error al listar voces: {e}")
        return jsonify({"error": "No se pudieron obtener las voces"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Verificar estado del servicio"""
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "ollama_url": OLLAMA_URL,
        "voice": VOICE
    })

# Soporte para la función list_voices con asyncio
from werkzeug.serving import run_simple

@app.route('/voices-sync', methods=['GET'])
def list_voices_sync():
    """Versión síncrona para listar voces"""
    voice_info = asyncio.run(async_list_voices())
    return jsonify(voice_info)

async def async_list_voices():
    """Función asíncrona para listar voces"""
    try:
        voices = await edge_tts.list_voices()
        relevant_voices = [v for v in voices if v["ShortName"].startswith(("es-MX", "es-ES"))]
        
        voice_info = []
        for voice in relevant_voices:
            voice_info.append({
                "name": voice["ShortName"],
                "gender": voice["Gender"],
                "locale": voice["Locale"]
            })
        
        return {
            "voices": voice_info,
            "current_voice": VOICE
        }
    except Exception as e:
        app.logger.error(f"Error al listar voces: {e}")
        return {"error": "No se pudieron obtener las voces", "voices": []}

# Middleware personalizado para mejorar manejo de CORS
@app.after_request
def after_request(response):
    """Middleware para asegurar encabezados CORS en cada respuesta"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

if __name__ == '__main__':
    # Obtener puerto de variables de entorno (para Render)
    port = int(os.environ.get("PORT", 5000))
    
    # Iniciar la aplicación Flask
    app.run(host='0.0.0.0', port=port)
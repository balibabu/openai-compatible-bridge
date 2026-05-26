from flask import Flask, request, jsonify, Response
import json
import time
from datetime import datetime, timezone
from google import genai
from google.genai import types

app = Flask(__name__)
import os
api_key = os.getenv("GAPI")

try:
    client = genai.Client(api_key=api_key)
    models = client.models.list()
    tags=[{"name": model.name.split('/')[1], "model": model.name.split('/')[1], "modified_at": '2026-05-26T05:07:21.206423+00:00', "size": 0, "digest": "mock"} for model in models]
except Exception as e:
    print(f"Error initializing client: {e}. Did you set GEMINI_API_KEY?")
    exit(1)

def get_current_time():
    return datetime.now(timezone.utc).isoformat()

@app.route('/api/tags', methods=['GET'])
def list_models():
    return jsonify({ "models": tags })

@app.route('/api/show', methods=['POST'])
def show_model_info():
    return jsonify({"modelfile": "FROM mymodel", "parameters": "", "template": "{{ .Prompt }}"})

@app.route('/api/chat', methods=['POST'])
def chat():
    """The core Ollama chat endpoint, now powered by Google Gemini."""
    data = request.json
    messages = data.get('messages', [])
    stream = data.get('stream', True)
    model = data.get('model', 'mymodel')
    
    # 1. Map Ollama messages to Gemini format
    gemini_contents = []
    system_instruction = None
    
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '')
        if role == 'system':
            system_instruction = content
        else:
            gemini_role = "model" if role == "assistant" else "user"
            gemini_contents.append(
                types.Content(role=gemini_role, parts=[types.Part.from_text(text=content)])
            )

    config = types.GenerateContentConfig()
    if system_instruction:
        config.system_instruction = system_instruction

    # 2. Handle Streaming (Ollama NDJSON format)
    if stream:
        def generate_stream():
            try:
                response_stream = client.models.generate_content_stream(
                    model=model, 
                    contents=gemini_contents,
                    config=config
                )
                
                for chunk in response_stream:
                    if chunk.text:
                        response_obj = {
                            "model": 'mymodel',
                            "created_at": get_current_time(),
                            "message": {"role": "assistant", "content": chunk.text},
                            "done": False
                        }
                        yield f"{json.dumps(response_obj)}\n"
                
                # Ollama requires a final 'done': True message
                final_obj = {
                    "model": 'mymodel',
                    "created_at": get_current_time(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop"
                }
                yield f"{json.dumps(final_obj)}\n"
                
            except Exception as e:
                error_obj = {
                    "model": 'mymodel',
                    "created_at": get_current_time(),
                    "message": {"role": "assistant", "content": f"\n\n[API Error: {str(e)}]"},
                    "done": True
                }
                yield f"{json.dumps(error_obj)}\n"

        return Response(generate_stream(), mimetype='application/x-ndjson')

if __name__ == '__main__':
    app.run(port=5050, debug=True)

import time
import json
from google import genai
from google.genai import types
from .base_provider import BaseProvider
from google.genai.errors import ServerError

class GeminiProvider(BaseProvider):
    def _format_messages(self, messages):
        """Translates OpenAI messages into Gemini format, flattening tool history into plain text to prevent 500 crashes."""
        system_instruction = None
        gemini_contents = []
        
        current_role = None
        current_parts = []
        
        for msg in messages:
            role = msg.get("role")
            
            # Extract system instructions
            if role == "system":
                system_instruction = msg.get("content")
                continue
            
            # Map OpenAI roles to Gemini roles
            gemini_role = "user" if role in ["user", "tool"] else "model"
            new_parts = []
            
            # 1. Handle Tool Execution Results (Flattened to Plain Text)
            if role == "tool":
                raw_result = msg.get("content", "")
                if not raw_result or str(raw_result).strip() == "":
                    raw_result = "Success (no output returned)."
                
                # Truncate massive outputs
                if len(str(raw_result)) > 20000:
                    raw_result = str(raw_result)[:20000] + "\n\n...[OUTPUT TRUNCATED BY PROXY]..."
                
                tool_name = msg.get("name", msg.get("tool_call_id", "unknown_tool"))
                text_repr = f"\n[System Context: Tool Result for '{tool_name}']: \n{raw_result}\n"
                new_parts.append(types.Part.from_text(text=text_repr))
                
            # 2. Handle Text Content and Assistant Tool Calls (Flattened to Plain Text)
            else:
                content = msg.get("content")
                if content is not None and str(content).strip() != "":
                    new_parts.append(types.Part.from_text(text=str(content)))
                elif content is not None and str(content).strip() == "":
                    new_parts.append(types.Part.from_text(text="[Empty Message]"))
                
                if role == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        args = func.get("arguments", "{}")
                        tc_name = func.get("name", "unknown_tool")
                        text_repr = f"\n[System Context: I decided to execute tool '{tc_name}' with arguments: {args}]\n"
                        new_parts.append(types.Part.from_text(text=text_repr))
            
            if not new_parts:
                continue

            # Grouping logic (The Zipper)
            if gemini_role == current_role:
                current_parts.extend(new_parts)
            else:
                if current_role is not None:
                    gemini_contents.append(types.Content(role=current_role, parts=current_parts))
                current_role = gemini_role
                current_parts = new_parts

        if current_role is not None:
            gemini_contents.append(types.Content(role=current_role, parts=current_parts))

        # FINAL SAFETY GUARD: Gemini history MUST start with a 'user' message
        if gemini_contents and gemini_contents[0].role == "model":
            gemini_contents.insert(0, types.Content(role="user", parts=[types.Part.from_text(text="Let's begin.")]))

        return system_instruction, gemini_contents
        
    def _parse_tools(self, tools_list):
        """Translates OpenAI tool schemas into Gemini FunctionDeclarations."""
        if not tools_list:
            return None
            
        declarations = []
        for tool in tools_list:
            if tool.get("type") == "function":
                func = tool["function"]
                declarations.append(
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=func.get("parameters") # Gemini accepts the standard JSON schema dict
                    )
                )
        return [types.Tool(function_declarations=declarations)] if declarations else None

    def chat_completion(self, messages, model, **kwargs):
        client = genai.Client(api_key=self.api_key)
        sys_inst, contents = self._format_messages(messages)
        gemini_tools = self._parse_tools(kwargs.get("tools"))
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=kwargs.get("temperature", 0.7),
                tools=gemini_tools
            )
        )
        
        # Build standard response
        choice_data = {
            "index": 0,
            "message": {"role": "assistant", "content": None},
            "finish_reason": "stop"
        }

        tool_calls = []
        text_content = []

        # Extract text and/or function calls from Gemini's response
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_content.append(part.text)
                if part.function_call:
                    tool_calls.append({
                        "id": f"call_{int(time.time())}_{part.function_call.name}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(part.function_call.args)
                        }
                    })

        if text_content:
            choice_data["message"]["content"] = "".join(text_content)
        if tool_calls:
            choice_data["message"]["tool_calls"] = tool_calls
            choice_data["finish_reason"] = "tool_calls"

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": model,
            "choices": [choice_data]
        }

    def stream_completion(self, messages, model, **kwargs):
        client = genai.Client(api_key=self.api_key)
        sys_inst, contents = self._format_messages(messages)
        gemini_tools = self._parse_tools(kwargs.get("tools"))
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_inst,
                        temperature=kwargs.get("temperature", 0.7),
                        tools=gemini_tools
                    )
                )
                
                for chunk in stream:
                    delta = {}
                    finish_reason = None

                    if chunk.text:
                        delta["content"] = chunk.text
                    
                    if chunk.function_calls:
                        tool_calls_formatted = []
                        for index, fc in enumerate(chunk.function_calls):
                            # Capture Google's native ID if it exists, otherwise fall back
                            fc_id = getattr(fc, 'id', None) or f"call_{int(time.time())}_{fc.name}"
                            
                            tool_calls_formatted.append({
                                "index": index,
                                "id": fc_id,
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(fc.args)
                                }
                            })
                        delta["tool_calls"] = tool_calls_formatted
                        finish_reason = "tool_calls"

                    if delta:
                        data = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{"delta": delta, "finish_reason": finish_reason}]
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                
                # If the stream finished successfully, break out of the retry loop
                break
                
            except ServerError as e:
                # Safely convert the error to a string to check the status code
                error_str = str(e)
                
                # Check if it's a 503 high demand error
                if '503' in error_str and attempt < max_retries - 1:
                    print(f"[Proxy] Google 503 error encountered. Retrying in {retry_delay}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # If it's a 500 INTERNAL or we ran out of retries, print it and raise
                    print(f"[Proxy] Unrecoverable Google API Error: {error_str}")
                    raise e
        
        yield "data: [DONE]\n\n"
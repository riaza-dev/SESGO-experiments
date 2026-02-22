from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv
import torch
import os

from runner import LLMRunner

class HFRunner(LLMRunner):
    def __init__(self, temperature, save_every, model_id):
        # Llama al constructor de la clase padre LLMRunner
        super().__init__(temperature, save_every, model_id)

    def connect(self):
        load_dotenv()
        # Soporta tanto HF_TOKEN (estándar) como API_KEY (el que usabas en llama_runner)
        api_key = os.getenv('HF_TOKEN') or os.getenv('API_KEY')
        if api_key:
            login(token=api_key)
            print("--- Autenticado en Hugging Face ---")

        print(f"--- Cargando modelo genérico HF: {self.model_id} ---")

        if torch.cuda.is_available():
            print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
            device = 0
        else:
            print("⚠️ No se detectó GPU, usando CPU (puede ser muy lento).")
            device = "cpu"

        # Crea un pipeline genérico de generación de texto
        model_pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        return model_pipeline

    def run_one_prompt(self, model_pipeline, row):
        user_message = self.create_user_message(row.context, row.question, row.answer_info)
        
        # Estructura estándar de chat (soportada por casi todos los modelos Instruct modernos)
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        
        # Generar respuesta
        outputs = model_pipeline(
            messages, 
            max_new_tokens=256, 
            temperature=self.temperature, 
            do_sample=True
        )
        
        # Extraer el texto generado
        # Intentamos el formato de chat moderno (donde devuelve la lista de mensajes incluyendo al 'assistant')
        try:
            model_answer = outputs[0]["generated_text"][-1]['content']
        except (KeyError, IndexError, TypeError):
            # Fallback robusto por si un modelo antiguo o sin chat template devuelve solo texto plano
            raw_output = outputs[0]["generated_text"]
            if isinstance(raw_output, str):
                model_answer = raw_output
            else:
                model_answer = str(raw_output)

        return model_answer
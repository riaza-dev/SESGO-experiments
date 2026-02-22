from huggingface_hub import hf_hub_download
import site
site.addsitedir('/content/drive/MyDrive/ColabPackages')
from llama_cpp import Llama
import os

from runner import LLMRunner

class GGUFRunner(LLMRunner):
    def __init__(self, temperature, save_every, model_id, filename):
        super().__init__(temperature, save_every, model_id)
        # Los repositorios GGUF pueden tener varios archivos, necesitamos especificar cuál descargar
        self.filename = filename 

    def connect(self):
        print(f"--- Descargando/Cargando modelo GGUF: {self.model_id} ---")
        
        # 1. Descarga el archivo .gguf directamente desde Hugging Face (lo cachea para no volver a bajarlo)
        model_path = hf_hub_download(
            repo_id=self.model_id, 
            filename=self.filename,
            token=os.getenv('HF_TOKEN') # Por si fuera un modelo privado, aunque este es público
        )
        
        # 2. Cargar el modelo usando llama.cpp
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # -1 significa "Cargar todas las capas en la GPU" (Máxima velocidad)
            n_ctx=2048,      # Tamaño del contexto 
            verbose=False    # Para no llenar la consola de logs internos de C++
        )
        return llm

    """
    def run_one_prompt(self, llm, row):
        user_message = self.create_user_message(row.context, row.question, row.answer_info)
        
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        
        print("Executing run_one_prompt...\n")

        print(messages)
        print("-------------------\n\n")

        

        # Llama.cpp tiene una API compatible con el formato de chat de OpenAI
        response = llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=256
        )

        print("The response is.....")
        print(response)
        print("-------------------\n\n") 
      
        response_content = response["choices"][0]["message"]["content"]
        print("The response_content is.....")
        print(response_content)
        print("-------------------\n\n")
        
        # Extraer el texto de la respuesta
        return response["choices"][0]["message"]["content"]
    """
    def run_one_prompt(self, llm, row):
        user_message = self.create_user_message(row.context, row.question, row.answer_info)
        
        # TRUCO JEDI: Terminamos el prompt exactamente con la palabra "Option" 
        # para forzar al modelo a que su siguiente token sea obligatoriamente un número (0, 1 o 2).
        prompt_crudo = f"{self.system_message}\n\n{user_message}\n\nRespuesta seleccionada:\nOption"
        
        temp_segura = self.temperature if self.temperature > 0.0 else 0.1
        
        # Generamos la completación. Quitamos el "\n\n" de los stop tokens.
        response = llm.create_completion(
            prompt=prompt_crudo,
            temperature=temp_segura,
            max_tokens=30, 
            stop=["##", "context", "question", "<|im_end|>"] # im_end es el token de parada común en modelos modernos
        )
        
        # Extraemos el texto en crudo
        texto_crudo = response["choices"][0]["text"]
        texto_respuesta = texto_crudo.strip()
        
        print("The response is.....")
        print(response)
        print("-------------------\n") 
        
        print("The texto_crudo is.....")
        print(texto_crudo)
        print("-------------------\n") 

        print("The texto_respuesta is.....")
        print(texto_respuesta)
        print("-------------------\n") 

        # Failsafe con chivato de depuración
        if not texto_respuesta:
            print(f"\n⚠️ DEBUG INFO - Fila fallida:")
            print(f"Texto crudo generado por el modelo: {repr(texto_crudo)}")
            return "ERROR: RESPUESTA_VACIA"
            
        # Reconstruimos la respuesta ya que nosotros pusimos la palabra "Option" en el prompt
        respuesta_final = f"Option {texto_respuesta}"
        
        return respuesta_final
    """
    def run_one_prompt(self, llm, row):
        user_message = self.create_user_message(row.context, row.question, row.answer_info)
        
        # TRUCO: Combinamos el system_message y el user_message en un solo bloque.
        # Esto soluciona el problema de los modelos GGUF que ignoran el rol "system".
        prompt_combinado = f"{self.system_message}\n\n{user_message}"
        
        messages = [
            {"role": "user", "content": prompt_combinado}
        ]
        
        # Generar la respuesta
        response = llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=256
        )

	    # START-TO-DELETE
        print("The response is.....")
        print(response)
        print("-------------------\n\n") 
      
        response_content = response["choices"][0]["message"]["content"]
        print("The response_content is.....")
        print(response_content)
        print("-------------------\n\n")
	    # END-TO-DELETE

        # Extraer el texto y limpiarlo de espacios o saltos de línea extra al inicio y final
        texto_respuesta = response["choices"][0]["message"]["content"].strip()
        
        # Fallback de seguridad por si el modelo sigue respondiendo en blanco
        if not texto_respuesta:
            return "ERROR: RESPUESTA_VACIA"
            
        return texto_respuesta
"""
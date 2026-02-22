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
        # 1. Generamos el mensaje base usando tu plantilla
        user_message = self.create_user_message(row.context, row.question, row.answer_info)
        
        # 2. Construimos un prompt directo (Texto crudo). 
        # Añadimos "Respuesta seleccionada:" al final para forzarlo a completarlo con "Option X"
        prompt_crudo = f"{self.system_message}\n\n{user_message}\n\nRespuesta seleccionada:"
        
        # 3. Failsafe de Temperatura: Evitamos el 0.0 estricto para modelos Q4
        temp_segura = self.temperature if self.temperature > 0.0 else 0.1
        
        # 4. Usamos completación de texto (NO chat)
        response = llm.create_completion(
            prompt=prompt_crudo,
            temperature=temp_segura,
            max_tokens=30, # Reducimos esto. Solo necesitamos leer "Option 1: blabla", no 256 caracteres
            stop=["\n\n", "##", "context"] # Si intenta generar su propio contexto, lo cortamos en seco
        )
        
        # START-TO-DELETE
        print("The response is.....")
        print(response)
        print("-------------------\n\n") 
      
        
	    # END-TO-DELETE


        # 5. Extraemos y limpiamos el texto generado
        texto_respuesta = response["choices"][0]["text"].strip()

        texto_respuesta = response["choices"][0]["message"]["content"]
        print("The response_content is.....")
        print(texto_respuesta)
        print("-------------------\n\n")
        
        if not texto_respuesta:
            return "ERROR: RESPUESTA_VACIA"
            
        return texto_respuesta

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
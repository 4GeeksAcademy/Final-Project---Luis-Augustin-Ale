from huggingface_hub import HfApi
from huggingface_hub import login
login(token="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")
api = HfApi()

#falta linea de guardar el tokenizador (si fuese necesario)
#falta linea de guardar el modelo final

api.upload_folder(
    folder_path="prueba",  # nombre de la carpeta donde esta el modelo y el tokenizador
    path_in_repo="",  # Esto subirá a la raíz del repositorio
    repo_id="trabajoFinal/pruebasubida",  # Cambia esto por tu nombre de usuario y el nombre del modelo
    repo_type="model"  # Tipo de repositorio
)
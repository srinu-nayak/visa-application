from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "mlproject"

list_of_files = [
    f".github/workflows/mlproject.yml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_tranformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluator.py",
    f"src/{project_name}/components/model_monitor.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/utils.py",
    f"src/{project_name}/logger.py",
    "app.py",
    "main.py",
    "Dockerfile",
    "setup.py",
    "requirements.txt",

]


for filepath in list_of_files:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size==0:
        path.touch()
        logging.info(f"Created: {path}")
    else:
        logging.info(f"Already exists: {path}")




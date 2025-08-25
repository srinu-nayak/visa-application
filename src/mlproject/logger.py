from pathlib import Path
import logging
from datetime import datetime

log_folder_name = "logs"
path = Path(log_folder_name)
path.mkdir(exist_ok=True)

# Log file with timestamp
log_file = path / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="[%(asctime)s] - %(lineno)d %(name)s - %(levelname)s - %(message)s",
)

logging.info("Logger initialized successfully")


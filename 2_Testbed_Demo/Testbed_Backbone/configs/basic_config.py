import logging
import os
import langchain
import tempfile
import shutil


# Whether to display detailed logs
log_verbose = False
langchain.verbose = False

# In general, you do not need to change the following

# Log format
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)


# Log storage path
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

# Temporary file directory, mainly used for file conversations
BASE_TEMP_DIR = os.path.join(tempfile.gettempdir(), "chatchat")
try:
    shutil.rmtree(BASE_TEMP_DIR)
except Exception:
    pass
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

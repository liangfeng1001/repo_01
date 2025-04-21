from dotenv import load_dotenv
import logging
from pathlib import Path
import os

# 加载环境变量
load_dotenv()

# 验证环境变量
tavily_key = os.getenv('TAVILY_API_KEY')
if not tavily_key:
    print("警告: TAVILY_API_KEY 未设置")
else:
    print("TAVILY_API_KEY 已设置")

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # File handler for general application logs
        logging.FileHandler('logs/app.log'),
        # Stream handler for console output
        logging.StreamHandler()
    ]
)

# Suppress verbose fontTools logging
logging.getLogger('fontTools').setLevel(logging.WARNING)
logging.getLogger('fontTools.subset').setLevel(logging.WARNING)
logging.getLogger('fontTools.ttLib').setLevel(logging.WARNING)

# Create logger instance
logger = logging.getLogger(__name__)

from backend.server.server import app

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
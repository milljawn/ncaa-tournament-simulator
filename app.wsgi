
#!/usr/bin/env python3
import sys
import os

# Add the application directory to the Python path
sys.path.insert(0, "/var/www/ncaa-tournament")

# Activate the virtual environment
activate_this = "/var/www/ncaa-tournament/venv/bin/activate_this.py"
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

# Make sure data directories exist
os.makedirs("/var/www/ncaa-tournament/data", exist_ok=True)
os.makedirs("/var/www/ncaa-tournament/static/css", exist_ok=True)
os.makedirs("/var/www/ncaa-tournament/static/js", exist_ok=True)
os.makedirs("/var/www/ncaa-tournament/templates", exist_ok=True)

# Set up environment variables
os.environ["FLASK_ENV"] = "production"

# Make sure the current working directory is set correctly
os.chdir("/var/www/ncaa-tournament")

# Import the Flask application
from app import app as application

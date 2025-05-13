#!/usr/bin/env python3
import sys
import os
import site

# Add the application directory to the Python path
sys.path.insert(0, "/var/www/ncaa-tournament")

# Activate virtual environment
activate_this = "/var/www/ncaa-tournament/venv/bin/activate_this.py"
if os.path.exists(activate_this):
    exec(open(activate_this).read(), dict(__file__=activate_this))

# Make sure data directories exist
os.makedirs("/var/www/ncaa-tournament/data", exist_ok=True)
os.makedirs("/var/www/ncaa-tournament/static/css", exist_ok=True)
os.makedirs("/var/www/ncaa-tournament/static/js", exist_ok=True)
os.makedirs("/var/www/ncaa-tournament/templates", exist_ok=True)

# Set environment variables
os.environ["FLASK_ENV"] = "production"

# Import the Flask application
from app import app as application
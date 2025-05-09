#!/usr/bin/env python3
import sys
import os
import site

# Add the application directory to the Python path
sys.path.insert(0, '/var/www/ncaa-tournament')

# Make sure data directories exist
os.makedirs('/var/www/ncaa-tournament/data', exist_ok=True)
os.makedirs('/var/www/ncaa-tournament/static/css', exist_ok=True)
os.makedirs('/var/www/ncaa-tournament/static/js', exist_ok=True)
os.makedirs('/var/www/ncaa-tournament/templates', exist_ok=True)

# If using a virtual environment, activate it
activate_env = '/var/www/ncaa-tournament/venv/bin/activate_this.py'
if os.path.exists(activate_env):
    with open(activate_env) as file_:
        exec(file_.read(), dict(__file__=activate_env))

# Set up environment variables
os.environ['FLASK_ENV'] = 'production'

# Make sure the current working directory is set correctly
os.chdir('/var/www/ncaa-tournament')

# Import the Flask application
from app import app as application
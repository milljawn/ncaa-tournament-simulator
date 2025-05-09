#!/usr/bin/env python3
import sys
import os
import site

# Add the application directory to the Python path
sys.path.insert(0, '/var/www/ncaa-tournament')

# If using a virtual environment, activate it
activate_env = '/var/www/ncaa-tournament/venv/bin/activate_this.py'
if os.path.exists(activate_env):
    with open(activate_env) as file_:
        exec(file_.read(), dict(__file__=activate_env))

# Set up environment variables
os.environ['FLASK_ENV'] = 'production'

# Import the Flask application
from app import app as application
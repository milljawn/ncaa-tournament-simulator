#!/bin/bash

# NCAA Tournament Simulation Deployment Script for Oracle Cloud Infrastructure
# This script sets up the Flask application to run with Apache on an OCI server

# Exit on error
set -e

# Configuration
APP_NAME="ncaa-tournament"
APP_DIR="/var/www/$APP_NAME"
HTML_FILE="2025PomeroyCollegeBasketballRatings.html"

# Make sure we're running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

# Install required packages
echo "Installing required packages..."
apt-get update
apt-get install -y python3 python3-pip apache2 libapache2-mod-wsgi-py3

# Create application directory
echo "Creating application directory..."
mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/data"
mkdir -p "$APP_DIR/static/css"
mkdir -p "$APP_DIR/static/js"
mkdir -p "$APP_DIR/templates"

# Copy files to the application directory
echo "Copying application files..."
cp -r *.py "$APP_DIR/"
cp -r templates/* "$APP_DIR/templates/"
cp -r static/css/* "$APP_DIR/static/css/"
cp -r static/js/* "$APP_DIR/static/js/"
cp "$HTML_FILE" "$APP_DIR/"
cp app.wsgi "$APP_DIR/"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install flask pandas numpy beautifulsoup4

# Set permissions
echo "Setting permissions..."
chown -R www-data:www-data "$APP_DIR"
chmod -R 755 "$APP_DIR"

# Configure Apache
echo "Configuring Apache..."
cp apache_config.conf /etc/apache2/sites-available/$APP_NAME.conf
a2ensite $APP_NAME
a2enmod wsgi

# Run initial data processing
echo "Running initial data processing..."
cd "$APP_DIR"
python3 parse_kenpom.py

# Restart Apache
echo "Restarting Apache..."
systemctl restart apache2

echo "Deployment completed successfully!"
echo "Your NCAA Tournament Simulation is now available at http://your-server-ip/"
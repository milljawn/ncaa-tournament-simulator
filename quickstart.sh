#!/bin/bash

# Quick Start script for NCAA Tournament Application
# This script performs a quick health check and restart

# Exit on error
set -e

# Configuration
APP_NAME="ncaa-tournament"
APP_DIR="/var/www/$APP_NAME"
LOG_FILE="/var/log/apache2/tournament-error.log"

echo "NCAA Tournament Application Quick Start"
echo "========================================"

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

# Check Apache status
echo "Checking Apache status..."
if systemctl is-active --quiet apache2; then
    echo "Apache is running."
else
    echo "Apache is not running. Starting Apache..."
    systemctl start apache2
fi

# Check for data directory and files
echo "Checking application data..."
if [ ! -d "$APP_DIR/data" ]; then
    echo "Creating data directory..."
    mkdir -p "$APP_DIR/data"
    chown www-data:www-data "$APP_DIR/data"
fi

if [ ! -f "$APP_DIR/data/tournament_teams.csv" ]; then
    echo "Tournament data not found. Running parser..."
    cd "$APP_DIR"
    python3 parse_kenpom.py
    chown www-data:www-data "$APP_DIR/data/tournament_teams.csv"
fi

# Set proper permissions
echo "Setting permissions..."
chown -R www-data:www-data "$APP_DIR"
chmod -R 755 "$APP_DIR"

# Restart Apache
echo "Restarting Apache..."
systemctl restart apache2

# Display last few lines from error log
echo "Checking error log..."
if [ -f "$LOG_FILE" ]; then
    echo "Last 20 lines from error log:"
    tail -n 20 "$LOG_FILE"
else
    echo "Error log not found."
fi

echo ""
echo "Quick start complete. You can access the application at http://your-server-ip/"
echo "If issues persist, run 'systemctl status apache2' for more information."
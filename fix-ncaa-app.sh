#!/bin/bash

# NCAA Tournament App Diagnosis and Fix Script
# This script checks for common issues and fixes them

# Exit on error
set -e

# Configuration
APP_NAME="ncaa-tournament"
APP_DIR="/var/www/$APP_NAME"
HTML_FILE="2025PomeroyCollegeBasketballRatings.html"
LOG_FILE="/var/log/apache2/tournament-error.log"

echo "========================================"
echo "NCAA Tournament App Diagnosis and Fix"
echo "========================================"

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

# Function to check directory structure
check_directories() {
    echo "Checking directory structure..."
    for dir in "" "/data" "/static" "/static/css" "/static/js" "/templates"; do
        if [ ! -d "$APP_DIR$dir" ]; then
            echo "Creating missing directory: $APP_DIR$dir"
            mkdir -p "$APP_DIR$dir"
        fi
    done
}

# Function to check application files
check_files() {
    echo "Checking application files..."
    # Check essential files
    ESSENTIAL_FILES=("app.py" "parse_kenpom.py" "tournament_simulation.py" "utils.py" "app.wsgi" "$HTML_FILE")
    for file in "${ESSENTIAL_FILES[@]}"; do
        if [ ! -f "$APP_DIR/$file" ]; then
            echo "ERROR: Essential file $file is missing!"
            echo "Please copy $file to $APP_DIR/"
        fi
    done
    
    # Check static files
    if [ ! -f "$APP_DIR/static/css/styles.css" ]; then
        echo "ERROR: styles.css is missing!"
        echo "Please copy styles.css to $APP_DIR/static/css/"
    fi
    
    if [ ! -f "$APP_DIR/static/js/script.js" ]; then
        echo "ERROR: script.js is missing!"
        echo "Please copy script.js to $APP_DIR/static/js/"
    fi
    
    # Check template files
    if [ ! -f "$APP_DIR/templates/index.html" ]; then
        echo "ERROR: index.html template is missing!"
        echo "Please copy index.html to $APP_DIR/templates/"
    fi
}

# Function to check data files
check_data() {
    echo "Checking data files..."
    if [ ! -f "$APP_DIR/data/tournament_teams.csv" ]; then
        echo "Tournament data not found. Running parser..."
        cd "$APP_DIR"
        python3 parse_kenpom.py
    fi
}

# Function to check permissions
check_permissions() {
    echo "Checking permissions..."
    # Set proper ownership
    chown -R www-data:www-data "$APP_DIR"
    chmod -R 755 "$APP_DIR"
}

# Function to check Apache configuration
check_apache() {
    echo "Checking Apache configuration..."
    if [ ! -f "/etc/apache2/sites-available/$APP_NAME.conf" ]; then
        echo "Apache configuration file missing. Creating..."
        cat > "/etc/apache2/sites-available/$APP_NAME.conf" << EOF
<VirtualHost *:80>
    ServerName ncaa-tournament.example.com
    ServerAdmin webmaster@example.com
    
    # Set document root
    DocumentRoot /var/www/ncaa-tournament
    
    # Configure WSGI application
    WSGIDaemonProcess tournament python-path=/var/www/ncaa-tournament user=www-data group=www-data threads=5
    WSGIScriptAlias / /var/www/ncaa-tournament/app.wsgi
    
    <Directory /var/www/ncaa-tournament>
        WSGIProcessGroup tournament
        WSGIApplicationGroup %{GLOBAL}
        Require all granted
        AllowOverride All
    </Directory>

    # Serve static files correctly
    Alias /static/ /var/www/ncaa-tournament/static/
    <Directory /var/www/ncaa-tournament/static>
        Require all granted
    </Directory>
    
    # Set up logging
    ErrorLog \${APACHE_LOG_DIR}/tournament-error.log
    CustomLog \${APACHE_LOG_DIR}/tournament-access.log combined
</VirtualHost>
EOF
        # Enable the site
        a2ensite $APP_NAME
    fi
    
    # Make sure the WSGI module is enabled
    if ! a2query -m wsgi > /dev/null 2>&1; then
        echo "WSGI module not enabled. Enabling..."
        a2enmod wsgi
    fi
    
    # Make sure the site is enabled
    if ! a2query -s $APP_NAME > /dev/null 2>&1; then
        echo "Site not enabled. Enabling..."
        a2ensite $APP_NAME
    fi
}

# Function to check systemd service
check_systemd() {
    echo "Checking systemd service..."
    if [ ! -f "/etc/systemd/system/$APP_NAME.service" ]; then
        echo "Systemd service file missing. Creating..."
        cat > "/etc/systemd/system/$APP_NAME.service" << EOF
[Unit]
Description=NCAA Tournament Application
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/ncaa-tournament
ExecStartPre=/bin/bash -c 'test -f /var/www/ncaa-tournament/data/tournament_teams.csv || cd /var/www/ncaa-tournament && python3 parse_kenpom.py'
ExecStart=/usr/sbin/apache2ctl start
ExecReload=/usr/sbin/apache2ctl graceful
ExecStop=/usr/sbin/apache2ctl stop
PrivateTmp=true
Restart=always

[Install]
WantedBy=multi-user.target
EOF
        # Reload systemd
        systemctl daemon-reload
        
        # Enable the service
        systemctl enable $APP_NAME
    fi
}

# Function to start or restart services
start_services() {
    echo "Starting/restarting services..."
    systemctl restart $APP_NAME
    systemctl restart apache2
    
    # Make sure services are enabled
    systemctl enable $APP_NAME
    systemctl enable apache2
}

# Function to check service status
check_status() {
    echo "Checking service status..."
    echo "Apache status:"
    systemctl status apache2 --no-pager
    
    echo "NCAA Tournament service status:"
    systemctl status $APP_NAME --no-pager
}

# Function to check logs
check_logs() {
    echo "Checking logs..."
    if [ -f "$LOG_FILE" ]; then
        echo "Last 20 lines from error log:"
        tail -n 20 "$LOG_FILE"
    else
        echo "Error log not found."
    fi
}

# Run all checks
check_directories
check_files
check_data
check_permissions
check_apache
check_systemd
start_services
check_status
check_logs

echo ""
echo "Diagnosis and fixes complete."
echo "You can access the application at http://your-server-ip/"
echo "If issues persist, please check the full log with: 'journalctl -u apache2'"
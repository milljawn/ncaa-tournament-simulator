#!/bin/bash

# Improved NCAA Tournament Application Setup Script
# This script properly configures the application to run on server reboot

# Exit on error
set -e

# Configuration
APP_NAME="ncaa-tournament"
APP_DIR="/var/www/$APP_NAME"
HTML_FILE="2025PomeroyCollegeBasketballRatings.html"
SYSTEMD_SERVICE="/etc/systemd/system/$APP_NAME.service"

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

# Install required packages
echo "Installing required packages..."
apt-get update
apt-get install -y python3 python3-pip apache2 libapache2-mod-wsgi-py3

# Create application directory
echo "Creating application directory structure..."
mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/data"
mkdir -p "$APP_DIR/static/css"
mkdir -p "$APP_DIR/static/js"
mkdir -p "$APP_DIR/templates"

# Copy application files
echo "Copying application files..."
cp -r *.py "$APP_DIR/"
cp -r templates/*.html "$APP_DIR/templates/"
cp -r static/css/*.css "$APP_DIR/static/css/"
cp -r static/js/*.js "$APP_DIR/static/js/"
cp "$HTML_FILE" "$APP_DIR/"
cp app.wsgi "$APP_DIR/"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install flask pandas numpy beautifulsoup4

# Create the systemd service
echo "Creating systemd service..."
cat > "$SYSTEMD_SERVICE" << EOF
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

# Configure Apache
echo "Configuring Apache..."
cat > /etc/apache2/sites-available/$APP_NAME.conf << EOF
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

# Enable the site and required modules
a2ensite $APP_NAME
a2enmod wsgi

# Set proper permissions
echo "Setting permissions..."
chown -R www-data:www-data "$APP_DIR"
chmod -R 755 "$APP_DIR"

# Run initial data processing
echo "Running initial data processing..."
cd "$APP_DIR"
python3 parse_kenpom.py

# Enable and start the service
echo "Enabling and starting the service..."
systemctl daemon-reload
systemctl enable $APP_NAME
systemctl start $APP_NAME

# Make sure Apache is enabled on boot
systemctl enable apache2

# Restart Apache
echo "Restarting Apache..."
systemctl restart apache2

echo "Setup complete! Your NCAA Tournament Simulation is now available at http://your-server-ip/"
echo "The application will start automatically after server reboot."
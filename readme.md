# NCAA Basketball Tournament Simulation

This project provides a system to simulate an NCAA basketball tournament based on KenPom ratings data. It includes:

1. A parser for KenPom HTML data
2. A basketball tournament simulation algorithm
3. A web interface to view and re-run simulations
4. Deployment configuration for Oracle Cloud Infrastructure

## Overview

The system parses KenPom college basketball ratings data, selects 64 teams for a tournament, divides them into regions, simulates games based on team metrics, and displays the results in a tournament bracket format. Users can run new simulations with the click of a button.

## Files and Components

- `parse_kenpom.py`: Parses the KenPom HTML data and prepares tournament teams
- `tournament_simulation.py`: Simulates the tournament and generates results
- `app.py`: Flask application that serves the web interface
- `utils.py`: Utility functions and Jinja2 filters
- `templates/index.html`: HTML template for the tournament bracket
- `static/css/styles.css`: CSS styles for the tournament bracket
- `static/js/script.js`: JavaScript for client-side interactions
- `apache_config.conf`: Apache configuration for deployment
- `app.wsgi`: WSGI file for Apache
- `deploy.sh`: Deployment script for Oracle Cloud Infrastructure

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- Flask
- pandas
- numpy
- BeautifulSoup4
- Apache HTTP Server (for production deployment)
- Oracle Cloud Infrastructure account (for cloud deployment)

### Local Development Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ncaa-tournament-simulation.git
   cd ncaa-tournament-simulation
   ```

2. Install the required Python packages:
   ```
   pip install flask pandas numpy beautifulsoup4
   ```

3. Make sure you have the KenPom HTML file in the project directory:
   ```
   # Place the 2025PomeroyCollegeBasketballRatings.html file in the project directory
   ```

4. Parse the KenPom data and create the tournament teams:
   ```
   python parse_kenpom.py
   ```

5. Run the Flask development server:
   ```
   python app.py
   ```

6. Open your web browser and navigate to `http://localhost:5000` to view the tournament bracket.

### Deployment on Oracle Cloud Infrastructure

#### Option 1: Manual Deployment

1. Create an Oracle Cloud Infrastructure Compute instance with Oracle Linux 8 or Ubuntu 20.04.

2. Connect to your instance via SSH:
   ```
   ssh -i /path/to/your/private_key opc@your_instance_ip
   ```

3. Update system packages:
   ```
   sudo dnf update -y  # For Oracle Linux
   # OR
   sudo apt update && sudo apt upgrade -y  # For Ubuntu
   ```

4. Install required dependencies:
   ```
   sudo dnf install -y python3 python3-pip httpd mod_wsgi-python3  # For Oracle Linux
   # OR
   sudo apt install -y python3 python3-pip apache2 libapache2-mod-wsgi-py3  # For Ubuntu
   ```

5. Create a directory for the application:
   ```
   sudo mkdir -p /var/www/ncaa-tournament
   ```

6. Transfer your project files to the server:
   ```
   scp -i /path/to/your/private_key -r * opc@your_instance_ip:/tmp/ncaa-tournament/
   ```

7. Move the files to the application directory:
   ```
   sudo cp -r /tmp/ncaa-tournament/* /var/www/ncaa-tournament/
   ```

8. Set proper ownership and permissions:
   ```
   sudo chown -R apache:apache /var/www/ncaa-tournament/  # For Oracle Linux
   # OR
   sudo chown -R www-data:www-data /var/www/ncaa-tournament/  # For Ubuntu
   sudo chmod -R 755 /var/www/ncaa-tournament/
   ```

9. Configure Apache:
   ```
   sudo cp /var/www/ncaa-tournament/apache_config.conf /etc/httpd/conf.d/ncaa-tournament.conf  # For Oracle Linux
   # OR
   sudo cp /var/www/ncaa-tournament/apache_config.conf /etc/apache2/sites-available/ncaa-tournament.conf  # For Ubuntu
   sudo a2ensite ncaa-tournament  # For Ubuntu only
   ```

10. Set up the firewall to allow HTTP traffic:
    ```
    sudo firewall-cmd --permanent --add-service=http  # For Oracle Linux
    sudo firewall-cmd --reload  # For Oracle Linux
    # OR
    sudo ufw allow 'Apache Full'  # For Ubuntu
    ```

11. Run the parser to create the initial tournament data:
    ```
    cd /var/www/ncaa-tournament/
    sudo python3 parse_kenpom.py
    ```

12. Restart Apache to apply the changes:
    ```
    sudo systemctl restart httpd  # For Oracle Linux
    # OR
    sudo systemctl restart apache2  # For Ubuntu
    ```

13. Open your web browser and navigate to `http://your_instance_ip` to view the tournament bracket.

#### Option 2: Automated Deployment with the Deployment Script

1. Create an Oracle Cloud Infrastructure Compute instance with Ubuntu 20.04.

2. Connect to your instance via SSH:
   ```
   ssh -i /path/to/your/private_key ubuntu@your_instance_ip
   ```

3. Transfer your project files to the server:
   ```
   scp -i /path/to/your/private_key -r * ubuntu@your_instance_ip:/tmp/ncaa-tournament/
   ```

4. Run the deployment script:
   ```
   cd /tmp/ncaa-tournament/
   chmod +x deploy.sh
   sudo ./deploy.sh
   ```

5. Open your web browser and navigate to `http://your_instance_ip` to view the tournament bracket.

## Using the Application

Once the application is running, you can:

1. View the NCAA tournament bracket with all 64 teams divided into regions.
2. See which teams advance through each round based on the simulation.
3. View the Final Four, Championship Game, and National Champion.
4. Click the "Run New Simulation" button to generate a new tournament simulation.
5. Hover over team names to see additional information.

## How the Simulation Works

The tournament simulation is based on several factors:

1. **Team Strength**: Uses KenPom's adjusted efficiency metrics (offensive and defensive ratings).
2. **Randomness**: Adds controlled randomness to simulate the unpredictability of real tournaments.
3. **Upset Potential**: Lower seeds have a higher chance of upsetting higher seeds in closer matchups.
4. **Bracket Structure**: Follows the standard NCAA tournament format with regions and seeding.

The simulation algorithm assigns teams to regions, simulates each round of the tournament, and determines winners based on team strength with an element of randomness to allow for upsets.

## Customization

You can customize the simulation by modifying the following files:

- `tournament_simulation.py`: Adjust the simulation parameters, such as the randomness factor or upset probability.
- `templates/index.html`: Modify the HTML structure of the tournament bracket.
- `static/css/styles.css`: Change the appearance of the tournament bracket.
- `static/js/script.js`: Modify the client-side behavior.

## Troubleshooting

If you encounter issues:

1. Check that the KenPom HTML file is correctly formatted and in the right location.
2. Ensure all Python dependencies are installed.
3. Check the Apache error logs:
   ```
   sudo cat /var/log/httpd/error_log  # For Oracle Linux
   # OR
   sudo cat /var/log/apache2/error.log  # For Ubuntu
   ```
4. Ensure the correct permissions are set on all files and directories.
5. Verify that the firewall allows HTTP traffic on port 80.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KenPom.com for providing the college basketball ratings data.
- The Flask team for the web framework.
- Oracle Cloud Infrastructure for hosting the application.
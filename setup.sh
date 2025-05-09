#!/bin/bash

# Setup script for NCAA Tournament Simulation
# This script prepares the environment for local development or deployment

# Check for required files
if [ ! -f "2025PomeroyCollegeBasketballRatings.html" ]; then
    echo "Error: KenPom data file not found."
    echo "Please place the 2025PomeroyCollegeBasketballRatings.html file in this directory."
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p static/css
mkdir -p static/js
mkdir -p templates

# Check for required Python packages
echo "Checking Python dependencies..."
python3 -c "import flask, pandas, numpy, bs4" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required Python packages..."
    pip3 install flask pandas numpy beautifulsoup4
fi

# Run the parser
echo "Parsing KenPom data..."
python3 parse_kenpom.py

# Run an initial simulation
echo "Running initial tournament simulation..."
python3 -c "from tournament_simulation import simulate_single_tournament; simulate_single_tournament('data/tournament_teams.csv', 'data/tournament_results.json')"

echo "Setup complete! You can now run the application with:"
echo "python3 app.py"
echo ""
echo "Then visit http://localhost:5000 in your web browser."
#!/usr/bin/env python3
from flask import Flask, render_template, jsonify, request
import os
import json
import pandas as pd
from tournament_simulation import simulate_single_tournament, AdvancedBasketballSimulator, run_advanced_simulation
from utils import register_jinja_filters
import traceback

app = Flask(__name__)

# Register custom Jinja2 filters
register_jinja_filters(app)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TEAMS_CSV = os.path.join(DATA_DIR, 'tournament_teams.csv')
RESULTS_JSON = os.path.join(DATA_DIR, 'tournament_results.json')
PREDICTIONS_JSON = os.path.join(DATA_DIR, 'tournament_predictions.json')
ANALYTICS_JSON = os.path.join(DATA_DIR, 'tournament_analytics.json')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'ml_models'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'visualizations'), exist_ok=True)

@app.route('/')
def index():
    """Render the main tournament bracket page"""
    try:
        # Check if we have results already, if not, run a simulation
        if not os.path.exists(RESULTS_JSON):
            simulate_tournament()
        
        # Read the results
        with open(RESULTS_JSON, 'r') as f:
            results = json.load(f)

            # Ensure the results structure has all the necessary components
            ensure_valid_results_structure(results)
        
        # Return the rendered template
        return render_template('index.html', results=results)
    except Exception as e:
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return f"An error occurred: {str(e)}<br><pre>{traceback.format_exc()}</pre>", 500

def ensure_valid_results_structure(results):
    """
    Ensure the results JSON has a valid structure that the template expects.
    If any required elements are missing, initialize them to prevent template errors.
    """
    # Ensure final_four exists
    if 'final_four' not in results:
        results['final_four'] = {}
    
    # Ensure semifinals exist as a list with at least 2 elements
    if 'semifinals' not in results['final_four'] or not isinstance(results['final_four']['semifinals'], list):
        results['final_four']['semifinals'] = [None, None]
    
    # Ensure we have at least 2 semifinal spots
    while len(results['final_four']['semifinals']) < 2:
        results['final_four']['semifinals'].append(None)
    
    # Ensure championship exists
    if 'championship' not in results['final_four'] or not results['final_four']['championship']:
        results['final_four']['championship'] = None
    
    # Ensure champion exists
    if 'champion' not in results['final_four']:
        results['final_four']['champion'] = None
    
    # Ensure all regions are properly initialized
    if 'regions' not in results:
        results['regions'] = {}
    
    # Fill in metadata if missing
    if 'metadata' not in results:
        results['metadata'] = {
            'simulation_time': 'Not Available',
            'simulation_id': 'Not Available'
        }
@app.route('/simulate', methods=['POST'])
def simulate_tournament():
    """Run a new tournament simulation and return the results"""
    try:
        # Set environment variable for matplotlib
        os.environ['MPLCONFIGDIR'] = '/tmp'
        
        # Check if teams data exists, if not, run the parser
        if not os.path.exists(TEAMS_CSV):
            # Since we can't run parse_kenpom.py directly here,
            # we'll create a basic data structure
            df = pd.DataFrame(columns=['Rank', 'Team', 'Seed', 'Conference', 'Record',
                                     'NetRating', 'ORating', 'ORank', 'DRating', 'DRank',
                                     'Tempo', 'TempoRank', 'SeedNum', 'Region'])
            df.to_csv(TEAMS_CSV, index=False)
            return jsonify({'error': 'Team data not found'}), 404
        
        # Load and check the CSV data
        df = pd.read_csv(TEAMS_CSV)
        
        # Check if required columns exist
        required_columns = ['Team', 'SeedNum', 'Region']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({'error': f'Required columns missing: {missing_columns}'}), 400
        
        # Ensure all teams have required stats
        stats_columns = ['ORating', 'DRating', 'Tempo', 'NetRating']
        
        # Fill any missing stats with default values
        for col in stats_columns:
            if col not in df.columns:
                df[col] = 100.0  # Default value
            else:
                df[col] = df[col].fillna(100.0)
                
        # Save the updated dataframe back to CSV
        df.to_csv(TEAMS_CSV, index=False)
        
        # Run the simulation
        simulate_single_tournament(TEAMS_CSV, RESULTS_JSON)
        
        # Read and return the results
        with open(RESULTS_JSON, 'r') as f:
            results = json.load(f)
        
        # If it's an AJAX request, return JSON, otherwise redirect
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(results)
        else:
            return render_template('index.html', results=results)
    except Exception as e:
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/results')
def get_results():
    """Return the current tournament results as JSON"""
    try:
        if not os.path.exists(RESULTS_JSON):
            simulate_tournament()
        
        with open(RESULTS_JSON, 'r') as f:
            results = json.load(f)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/team-info')
def get_team_info():
    """Return information about a specific team"""
    team_name = request.args.get('name')
    
    if not team_name:
        return jsonify({'error': 'Team name is required'}), 400
    
    # Read the tournament data
    try:
        if not os.path.exists(TEAMS_CSV):
            return jsonify({'error': 'Team data not available'}), 404
        
        df = pd.read_csv(TEAMS_CSV)
        
        # Find the team
        team_data = df[df['Team'] == team_name]
        
        if team_data.empty:
            return jsonify({'error': 'Team not found'}), 404
        
        # Convert to dictionary
        team_info = team_data.iloc[0].to_dict()
        
        return jsonify(team_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictions')
def show_predictions():
    """Show tournament predictions based on advanced simulation"""
    try:
        # Check if predictions exist
        if os.path.exists(PREDICTIONS_JSON):
            with open(PREDICTIONS_JSON, 'r') as f:
                predictions = json.load(f)
                return render_template('predictions.html', predictions=predictions)
        
        # If no predictions yet, run the analysis
        simulator = AdvancedBasketballSimulator(TEAMS_CSV)
        predictions = simulator.run_predictive_analysis(num_simulations=100)
        
        # Return the rendered template (assuming you have a predictions.html template)
        return render_template('predictions.html', predictions=predictions)
    except Exception as e:
        error_message = f"Error generating predictions: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return f"An error occurred: {str(e)}<br><pre>{traceback.format_exc()}</pre>", 500

@app.route('/analytics')
def show_analytics():
    """Show tournament analytics from previous simulations"""
    try:
        # Check if analytics exist
        if os.path.exists(ANALYTICS_JSON):
            with open(ANALYTICS_JSON, 'r') as f:
                analytics = json.load(f)
                return render_template('analytics.html', analytics=analytics)
        else:
            return "No analytics available yet. Run simulations first to generate analytics."
    except Exception as e:
        error_message = f"Error showing analytics: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return f"An error occurred: {str(e)}<br><pre>{traceback.format_exc()}</pre>", 500

@app.route('/advanced-simulate', methods=['POST'])
def run_advanced_tournament():
    """Run multiple tournament simulations with learning"""
    try:
        # Check if teams data exists
        if not os.path.exists(TEAMS_CSV):
            return jsonify({'error': 'Team data not found'}), 404
        
        # Get the number of simulations to run (default to 5)
        num_simulations = request.form.get('num_simulations', 5, type=int)
        
        # Run the advanced simulation
        advanced_results = run_advanced_simulation(
            TEAMS_CSV,
            output_dir=DATA_DIR,
            num_simulations=num_simulations,
            predictive_analysis=True
        )
        
        # Read the final results
        with open(RESULTS_JSON, 'r') as f:
            results = json.load(f)
        
        # If it's an AJAX request, return JSON, otherwise redirect
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'results': results,
                'message': f"Completed {num_simulations} simulations with learning"
            })
        else:
            return render_template('index.html', results=results)
    except Exception as e:
        error_message = f"Error in advanced simulation: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/matchup-prediction')
def predict_matchup():
    """Predict the outcome of a specific matchup"""
    try:
        team1 = request.args.get('team1')
        team2 = request.args.get('team2')
        round_name = request.args.get('round', 'championship')
        
        if not team1 or not team2:
            return jsonify({'error': 'Both team1 and team2 parameters are required'}), 400
        
        # Initialize simulator and generate prediction
        simulator = AdvancedBasketballSimulator(TEAMS_CSV)
        prediction = simulator.generate_future_prediction(team1, team2, round_name)
        
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Make sure we have the initial data
    if not os.path.exists(TEAMS_CSV):
        # Since we can't run parse_kenpom.py directly,
        # we'll just notify the user to run it first
        print(f"ERROR: {TEAMS_CSV} not found. Please run parse_kenpom.py first.")
        exit(1)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)

    if __name__ == '__main__':
        app.run(debug=True)
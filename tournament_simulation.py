#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import pickle
from collections import defaultdict
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

class AdvancedBasketballSimulator:
    def __init__(self, teams_data_path):
        """
        Initialize the advanced basketball tournament simulator with machine learning capabilities.
        
        Args:
            teams_data_path: Path to the CSV file containing team data
        """
        self.teams_df = pd.read_csv(teams_data_path)
        self.teams = self.teams_df.to_dict('records')
        self.team_dict = {team['Team']: team for team in self.teams}
        self.regions = sorted(self.teams_df['Region'].unique())
        
        # Enhanced Power 5 conference weighting
        self.power_conferences = {
            'B12': 1.5,  # Big 12
            'SEC': 1.4,  # Southeastern Conference
            'B10': 1.4,  # Big Ten
            'ACC': 1.4,  # Atlantic Coast Conference
            'P12': 1.3,  # Pac-12
            'BE': 1.2,   # Big East (historically strong basketball conference)
            'AAC': 1.1,  # American Athletic Conference
            'MWC': 1.1,  # Mountain West Conference
        }
        
        # Group teams by region and seed
        region_teams = {}
        for region in self.regions:
            region_teams[region] = self.teams_df[self.teams_df['Region'] == region].sort_values('SeedNum').to_dict('records')
        self.region_teams = region_teams
        
        # Set up bracket structure
        self.bracket = self._initialize_bracket()
        
        # Load historical results and ML models
        self.historical_results = self._load_historical_results()
        self.ml_models = self._load_ml_models()
        
        # Define the statistic category weights (from most to least important)
        self.base_stat_weights = {
            'Tempo': 1.0,
            'TempoRank': 0.9,
            'ORating': 3.0,       # Increased weight for offensive efficiency
            'DRating': 3.0,       # Increased weight for defensive efficiency
            'ORank': 1.2,
            'DRank': 1.2,
            'NetRating': 2.5,     # Increased weight for net rating
            'Seed': 1.0,
            'SeedNum': 1.0,
            'WinPct': 2.0,        # New metric for win percentage
            'ConferenceStrength': 1.5,  # New metric for conference strength
            'HistoricalSuccess': 1.0,   # New metric for historical success
            'ExpectedPointsMargin': 2.0  # New advanced metric
        }
        
        # Dynamically adjusted weights (will be updated based on learning)
        self.dynamic_weights = self.base_stat_weights.copy()
        self._load_dynamic_weights()
        
        # Statistical correlation trackers
        self.stat_to_success_correlation = {}
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Initialize the simulation insights store
        self.simulation_insights = defaultdict(list)
        
        # Track matchup-specific insights
        self.matchup_insights = defaultdict(lambda: defaultdict(list))
        
        # Set the default model
        self.active_model = 'gradient_boosting'
        
        # Learning rate for weight adjustments (smaller = more conservative updates)
        self.learning_rate = 0.05
        
        # Round-specific difficulty adjustments
        self.round_difficulty = {
            'first_round': 1.0,
            'second_round': 1.2,
            'sweet_sixteen': 1.5,
            'elite_eight': 1.8,
            'semifinals': 2.0,
            'championship': 2.5
        }
        
        # Parameters that control simulation behavior
        self.simulation_params = {
            'upset_factor': 1.0,  # Baseline value, will be adjusted over time
            'randomness_factor': 1.0,  # Baseline value, will be adjusted over time
            'conference_factor': 1.0,  # How much conference strength matters
            'seed_factor': 1.0,  # How much seed value matters
            'historical_factor': 1.0,  # How much historical performance matters
            'recency_factor': 1.0  # How much recent performances matter
        }
        
        # Initialize the simulation counter (used for recency weighting)
        self.simulation_counter = self.historical_results.get('simulation_count', 0)
    
    def _load_historical_results(self):
        """Load historical tournament results and simulation data for ML training."""
        history_file = 'data/tournament_history.pkl'
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'rb') as f:
                    history_data = pickle.load(f)
                    
                # Update the structure if it's from an older version
                if 'simulation_insights' not in history_data:
                    history_data['simulation_insights'] = defaultdict(list)
                if 'matchup_insights' not in history_data:
                    history_data['matchup_insights'] = defaultdict(lambda: defaultdict(list))
                if 'stat_correlations' not in history_data:
                    history_data['stat_correlations'] = {}
                if 'feature_importance' not in history_data:
                    history_data['feature_importance'] = {}
                if 'dynamic_weights' not in history_data:
                    history_data['dynamic_weights'] = self.base_stat_weights.copy()
                
                return history_data
                
            except Exception as e:
                print(f"Error loading historical data: {e}")
                return self._initialize_historical_results()
        else:
            return self._initialize_historical_results()
    
    def _initialize_historical_results(self):
        """Initialize the enhanced historical results structure."""
        return {
            'matchups': defaultdict(lambda: defaultdict(int)),  # Store team vs team results
            'team_stats': defaultdict(lambda: {
                'wins': 0,
                'losses': 0,
                'round_reached': defaultdict(int),
                'recent_performance': [],  # List of recent performance metrics
                'average_margin': [],  # List of score margins
                'upsets_created': 0,  # Number of times team upset higher seeds
                'been_upset': 0,  # Number of times team lost to lower seeds
            }),
            'simulation_count': 0,
            'simulation_insights': defaultdict(list),  # Store insights from simulations
            'matchup_insights': defaultdict(lambda: defaultdict(list)),  # Matchup-specific insights
            'stat_correlations': {},  # Correlation between stats and success
            'feature_importance': {},  # Importance of different features in prediction
            'dynamic_weights': self.base_stat_weights.copy(),  # Dynamically adjusted weights
            'upset_tracker': {  # Track upset patterns
                'by_seed_diff': defaultdict(lambda: {'occurred': 0, 'total': 0}),
                'by_round': defaultdict(lambda: {'occurred': 0, 'total': 0}),
                'by_conference': defaultdict(lambda: {'occurred': 0, 'total': 0})
            },
            'training_data': {  # Store data for ML training
                'features': [],
                'outcomes': []
            }
        }
    
    def _load_dynamic_weights(self):
        """Load the dynamically learned weights from historical data."""
        if 'dynamic_weights' in self.historical_results:
            self.dynamic_weights = self.historical_results['dynamic_weights']
    
    def _load_ml_models(self):
        """Load or initialize machine learning models for game prediction."""
        models_dir = 'data/ml_models'
        os.makedirs(models_dir, exist_ok=True)
        
        models = {
            'random_forest': None,
            'gradient_boosting': None
        }
        
        # Try to load existing models
        for model_name in models.keys():
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            if os.path.exists(model_path):
                try:
                    models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} model from {model_path}")
                except Exception as e:
                    print(f"Error loading {model_name} model: {e}")
                    models[model_name] = self._initialize_ml_model(model_name)
            else:
                models[model_name] = self._initialize_ml_model(model_name)
        
        return models
    
    def _initialize_ml_model(self, model_type):
        """Initialize a new machine learning model of the specified type."""
        if model_type == 'random_forest':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ))
            ])
            return pipeline
        
        elif model_type == 'gradient_boosting':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ))
            ])
            return pipeline
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _save_ml_models(self):
        """Save the trained machine learning models."""
        models_dir = 'data/ml_models'
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.ml_models.items():
            if model is not None:
                model_path = os.path.join(models_dir, f"{model_name}.joblib")
                try:
                    joblib.dump(model, model_path)
                    print(f"Saved {model_name} model to {model_path}")
                except Exception as e:
                    print(f"Error saving {model_name} model: {e}")
    
    def _save_historical_results(self):
        """Save the enhanced historical results to disk."""
        history_file = 'data/tournament_history.pkl'
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        # Update the historical results with the latest simulation insights
        self.historical_results['simulation_insights'] = self.simulation_insights
        self.historical_results['matchup_insights'] = self.matchup_insights
        self.historical_results['stat_correlations'] = self.stat_to_success_correlation
        self.historical_results['feature_importance'] = self.feature_importance
        self.historical_results['dynamic_weights'] = self.dynamic_weights
        
        try:
            with open(history_file, 'wb') as f:
                pickle.dump(self.historical_results, f)
            print(f"Saved historical data to {history_file}")
        except Exception as e:
            print(f"Error saving historical data: {e}")
    
    def _initialize_bracket(self):
        """Initialize the tournament bracket structure"""
        bracket = {
            'regions': {},
            'final_four': {
                'semifinals': [None, None],
                'championship': None,
                'champion': None
            }
        }
        
        # Create the initial matchups for each region
        for region in self.regions:
            region_teams = self.region_teams[region]
            first_round = []
            
            # Create 8 first-round matchups in the standard NCAA format (1 vs 16, 8 vs 9, etc.)
            seed_pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
            
            for seed1, seed2 in seed_pairs:
                team1 = next((t for t in region_teams if t['SeedNum'] == seed1), None)
                team2 = next((t for t in region_teams if t['SeedNum'] == seed2), None)
                
                if team1 and team2:
                    matchup = {
                        'team1': team1['Team'],
                        'team2': team2['Team'],
                        'winner': None,
                        'score_margin': None,
                        'key_factors': []
                    }
                    first_round.append(matchup)
            
            bracket['regions'][region] = {
                'first_round': first_round,
                'second_round': [None] * 4,
                'sweet_sixteen': [None] * 2,
                'elite_eight': None,
                'regional_champion': None
            }
        
        return bracket
    
    def extract_team_features(self, team_name, opponent_name=None, round_name='first_round'):
        """
        Extract comprehensive features for a team to be used in ML prediction.
        
        Args:
            team_name: The name of the team
            opponent_name: The name of the opponent (optional, for relative comparisons)
            round_name: The current tournament round
            
        Returns:
            Dictionary of features
        """
        team = self.team_dict[team_name]
        features = {}
        
        # Basic team metrics
        features['Tempo'] = team.get('Tempo', 0)
        features['ORating'] = team.get('ORating', 0)
        features['DRating'] = team.get('DRating', 0)
        features['NetRating'] = team.get('NetRating', 0)
        features['SeedNum'] = team.get('SeedNum', 16)  # Default to lowest seed if missing
        
        # Add rank features (with default values if missing)
        features['ORank'] = team.get('ORank', 350)
        features['DRank'] = team.get('DRank', 350)
        features['TempoRank'] = team.get('TempoRank', 350)
        
        # Add conference strength
        conference = team.get('Conference', '')
        features['ConferenceStrength'] = self.power_conferences.get(conference, 1.0)
        
        # Calculate win percentage from record
        if 'Record' in team:
            try:
                w, l = map(int, team['Record'].split('-'))
                features['WinPct'] = w / (w + l)
                
                # Adjust win percentage based on conference strength
                if conference in self.power_conferences:
                    features['AdjustedWinPct'] = features['WinPct'] * self.power_conferences[conference]
                else:
                    features['AdjustedWinPct'] = features['WinPct']
                    
            except:
                features['WinPct'] = 0.5
                features['AdjustedWinPct'] = 0.5
        else:
            features['WinPct'] = 0.5
            features['AdjustedWinPct'] = 0.5
        
        # Add historical performance metrics
        team_history = self.historical_results['team_stats'].get(team_name, {})
        
        total_games = team_history.get('wins', 0) + team_history.get('losses', 0)
        if total_games > 0:
            features['HistoricalWinPct'] = team_history.get('wins', 0) / total_games
        else:
            features['HistoricalWinPct'] = 0.5
        
        # Historical tournament success metrics
        for r in ['sweet_sixteen', 'elite_eight', 'semifinals', 'championship', 'champion']:
            round_count = team_history.get('round_reached', {}).get(r, 0)
            features[f'Historical_{r}'] = round_count
        
        # Calculate a composite tournament success score
        success_score = (
            team_history.get('round_reached', {}).get('sweet_sixteen', 0) * 1 +
            team_history.get('round_reached', {}).get('elite_eight', 0) * 2 +
            team_history.get('round_reached', {}).get('semifinals', 0) * 4 +
            team_history.get('round_reached', {}).get('championship', 0) * 8 +
            team_history.get('round_reached', {}).get('champion', 0) * 16
        )
        features['TournamentSuccessScore'] = success_score
        
        # Add upset-related metrics
        features['UpsetsCreated'] = team_history.get('upsets_created', 0)
        features['BeenUpset'] = team_history.get('been_upset', 0)
        
        # Calculate an upset potential score
        if features['BeenUpset'] > 0:
            upset_vulnerability = features['BeenUpset'] / max(1, features['BeenUpset'] + features['UpsetsCreated'])
        else:
            upset_vulnerability = 0
        
        features['UpsetVulnerability'] = upset_vulnerability
        
        # Add expected points margin based on offensive and defensive efficiency
        # Formula: (ORating/100 - DRating/100) * (Tempo/70)
        features['ExpectedPointsMargin'] = (features['ORating']/100 - features['DRating']/100) * (features['Tempo']/70)
        
        # If opponent is provided, add relative comparison features
        if opponent_name:
            opponent = self.team_dict[opponent_name]
            opponent_features = self.extract_team_features(opponent_name, round_name=round_name)
            
            # Add relative features (team vs opponent)
            for key in ['ORating', 'DRating', 'NetRating', 'Tempo', 'ConferenceStrength', 
                        'WinPct', 'AdjustedWinPct', 'TournamentSuccessScore', 'ExpectedPointsMargin']:
                if key in features and key in opponent_features:
                    features[f'Rel_{key}'] = features[key] - opponent_features[key]
            
            # Add seed difference (positive if team is higher seeded, negative if lower)
            features['SeedDiff'] = opponent_features['SeedNum'] - features['SeedNum']
            
            # Add historical matchup data
            matchup_key = tuple(sorted([team_name, opponent_name]))
            matchup_history = self.historical_results['matchups'][matchup_key]
            
            if team_name in matchup_history and opponent_name in matchup_history:
                team_wins = matchup_history[team_name]
                opponent_wins = matchup_history[opponent_name]
                total_matchups = team_wins + opponent_wins
                
                if total_matchups > 0:
                    features['HeadToHeadWinPct'] = team_wins / total_matchups
                else:
                    features['HeadToHeadWinPct'] = 0.5
            else:
                features['HeadToHeadWinPct'] = 0.5
            
            # Add matchup-specific insights if available
            if team_name in self.matchup_insights and opponent_name in self.matchup_insights[team_name]:
                insights = self.matchup_insights[team_name][opponent_name]
                if insights:
                    features['MatchupInsightScore'] = sum(insights) / len(insights)
                else:
                    features['MatchupInsightScore'] = 0
            else:
                features['MatchupInsightScore'] = 0
        
        # Add round-specific difficulty multiplier
        features['RoundDifficulty'] = self.round_difficulty.get(round_name, 1.0)
        
        return features
    
    def prepare_ml_features(self, team1_name, team2_name, round_name='first_round'):
        """
        Prepare features for machine learning prediction of a game outcome.
        
        Args:
            team1_name: First team
            team2_name: Second team
            round_name: Tournament round
            
        Returns:
            Feature array for ML prediction
        """
        # Extract detailed features for both teams
        team1_features = self.extract_team_features(team1_name, team2_name, round_name)
        team2_features = self.extract_team_features(team2_name, team1_name, round_name)
        
        # Combine features for ML prediction
        combined_features = {}
        
        # Add direct team1 features
        for key, value in team1_features.items():
            combined_features[f'team1_{key}'] = value
        
        # Add direct team2 features
        for key, value in team2_features.items():
            combined_features[f'team2_{key}'] = value
        
        # Add relative features (team1 vs team2)
        for key in ['ORating', 'DRating', 'NetRating', 'Tempo', 'ConferenceStrength', 
                    'WinPct', 'AdjustedWinPct', 'TournamentSuccessScore', 'ExpectedPointsMargin']:
            if key in team1_features and key in team2_features:
                combined_features[f'diff_{key}'] = team1_features[key] - team2_features[key]
        
        # Add seed difference (positive if team1 is higher seeded)
        combined_features['seed_diff'] = team2_features['SeedNum'] - team1_features['SeedNum']
        
        # Convert the dictionary to a list in a consistent order
        # This is crucial for ML models to have consistent feature ordering
        feature_names = sorted(combined_features.keys())
        feature_values = [combined_features[name] for name in feature_names]
        
        return np.array(feature_values).reshape(1, -1), feature_names
    
    def train_ml_models(self):
        """
        Train the machine learning models on historical data.
        
        Returns:
            True if training was successful, False otherwise
        """
        # Check if we have enough data for training
        if len(self.historical_results['training_data']['features']) < 50:
            print("Not enough training data yet. Need at least 50 games.")
            return False
        
        try:
            # Convert the stored features and outcomes to numpy arrays
            X = np.array(self.historical_results['training_data']['features'])
            y = np.array(self.historical_results['training_data']['outcomes'])
            
            # Split the data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train the Random Forest model
            self.ml_models['random_forest'].fit(X_train, y_train)
            rf_val_accuracy = accuracy_score(y_val, self.ml_models['random_forest'].predict(X_val))
            print(f"Random Forest validation accuracy: {rf_val_accuracy:.2f}")
            
            # Train the Gradient Boosting model
            self.ml_models['gradient_boosting'].fit(X_train, y_train)
            gb_val_accuracy = accuracy_score(y_val, self.ml_models['gradient_boosting'].predict(X_val))
            print(f"Gradient Boosting validation accuracy: {gb_val_accuracy:.2f}")
            
            # Extract feature importance from the Random Forest model
            feature_names = sorted(self.historical_results['training_data']['feature_names'])
            if hasattr(self.ml_models['random_forest']['model'], 'feature_importances_'):
                importances = self.ml_models['random_forest']['model'].feature_importances_
                self.feature_importance = {
                    name: importance for name, importance in zip(feature_names, importances)
                }
            
            # Update the dynamic weights based on feature importance
            self._update_dynamic_weights()
            
            # Save the trained models
            self._save_ml_models()
            
            return True
            
        except Exception as e:
            print(f"Error training ML models: {e}")
            return False
    
    def _update_dynamic_weights(self):
        """Update the dynamic weights based on feature importance."""
        if not self.feature_importance:
            return
        
        # Focus on key features that directly relate to the base weights
        weight_related_features = {
            'ORating': ['team1_ORating', 'team2_ORating', 'diff_ORating'],
            'DRating': ['team1_DRating', 'team2_DRating', 'diff_DRating'],
            'NetRating': ['team1_NetRating', 'team2_NetRating', 'diff_NetRating'],
            'Tempo': ['team1_Tempo', 'team2_Tempo', 'diff_Tempo'],
            'SeedNum': ['team1_SeedNum', 'team2_SeedNum', 'seed_diff'],
            'WinPct': ['team1_WinPct', 'team2_WinPct', 'diff_WinPct'],
            'ConferenceStrength': ['team1_ConferenceStrength', 'team2_ConferenceStrength', 'diff_ConferenceStrength'],
            'HistoricalSuccess': ['team1_TournamentSuccessScore', 'team2_TournamentSuccessScore'],
            'ExpectedPointsMargin': ['team1_ExpectedPointsMargin', 'team2_ExpectedPointsMargin', 'diff_ExpectedPointsMargin']
        }
        
        # Compute average importance for each weight category
        for weight_key, feature_keys in weight_related_features.items():
            importance_values = [
                self.feature_importance[feature] 
                for feature in feature_keys 
                if feature in self.feature_importance
            ]
            
            if importance_values:
                avg_importance = sum(importance_values) / len(importance_values)
                
                # Adjust the dynamic weight with a learning rate
                current_weight = self.dynamic_weights.get(weight_key, self.base_stat_weights.get(weight_key, 1.0))
                new_weight = current_weight * (1 - self.learning_rate) + avg_importance * 10 * self.learning_rate
                
                # Ensure the weight doesn't drop too low or go too high
                new_weight = max(0.1, min(5.0, new_weight))
                
                # Update the dynamic weight
                self.dynamic_weights[weight_key] = new_weight
        
        # Special handling for Power 5 conferences
        if self.simulation_counter > 50:  # Once we have enough simulations
            # Analyze winning percentage by conference
            conference_success = {}
            for team_name, stats in self.historical_results['team_stats'].items():
                if team_name in self.team_dict:
                    conference = self.team_dict[team_name].get('Conference', '')
                    if conference:
                        wins = stats.get('wins', 0)
                        losses = stats.get('losses', 0)
                        
                        if wins + losses > 0:
                            if conference not in conference_success:
                                conference_success[conference] = {'wins': 0, 'losses': 0}
                            
                            conference_success[conference]['wins'] += wins
                            conference_success[conference]['losses'] += losses
            
            # Calculate win percentage for each conference
            for conf, stats in conference_success.items():
                total_games = stats['wins'] + stats['losses']
                if total_games > 10:  # Only consider conferences with enough games
                    win_pct = stats['wins'] / total_games
                    
                    # Adjust Power 5 conference factors based on performance
                    if conf in self.power_conferences:
                        # Scale from 1.0 to 2.0 based on win percentage (0.5 to 0.8)
                        new_factor = 1.0 + (win_pct - 0.5) * 3.3
                        new_factor = max(1.0, min(2.0, new_factor))
                        
                        # Apply a conservative adjustment
                        current_factor = self.power_conferences[conf]
                        self.power_conferences[conf] = current_factor * 0.9 + new_factor * 0.1
    
    def calculate_team_strength(self, team_name, opponent_name=None, round_name='first_round'):
        """
        Calculate team strength using the enhanced model with ML predictions.
        
        Args:
            team_name: The name of the team
            opponent_name: The opponent (required for ML model)
            round_name: The current tournament round
            
        Returns:
            Strength score for the team
        """
        # If ML models are available and opponent is provided, use ML prediction
        if opponent_name and all(model is not None for model in self.ml_models.values()) and self.historical_results['simulation_count'] >= 50:
            # Prepare features for ML prediction
            features, feature_names = self.prepare_ml_features(team_name, opponent_name, round_name)
            
            # Get predictions from both models
            rf_prob = 0.5
            gb_prob = 0.5
            
            try:
                # Random Forest prediction
                if hasattr(self.ml_models['random_forest'], 'predict_proba'):
                    rf_prob = self.ml_models['random_forest'].predict_proba(features)[0][1]
                else:
                    rf_pred = self.ml_models['random_forest'].predict(features)[0]
                    rf_prob = 1.0 if rf_pred == 1 else 0.0
                
                # Gradient Boosting prediction
                if hasattr(self.ml_models['gradient_boosting'], 'predict_proba'):
                    gb_prob = self.ml_models['gradient_boosting'].predict_proba(features)[0][1]
                else:
                    gb_pred = self.ml_models['gradient_boosting'].predict(features)[0]
                    gb_prob = 1.0 if gb_pred == 1 else 0.0
                
                # Weighted ensemble of models (favoring gradient boosting slightly)
                ml_win_prob = 0.4 * rf_prob + 0.6 * gb_prob
                
                # Convert ML probability to a strength score
                # Scale from 0-1 to a wider range for better differentiation
                ml_strength = ml_win_prob * 10  # Scale to 0-10 range
                
                # Add a record of the prediction for later analysis
                self.simulation_insights[f"{team_name}_vs_{opponent_name}"].append({
                    'round': round_name,
                    'rf_prob': rf_prob,
                    'gb_prob': gb_prob,
                    'combined_prob': ml_win_prob,
                    'features': {name: value for name, value in zip(feature_names, features[0])}
                })
                
                return ml_strength
                
            except Exception as e:
                print(f"Error in ML prediction: {e}, falling back to heuristic calculation")
                # Fall back to heuristic calculation
                pass
        
        # Heuristic calculation (used when ML is not available or as fallback)
        team = self.team_dict[team_name]
        
        # Initialize strength score
        strength = 0
        
        # Apply dynamic weights to each statistic
        if 'Tempo' in team:
            strength += team['Tempo'] * self.dynamic_weights['Tempo'] / 100
        
        if 'ORating' in team:
            strength += team['ORating'] * self.dynamic_weights['ORating'] / 100
        
        if 'DRating' in team:
            # Lower defensive rating is better
            strength += (120 - team['DRating']) * self.dynamic_weights['DRating'] / 120
        
        if 'NetRating' in team:
            # Adjust to ensure positive contribution for good teams
            strength += (team['NetRating'] + 50) * self.dynamic_weights['NetRating'] / 100
        
        if 'SeedNum' in team:
            # Lower seed is better
            strength += (17 - team['SeedNum']) * self.dynamic_weights['SeedNum'] / 16
        
        # Win-loss record contribution with enhanced Power 5 weighting
        if 'Record' in team:
            try:
                w, l = map(int, team['Record'].split('-'))
                win_pct = w / (w + l)
                
                # Apply conference-based multiplier for Power 5 teams
                conference = team.get('Conference', '')
                if conference in self.power_conferences:
                    # Increase weight for Power 5 conference W-L records
                    conf_multiplier = self.power_conferences[conference]
                    win_pct = win_pct * conf_multiplier
                
                strength += win_pct * self.dynamic_weights['WinPct']
            except:
                pass
        
        # Advanced conference strength contribution
        if 'Conference' in team:
            conf_factor = self.power_conferences.get(team['Conference'], 1.0)
            strength += conf_factor * self.dynamic_weights['ConferenceStrength']
        
        # Historical performance contribution
        team_history = self.historical_results['team_stats'].get(team_name, {'wins': 0, 'losses': 0})
        
        # Calculate win percentage from historical simulations
        total_games = team_history.get('wins', 0) + team_history.get('losses', 0)
        if total_games > 0:
            historical_win_pct = team_history.get('wins', 0) / total_games
            
            # Weight historical performance by round
            round_weights = {
                'first_round': 0.1,
                'second_round': 0.15,
                'sweet_sixteen': 0.2,
                'elite_eight': 0.25,
                'semifinals': 0.3,
                'championship': 0.35
            }
            
            strength += historical_win_pct * round_weights.get(round_name, 0.2) * self.dynamic_weights['HistoricalSuccess']
            
            # Add bonus for teams that have historically gone far
            deep_run_bonus = 0
            for r, count in team_history.get('round_reached', {}).items():
                round_value = {
                    'sweet_sixteen': 0.1,
                    'elite_eight': 0.2,
                    'semifinals': 0.4,
                    'championship': 0.6,
                    'champion': 1.0
                }.get(r, 0)
                
                deep_run_bonus += count * round_value
            
            # Cap and scale the bonus
            deep_run_bonus = min(deep_run_bonus, 3.0)
            strength += deep_run_bonus * self.dynamic_weights['HistoricalSuccess'] / 3
        
        # Expected points margin contribution
        if 'ORating' in team and 'DRating' in team and 'Tempo' in team:
            expected_margin = (team['ORating']/100 - team['DRating']/100) * (team['Tempo']/70)
            strength += expected_margin * self.dynamic_weights['ExpectedPointsMargin']
        
        # If opponent is provided, add head-to-head and matchup-specific adjustments
        if opponent_name:
            # Get historical head-to-head results
            matchup_key = tuple(sorted([team_name, opponent_name]))
            matchup_history = self.historical_results['matchups'][matchup_key]
            
            if team_name in matchup_history and opponent_name in matchup_history:
                team_wins = matchup_history[team_name]
                opponent_wins = matchup_history[opponent_name]
                total_matchups = team_wins + opponent_wins
                
                if total_matchups > 0:
                    # Calculate head-to-head advantage
                    h2h_advantage = (team_wins / total_matchups - 0.5) * 2  # Range: -1 to 1
                    
                    # Adjust strength based on head-to-head history, weighted by tournament round
                    h2h_weight = {
                        'first_round': 0.2,
                        'second_round': 0.25,
                        'sweet_sixteen': 0.3,
                        'elite_eight': 0.35,
                        'semifinals': 0.4,
                        'championship': 0.5
                    }.get(round_name, 0.3)
                    
                    strength += h2h_advantage * h2h_weight
            
            # Add matchup-specific insights if available
            if team_name in self.matchup_insights and opponent_name in self.matchup_insights[team_name]:
                insights = self.matchup_insights[team_name][opponent_name]
                if insights:
                    # Calculate the average insight score
                    avg_insight = sum(insights) / len(insights)
                    
                    # Weight more heavily for more recent insights
                    recency_weight = min(1.0, len(insights) / 10)  # Cap at 1.0
                    
                    strength += avg_insight * recency_weight
        
        return strength
    
    def _analyze_key_factors(self, team1_name, team2_name, team1_strength, team2_strength):
        """
        Analyze key factors that contributed to the predicted outcome.
        
        Args:
            team1_name: First team name
            team2_name: Second team name
            team1_strength: Calculated strength for team1
            team2_strength: Calculated strength for team2
            
        Returns:
            List of key factors as strings
        """
        team1 = self.team_dict[team1_name]
        team2 = self.team_dict[team2_name]
        key_factors = []
        
        # Determine the predicted winner
        predicted_winner = team1_name if team1_strength > team2_strength else team2_name
        predicted_loser = team2_name if predicted_winner == team1_name else team1_name
        
        # Check seed difference
        seed_diff = abs(team1['SeedNum'] - team2['SeedNum'])
        if seed_diff >= 4:
            higher_seed = team1_name if team1['SeedNum'] < team2['SeedNum'] else team2_name
            lower_seed = team2_name if higher_seed == team1_name else team1_name
            
            if predicted_winner == lower_seed:
                key_factors.append(f"Upset: {lower_seed} (seed {self.team_dict[lower_seed]['SeedNum']}) over {higher_seed} (seed {self.team_dict[higher_seed]['SeedNum']})")
            else:
                key_factors.append(f"Seed advantage: {higher_seed} (seed {self.team_dict[higher_seed]['SeedNum']})")
        
        # Check offensive efficiency difference
        if 'ORating' in team1 and 'ORating' in team2:
            o_diff = abs(team1['ORating'] - team2['ORating'])
            if o_diff > 5:
                better_offense = team1_name if team1['ORating'] > team2['ORating'] else team2_name
                if predicted_winner == better_offense:
                    key_factors.append(f"Offensive advantage: {better_offense} (ORating: {self.team_dict[better_offense]['ORating']:.1f})")
        
        # Check defensive efficiency difference
        if 'DRating' in team1 and 'DRating' in team2:
            d_diff = abs(team1['DRating'] - team2['DRating'])
            if d_diff > 5:
                better_defense = team1_name if team1['DRating'] < team2['DRating'] else team2_name
                if predicted_winner == better_defense:
                    key_factors.append(f"Defensive advantage: {better_defense} (DRating: {self.team_dict[better_defense]['DRating']:.1f})")
        
        # Check conference difference
        conf1 = team1.get('Conference', '')
        conf2 = team2.get('Conference', '')
        conf1_strength = self.power_conferences.get(conf1, 1.0)
        conf2_strength = self.power_conferences.get(conf2, 1.0)
        
        if abs(conf1_strength - conf2_strength) > 0.2:
            stronger_conf_team = team1_name if conf1_strength > conf2_strength else team2_name
            stronger_conf = self.team_dict[stronger_conf_team]['Conference']
            
            if predicted_winner == stronger_conf_team:
                key_factors.append(f"Conference advantage: {stronger_conf_team} ({stronger_conf})")
        
        # Check historical tournament success
        team1_history = self.historical_results['team_stats'].get(team1_name, {})
        team2_history = self.historical_results['team_stats'].get(team2_name, {})
        
        team1_success = sum(team1_history.get('round_reached', {}).get(r, 0) for r in ['sweet_sixteen', 'elite_eight', 'semifinals', 'championship', 'champion'])
        team2_success = sum(team2_history.get('round_reached', {}).get(r, 0) for r in ['sweet_sixteen', 'elite_eight', 'semifinals', 'championship', 'champion'])
        
        if abs(team1_success - team2_success) > 3:
            experienced_team = team1_name if team1_success > team2_success else team2_name
            if predicted_winner == experienced_team:
                key_factors.append(f"Tournament experience: {experienced_team}")
        
        # Check head-to-head history
        matchup_key = tuple(sorted([team1_name, team2_name]))
        matchup_history = self.historical_results['matchups'][matchup_key]
        
        if team1_name in matchup_history and team2_name in matchup_history:
            team1_wins = matchup_history[team1_name]
            team2_wins = matchup_history[team2_name]
            
            if team1_wins > team2_wins and team1_wins >= 2:
                if predicted_winner == team1_name:
                    key_factors.append(f"Head-to-head advantage: {team1_name} ({team1_wins}-{team2_wins})")
            elif team2_wins > team1_wins and team2_wins >= 2:
                if predicted_winner == team2_name:
                    key_factors.append(f"Head-to-head advantage: {team2_name} ({team2_wins}-{team1_wins})")
        
        return key_factors
    
    def simulate_game(self, team1_name, team2_name, round_name='first_round'):
        """
        Simulate a single game between two teams, returning the winner and key insights.
        
        Args:
            team1_name: First team
            team2_name: Second team
            round_name: Tournament round
            
        Returns:
            Tuple of (winner_name, score_margin, key_factors)
        """
        # Handle None values
        if team1_name is None:
            return team2_name, 10, ["Walkover"]
        if team2_name is None:
            return team1_name, 10, ["Walkover"]
        
        # Calculate team strengths
        team1_strength = self.calculate_team_strength(team1_name, team2_name, round_name)
        team2_strength = self.calculate_team_strength(team2_name, team1_name, round_name)
        
        # Analyze key factors that influenced the outcome
        key_factors = self._analyze_key_factors(team1_name, team2_name, team1_strength, team2_strength)
        
        # Add randomness factor (more randomness in earlier rounds)
        randomness_factor = {
            'first_round': 0.3,
            'second_round': 0.25,
            'sweet_sixteen': 0.2,
            'elite_eight': 0.15,
            'semifinals': 0.1,
            'championship': 0.05
        }.get(round_name, 0.2)
        
        # Scale randomness by upset factor from simulation parameters
        randomness_factor *= self.simulation_params['randomness_factor']
        
        team1_strength += np.random.normal(0, randomness_factor * team1_strength)
        team2_strength += np.random.normal(0, randomness_factor * team2_strength)
        
        # Calculate expected score margin based on strength difference
        # This is a heuristic approximation: each point of strength roughly equals 2 points in the game
        expected_margin = (team1_strength - team2_strength) * 2
        
        # Add random variation to the margin
        actual_margin = expected_margin + np.random.normal(0, 5)
        
        # Determine winner based on actual margin
        if actual_margin > 0:
            winner = team1_name
            margin = actual_margin
        else:
            winner = team2_name
            margin = -actual_margin
        
        # Prepare feature data for ML training
        features, feature_names = self.prepare_ml_features(team1_name, team2_name, round_name)
        outcome = 1 if winner == team1_name else 0
        
        # Store the data for future ML training
        self.historical_results['training_data']['features'].append(features[0])
        self.historical_results['training_data']['outcomes'].append(outcome)
        
        # Store feature names if not already stored
        if 'feature_names' not in self.historical_results['training_data']:
            self.historical_results['training_data']['feature_names'] = feature_names
        
        # Update historical matchup data
        self._update_historical_results(team1_name, team2_name, winner, round_name, margin)
        
        # Update team-specific insights based on the game outcome
        self._update_team_insights(team1_name, team2_name, winner, round_name, margin)
        
        # Track if this was an upset
        self._track_upset(team1_name, team2_name, winner, round_name)
        
        return winner, round(margin, 1), key_factors
    
    def _track_upset(self, team1_name, team2_name, winner, round_name):
        """Track if the game result was an upset based on seed."""
        team1 = self.team_dict[team1_name]
        team2 = self.team_dict[team2_name]
        
        # Compare seeds to detect upsets
        if team1['SeedNum'] < team2['SeedNum']:
            favored_team = team1_name
            underdog = team2_name
            seed_diff = team2['SeedNum'] - team1['SeedNum']
        else:
            favored_team = team2_name
            underdog = team1_name
            seed_diff = team1['SeedNum'] - team2['SeedNum']
        
        # Only consider significant seed differences as potential upsets
        if seed_diff >= 3:
            upset_tracker = self.historical_results['upset_tracker']
            
            # Track by seed difference
            upset_tracker['by_seed_diff'][seed_diff]['total'] += 1
            
            # Track by round
            upset_tracker['by_round'][round_name]['total'] += 1
            
            # Track by conference matchup
            favored_conf = self.team_dict[favored_team].get('Conference', 'Unknown')
            underdog_conf = self.team_dict[underdog].get('Conference', 'Unknown')
            conf_matchup = f"{underdog_conf}_vs_{favored_conf}"
            upset_tracker['by_conference'][conf_matchup]['total'] += 1
            
            # If the underdog won, it's an upset
            if winner == underdog:
                # Update upset stats
                upset_tracker['by_seed_diff'][seed_diff]['occurred'] += 1
                upset_tracker['by_round'][round_name]['occurred'] += 1
                upset_tracker['by_conference'][conf_matchup]['occurred'] += 1
                
                # Update team records for upsets
                self.historical_results['team_stats'][underdog]['upsets_created'] += 1
                self.historical_results['team_stats'][favored_team]['been_upset'] += 1
    
    def _update_historical_results(self, team1, team2, winner, round_name, margin=None):
        """
        Update historical results with enhanced tracking.
        
        Args:
            team1, team2: The teams that played
            winner: The winning team
            round_name: The tournament round
            margin: The score margin (if available)
        """
        # Skip if either team is None
        if team1 is None or team2 is None:
            return
        
        # Update matchup history
        matchup_key = tuple(sorted([team1, team2]))
        self.historical_results['matchups'][matchup_key][winner] += 1
        
        # Update team win/loss records
        loser = team2 if winner == team1 else team1
        self.historical_results['team_stats'][winner]['wins'] += 1
        self.historical_results['team_stats'][loser]['losses'] += 1
        
        # Update round reached for both teams
        self._update_round_reached(winner, round_name)
        
        # Track score margin if provided
        if margin is not None:
            self.historical_results['team_stats'][winner]['average_margin'].append(margin)
            self.historical_results['team_stats'][loser]['average_margin'].append(-margin)
        
        # Update recent performance with a recency bias
        # More recent games are given higher weight in the history
        self.historical_results['team_stats'][winner]['recent_performance'].append(1)
        self.historical_results['team_stats'][loser]['recent_performance'].append(0)
        
        # Keep only the most recent 20 games
        for team in [winner, loser]:
            recent_perf = self.historical_results['team_stats'][team]['recent_performance']
            if len(recent_perf) > 20:
                self.historical_results['team_stats'][team]['recent_performance'] = recent_perf[-20:]
    
    def _update_team_insights(self, team1, team2, winner, round_name, margin):
        """
        Update team-specific insights based on the game outcome.
        These insights help refine future predictions.
        """
        # Calculate insight score based on winner and margin
        # Positive score means team1 performed better than expected
        # Negative score means team2 performed better than expected
        if winner == team1:
            insight_score = min(1.0, margin / 20)  # Cap at 1.0
        else:
            insight_score = -min(1.0, margin / 20)  # Cap at -1.0
        
        # Store the insight for future matchups
        self.matchup_insights[team1][team2].append(insight_score)
        self.matchup_insights[team2][team1].append(-insight_score)
        
        # Keep only the most recent 10 insights
        if len(self.matchup_insights[team1][team2]) > 10:
            self.matchup_insights[team1][team2] = self.matchup_insights[team1][team2][-10:]
        if len(self.matchup_insights[team2][team1]) > 10:
            self.matchup_insights[team2][team1] = self.matchup_insights[team2][team1][-10:]
    
    def _update_round_reached(self, team, round_name):
        """Update the furthest round reached by a team."""
        self.historical_results['team_stats'][team]['round_reached'][round_name] += 1
    
    def simulate_round(self, round_type, region=None):
        """
        Simulate a round of the tournament with enhanced insights tracking.
        
        Args:
            round_type: The round to simulate ('first_round', 'second_round', etc.)
            region: The region to simulate (None for Final Four)
        """
        if region:
            if round_type == 'first_round':
                matchups = self.bracket['regions'][region][round_type]
                next_round = 'second_round'
                next_round_idx = 0
                
                for i, matchup in enumerate(matchups):
                    winner, margin, key_factors = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                    matchup['winner'] = winner
                    matchup['score_margin'] = margin
                    matchup['key_factors'] = key_factors
                    
                    # Set up the next round
                    if i % 2 == 0:
                        self.bracket['regions'][region][next_round][next_round_idx] = {
                            'team1': winner,
                            'team2': None,
                            'winner': None,
                            'score_margin': None,
                            'key_factors': []
                        }
                    else:
                        self.bracket['regions'][region][next_round][next_round_idx]['team2'] = winner
                        next_round_idx += 1
                
            elif round_type == 'second_round':
                matchups = self.bracket['regions'][region][round_type]
                next_round = 'sweet_sixteen'
                next_round_idx = 0
                
                for i, matchup in enumerate(matchups):
                    winner, margin, key_factors = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                    matchup['winner'] = winner
                    matchup['score_margin'] = margin
                    matchup['key_factors'] = key_factors
                    
                    # Set up the Sweet 16
                    if i % 2 == 0:
                        self.bracket['regions'][region][next_round][next_round_idx] = {
                            'team1': winner,
                            'team2': None,
                            'winner': None,
                            'score_margin': None,
                            'key_factors': []
                        }
                    else:
                        self.bracket['regions'][region][next_round][next_round_idx]['team2'] = winner
                        next_round_idx += 1
                
            elif round_type == 'sweet_sixteen':
                matchups = self.bracket['regions'][region][round_type]
                
                for i, matchup in enumerate(matchups):
                    winner, margin, key_factors = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                    matchup['winner'] = winner
                    matchup['score_margin'] = margin
                    matchup['key_factors'] = key_factors
                    
                    # Set up the Elite 8
                    if i == 0:
                        self.bracket['regions'][region]['elite_eight'] = {
                            'team1': winner,
                            'team2': None,
                            'winner': None,
                            'score_margin': None,
                            'key_factors': []
                        }
                    else:
                        self.bracket['regions'][region]['elite_eight']['team2'] = winner
                
            elif round_type == 'elite_eight':
                matchup = self.bracket['regions'][region][round_type]
                winner, margin, key_factors = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                matchup['winner'] = winner
                matchup['score_margin'] = margin
                matchup['key_factors'] = key_factors
                
                # Set the regional champion
                self.bracket['regions'][region]['regional_champion'] = winner
                
                # Determine which Final Four spot to fill
                region_idx = self.regions.index(region)
                if region_idx == 0:
                    if not self.bracket['final_four']['semifinals'][0]:
                        self.bracket['final_four']['semifinals'][0] = {
                            'team1': winner,
                            'team2': None,
                            'winner': None,
                            'score_margin': None,
                            'key_factors': []
                        }
                    else:
                        self.bracket['final_four']['semifinals'][0]['team2'] = winner
                elif region_idx == 1:
                    if not self.bracket['final_four']['semifinals'][0]:
                        self.bracket['final_four']['semifinals'][0] = {
                            'team1': None,
                            'team2': winner,
                            'winner': None,
                            'score_margin': None,
                            'key_factors': []
                        }
                    else:
                        self.bracket['final_four']['semifinals'][0]['team2'] = winner
                elif region_idx == 2:
                    if not self.bracket['final_four']['semifinals'][1]:
                        self.bracket['final_four']['semifinals'][1] = {
                            'team1': winner,
                            'team2': None,
                            'winner': None,
                            'score_margin': None,
                            'key_factors': []
                        }
                    else:
                        self.bracket['final_four']['semifinals'][1]['team2'] = winner
                elif region_idx == 3:
                    if not self.bracket['final_four']['semifinals'][1]:
                        self.bracket['final_four']['semifinals'][1] = {
                            'team1': None,
                            'team2': winner,
                            'winner': None,
                            'score_margin': None,
                            'key_factors': []
                        }
                    else:
                        self.bracket['final_four']['semifinals'][1]['team2'] = winner
        
        else:  # Final Four
            if round_type == 'semifinals':
                for i, matchup in enumerate(self.bracket['final_four']['semifinals']):
                    if matchup and matchup['team1'] and matchup['team2']:
                        winner, margin, key_factors = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                        matchup['winner'] = winner
                        matchup['score_margin'] = margin
                        matchup['key_factors'] = key_factors
                        
                        # Set up the championship game
                        if i == 0:
                            self.bracket['final_four']['championship'] = {
                                'team1': winner,
                                'team2': None,
                                'winner': None,
                                'score_margin': None,
                                'key_factors': []
                            }
                        else:
                            if self.bracket['final_four']['championship']:
                                self.bracket['final_four']['championship']['team2'] = winner
                            else:
                                self.bracket['final_four']['championship'] = {
                                    'team1': None,
                                    'team2': winner,
                                    'winner': None,
                                    'score_margin': None,
                                    'key_factors': []
                                }
            
            elif round_type == 'championship':
                matchup = self.bracket['final_four']['championship']
                if matchup and matchup['team1'] and matchup['team2']:
                    winner, margin, key_factors = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                    matchup['winner'] = winner
                    matchup['score_margin'] = margin
                    matchup['key_factors'] = key_factors
                    
                    # Set the champion
                    self.bracket['final_four']['champion'] = winner
    
    def analyze_simulation_results(self):
        """
        Analyze the results of the tournament simulation to extract insights.
        Updates the internal model to improve future predictions.
        """
        # Skip if we haven't run any simulations yet
        if self.historical_results['simulation_count'] < 1:
            return
        
        # Train ML models if we have enough data
        if len(self.historical_results['training_data']['features']) >= 50:
            self.train_ml_models()
        
        # Calculate correlation between team stats and tournament success
        stat_correlations = {}
        
        # Collect team stats and success levels
        team_stats = []
        success_levels = []
        
        for team_name, stats in self.historical_results['team_stats'].items():
            if team_name in self.team_dict:
                team = self.team_dict[team_name]
                
                # Calculate tournament success score
                success_score = (
                    stats.get('round_reached', {}).get('sweet_sixteen', 0) * 1 +
                    stats.get('round_reached', {}).get('elite_eight', 0) * 2 +
                    stats.get('round_reached', {}).get('semifinals', 0) * 4 +
                    stats.get('round_reached', {}).get('championship', 0) * 8 +
                    stats.get('round_reached', {}).get('champion', 0) * 16
                )
                
                # Skip teams with no tournament history
                if success_score == 0:
                    continue
                
                # Collect relevant stats
                team_stat = {}
                for key in ['Tempo', 'ORating', 'DRating', 'NetRating', 'SeedNum']:
                    if key in team:
                        team_stat[key] = team[key]
                
                # Parse win percentage from record
                if 'Record' in team:
                    try:
                        w, l = map(int, team['Record'].split('-'))
                        team_stat['WinPct'] = w / (w + l)
                    except:
                        team_stat['WinPct'] = 0.5
                else:
                    team_stat['WinPct'] = 0.5
                
                # Add conference strength
                conference = team.get('Conference', '')
                team_stat['ConferenceStrength'] = self.power_conferences.get(conference, 1.0)
                
                # Add to our collections
                team_stats.append(team_stat)
                success_levels.append(success_score)
        
        # Calculate correlations for each stat
        if team_stats and len(team_stats) > 5:
            for stat in team_stats[0].keys():
                values = [t.get(stat, 0) for t in team_stats]
                if len(set(values)) > 1:  # Ensure we have varying values
                    try:
                        correlation, p_value = pearsonr(values, success_levels)
                        stat_correlations[stat] = {
                            'correlation': correlation,
                            'p_value': p_value
                        }
                    except:
                        continue
        
        self.stat_to_success_correlation = stat_correlations
        
        # Use correlations to adjust dynamic weights
        for stat, corr_data in stat_correlations.items():
            if abs(corr_data['correlation']) > 0.2 and corr_data['p_value'] < 0.05:
                # Significant correlation found, adjust weight
                if stat in self.dynamic_weights:
                    # Scale correlation to a reasonable weight adjustment
                    adjustment = abs(corr_data['correlation']) * 2  # Scale factor
                    
                    # Different handling for positive and negative correlations
                    if corr_data['correlation'] > 0:
                        # Positive correlation: higher stat -> more success
                        new_weight = self.dynamic_weights[stat] * (1 + adjustment * 0.1)
                    else:
                        # Negative correlation: lower stat -> more success
                        new_weight = self.dynamic_weights[stat] * (1 - adjustment * 0.1)
                    
                    # Ensure weight stays in reasonable range
                    new_weight = max(0.5, min(5.0, new_weight))
                    
                    # Apply a conservative adjustment using learning rate
                    self.dynamic_weights[stat] = self.dynamic_weights[stat] * (1 - self.learning_rate) + new_weight * self.learning_rate
        
        # Analyze upset patterns
        upset_tracker = self.historical_results['upset_tracker']
        
        # Calculate upset rates by seed difference
        for seed_diff, data in upset_tracker['by_seed_diff'].items():
            if data['total'] >= 5:  # Only consider with enough samples
                upset_rate = data['occurred'] / data['total']
                
                # Adjust upset factor in simulation parameters
                if seed_diff >= 5 and upset_rate > 0.2:
                    # More upsets happening than expected for big seed differences
                    self.simulation_params['upset_factor'] *= 1.05
                elif seed_diff >= 5 and upset_rate < 0.1:
                    # Fewer upsets than expected
                    self.simulation_params['upset_factor'] *= 0.95
        
        # Ensure parameters stay in reasonable ranges
        self.simulation_params['upset_factor'] = max(0.5, min(2.0, self.simulation_params['upset_factor']))
        
        # Analyze conference strength in upsets
        for conf_matchup, data in upset_tracker['by_conference'].items():
            if data['total'] >= 5:
                upset_rate = data['occurred'] / data['total']
                
                # Extract conferences from the matchup string
                underdog_conf, favored_conf = conf_matchup.split('_vs_')
                
                # If a specific conference consistently creates upsets, increase its strength factor
                if upset_rate > 0.3 and underdog_conf in self.power_conferences:
                    current_strength = self.power_conferences[underdog_conf]
                    self.power_conferences[underdog_conf] = min(2.0, current_strength * 1.05)
                
                # If a specific conference is consistently upset, decrease its strength
                if upset_rate > 0.3 and favored_conf in self.power_conferences:
                    current_strength = self.power_conferences[favored_conf]
                    self.power_conferences[favored_conf] = max(1.0, current_strength * 0.95)
    
    def simulate_tournament(self):
        """
        Simulate the entire tournament from start to finish with enhanced learning.
        """
        print("Starting tournament simulation with enhanced learning...")
        
        # First simulate all regional games
        for region in self.regions:
            self.simulate_round('first_round', region)
            self.simulate_round('second_round', region)
            self.simulate_round('sweet_sixteen', region)
            self.simulate_round('elite_eight', region)
        
        # Then simulate the Final Four
        self.simulate_round('semifinals')
        self.simulate_round('championship')
        
        # Increment simulation count
        self.simulation_counter += 1
        self.historical_results['simulation_count'] += 1
        
        # Analyze results to improve future predictions
        self.analyze_simulation_results()
        
        # Save the updated historical results
        self._save_historical_results()
        
        # Create advanced analytics
        self.generate_analytics()
        
        print(f"Tournament simulation complete. Simulation #{self.historical_results['simulation_count']}")
        
        return self.bracket
    
    def generate_analytics(self):
        """Generate advanced analytics from the simulation results"""
        analytics = {
            'champion_history': {},
            'seed_success_rate': {},
            'conference_performance': {},
            'upset_analysis': {},
            'ml_model_accuracy': None,
            'dynamic_weights': self.dynamic_weights.copy(),
            'power_conferences': self.power_conferences.copy(),
            'simulation_count': self.historical_results['simulation_count']
        }
        
        # Track champion history
        for team_name, stats in self.historical_results['team_stats'].items():
            championships = stats.get('round_reached', {}).get('champion', 0)
            if championships > 0:
                analytics['champion_history'][team_name] = championships
        
        # Calculate seed success rates
        for team_name, stats in self.historical_results['team_stats'].items():
            if team_name in self.team_dict:
                seed = self.team_dict[team_name].get('SeedNum', 0)
                if seed not in analytics['seed_success_rate']:
                    analytics['seed_success_rate'][seed] = {
                        'sweet_sixteen': 0,
                        'elite_eight': 0,
                        'final_four': 0,
                        'championship': 0,
                        'champion': 0,
                        'total_teams': 0
                    }
                
                analytics['seed_success_rate'][seed]['total_teams'] += 1
                
                for round_name, count in stats.get('round_reached', {}).items():
                    if round_name == 'sweet_sixteen' and count > 0:
                        analytics['seed_success_rate'][seed]['sweet_sixteen'] += 1
                    elif round_name == 'elite_eight' and count > 0:
                        analytics['seed_success_rate'][seed]['elite_eight'] += 1
                    elif round_name == 'semifinals' and count > 0:
                        analytics['seed_success_rate'][seed]['final_four'] += 1
                    elif round_name == 'championship' and count > 0:
                        analytics['seed_success_rate'][seed]['championship'] += 1
                    elif round_name == 'champion' and count > 0:
                        analytics['seed_success_rate'][seed]['champion'] += 1
        
        # Convert to percentages
        for seed, data in analytics['seed_success_rate'].items():
            total = data['total_teams']
            if total > 0:
                for round_name in ['sweet_sixteen', 'elite_eight', 'final_four', 'championship', 'champion']:
                    data[round_name] = data[round_name] / total
        
        # Calculate conference performance
        conference_teams = {}
        conference_success = {}
        
        for team_name, stats in self.historical_results['team_stats'].items():
            if team_name in self.team_dict:
                conference = self.team_dict[team_name].get('Conference', '')
                if not conference:
                    continue
                
                if conference not in conference_teams:
                    conference_teams[conference] = 0
                    conference_success[conference] = {
                        'wins': 0,
                        'losses': 0,
                        'championships': 0,
                        'final_fours': 0
                    }
                
                conference_teams[conference] += 1
                conference_success[conference]['wins'] += stats.get('wins', 0)
                conference_success[conference]['losses'] += stats.get('losses', 0)
                conference_success[conference]['championships'] += stats.get('round_reached', {}).get('champion', 0)
                conference_success[conference]['final_fours'] += stats.get('round_reached', {}).get('semifinals', 0)
        
        # Calculate win percentage for each conference
        for conf, data in conference_success.items():
            total_games = data['wins'] + data['losses']
            if total_games > 0:
                data['win_pct'] = data['wins'] / total_games
            else:
                data['win_pct'] = 0
        
        analytics['conference_performance'] = conference_success
        
        # ML model accuracy
        if len(self.historical_results['training_data']['features']) >= 50:
            try:
                X = np.array(self.historical_results['training_data']['features'])
                y = np.array(self.historical_results['training_data']['outcomes'])
                
                # Use a simple train/test split for evaluation
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train and evaluate models
                model_accuracy = {}
                
                for model_name, model in self.ml_models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    model_accuracy[model_name] = accuracy
                
                analytics['ml_model_accuracy'] = model_accuracy
                
            except Exception as e:
                print(f"Error evaluating ML models: {e}")
        
        # Upset analysis
        analytics['upset_analysis'] = {
            'by_seed_diff': {},
            'by_round': {}
        }
        
        for seed_diff, data in self.historical_results['upset_tracker']['by_seed_diff'].items():
            if data['total'] > 0:
                analytics['upset_analysis']['by_seed_diff'][seed_diff] = data['occurred'] / data['total']
        
        for round_name, data in self.historical_results['upset_tracker']['by_round'].items():
            if data['total'] > 0:
                analytics['upset_analysis']['by_round'][round_name] = data['occurred'] / data['total']
        
        # Save analytics to file
        analytics_file = 'data/tournament_analytics.json'
        os.makedirs(os.path.dirname(analytics_file), exist_ok=True)
        
        try:
            with open(analytics_file, 'w') as f:
                json.dump(analytics, f, indent=2)
            print(f"Saved analytics to {analytics_file}")
        except Exception as e:
            print(f"Error saving analytics: {e}")
        
        # Generate visualizations if matplotlib is available
        try:
            self._generate_visualizations(analytics)
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    def _generate_visualizations(self, analytics):
        """Generate visualizations of the simulation analytics"""
        # Create a directory for visualizations
        viz_dir = 'data/visualizations'
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set up the plots
        plt.figure(figsize=(12, 8))
        
        # 1. Champion distribution
        plt.subplot(2, 2, 1)
        champion_data = sorted(analytics['champion_history'].items(), key=lambda x: x[1], reverse=True)[:10]
        teams = [item[0] for item in champion_data]
        championships = [item[1] for item in champion_data]
        
        bars = plt.bar(teams, championships)
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 Champions')
        plt.ylabel('Championships')
        plt.tight_layout()
        
        # Add seed numbers to the bars
        for i, team in enumerate(teams):
            if team in self.team_dict:
                seed = self.team_dict[team].get('SeedNum', 'N/A')
                plt.text(i, championships[i] + 0.1, f"Seed {seed}", ha='center')
        
        # 2. Seed success rates
        plt.subplot(2, 2, 2)
        seeds = sorted(analytics['seed_success_rate'].keys())
        sweet_16_rates = [analytics['seed_success_rate'][s]['sweet_sixteen'] for s in seeds]
        elite_8_rates = [analytics['seed_success_rate'][s]['elite_eight'] for s in seeds]
        final_4_rates = [analytics['seed_success_rate'][s]['final_four'] for s in seeds]
        
        plt.plot(seeds, sweet_16_rates, 'o-', label='Sweet 16')
        plt.plot(seeds, elite_8_rates, 's-', label='Elite 8')
        plt.plot(seeds, final_4_rates, '^-', label='Final Four')
        plt.xlabel('Seed')
        plt.ylabel('Success Rate')
        plt.title('Tournament Success by Seed')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Conference performance
        plt.subplot(2, 2, 3)
        conferences = sorted([c for c in analytics['conference_performance'].keys() if analytics['conference_performance'][c]['wins'] >= 10])
        win_pcts = [analytics['conference_performance'][c]['win_pct'] for c in conferences]
        
        bars = plt.bar(conferences, win_pcts)
        plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Win Percentage')
        plt.title('Conference Performance')
        plt.tight_layout()
        
        # 4. Upset rates by seed difference
        plt.subplot(2, 2, 4)
        seed_diffs = sorted([int(sd) for sd in analytics['upset_analysis']['by_seed_diff'].keys() if int(sd) >= 3])
        upset_rates = [analytics['upset_analysis']['by_seed_diff'][str(sd)] for sd in seed_diffs]
        
        plt.plot(seed_diffs, upset_rates, 'o-', color='purple')
        plt.xlabel('Seed Difference')
        plt.ylabel('Upset Rate')
        plt.title('Upset Probability by Seed Difference')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'tournament_analytics.png'), dpi=300)
        plt.close()
        
        # 5. Generate feature importance plot
        if self.feature_importance:
            plt.figure(figsize=(12, 8))
            
            # Sort features by importance
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            feature_names = [item[0] for item in sorted_features]
            importance_values = [item[1] for item in sorted_features]
            
            # Create horizontal bar chart
            bars = plt.barh(feature_names, importance_values, color='green')
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importance in Predicting Game Outcomes')
            plt.tight_layout()
            
            plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300)
            plt.close()
        
        # 6. Generate learning curve for ML models
        if 'ml_model_accuracy' in analytics and analytics['ml_model_accuracy']:
            # Plot history of ML model accuracy over time
            plt.figure(figsize=(10, 6))
            
            for model_name, accuracy in analytics['ml_model_accuracy'].items():
                plt.bar(model_name, accuracy)
            
            plt.ylabel('Accuracy')
            plt.title('ML Model Prediction Accuracy')
            plt.ylim(0.5, 1.0)  # Accuracy range from 0.5 (random) to 1.0 (perfect)
            
            plt.savefig(os.path.join(viz_dir, 'ml_accuracy.png'), dpi=300)
            plt.close()
    
    def get_team_details(self, team_name):
        """Get the details for a specific team"""
        if team_name in self.team_dict:
            return self.team_dict[team_name]
        return None
    
    def export_results(self, output_path):
        """Export the tournament results to a JSON file"""
        # Add team details to each matchup
        enriched_bracket = self._enrich_bracket_with_team_details(self.bracket)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(enriched_bracket, f, indent=2)
        
        return output_path
    
    def _enrich_bracket_with_team_details(self, bracket):
        """Add team details to each matchup in the bracket"""
        enriched = {
            'regions': {},
            'final_four': {
                'semifinals': [],
                'championship': None,
                'champion': None
            }
        }
        
        # Enrich regional games
        for region, rounds in bracket['regions'].items():
            enriched['regions'][region] = {}
            
            for round_name, matchups in rounds.items():
                if round_name == 'regional_champion':
                    enriched['regions'][region][round_name] = bracket['regions'][region][round_name]
                    continue
                
                if isinstance(matchups, list):
                    enriched['regions'][region][round_name] = []
                    for matchup in matchups:
                        if matchup:
                            enriched_matchup = {
                                'team1': {
                                    'name': matchup['team1'],
                                    'details': self.get_team_details(matchup['team1'])
                                },
                                'team2': {
                                    'name': matchup['team2'],
                                    'details': self.get_team_details(matchup['team2'])
                                },
                                'winner': matchup['winner'],
                                'score_margin': matchup.get('score_margin'),
                                'key_factors': matchup.get('key_factors', [])
                            }
                            enriched['regions'][region][round_name].append(enriched_matchup)
                        else:
                            enriched['regions'][region][round_name].append(None)
                else:
                    if matchups:
                        enriched['regions'][region][round_name] = {
                            'team1': {
                                'name': matchups['team1'],
                                'details': self.get_team_details(matchups['team1'])
                            },
                            'team2': {
                                'name': matchups['team2'],
                                'details': self.get_team_details(matchups['team2'])
                            },
                            'winner': matchups['winner'],
                            'score_margin': matchups.get('score_margin'),
                            'key_factors': matchups.get('key_factors', [])
                        }
                    else:
                        enriched['regions'][region][round_name] = None
        
        # Enrich Final Four
        for i, semifinal in enumerate(bracket['final_four']['semifinals']):
            if semifinal:
                enriched_semifinal = {
                    'team1': {
                        'name': semifinal['team1'],
                        'details': self.get_team_details(semifinal['team1'])
                    },
                    'team2': {
                        'name': semifinal['team2'],
                        'details': self.get_team_details(semifinal['team2'])
                    },
                    'winner': semifinal['winner'],
                    'score_margin': semifinal.get('score_margin'),
                    'key_factors': semifinal.get('key_factors', [])
                }
                enriched['final_four']['semifinals'].append(enriched_semifinal)
            else:
                enriched['final_four']['semifinals'].append(None)
        
        if bracket['final_four']['championship']:
            enriched['final_four']['championship'] = {
                'team1': {
                    'name': bracket['final_four']['championship']['team1'],
                    'details': self.get_team_details(bracket['final_four']['championship']['team1'])
                },
                'team2': {
                    'name': bracket['final_four']['championship']['team2'],
                    'details': self.get_team_details(bracket['final_four']['championship']['team2'])
                },
                'winner': bracket['final_four']['championship']['winner'],
                'score_margin': bracket['final_four']['championship'].get('score_margin'),
                'key_factors': bracket['final_four']['championship'].get('key_factors', [])
            }
        
        enriched['final_four']['champion'] = bracket['final_four']['champion']
        
        # Add simulation metadata
        enriched['metadata'] = {
            'simulation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'simulation_id': f"sim-{self.historical_results['simulation_count']}",
            'historical_simulations': self.historical_results['simulation_count'],
            'ml_model_used': self.active_model if self.historical_results['simulation_count'] >= 50 else 'heuristic',
            'dynamic_weights': self.dynamic_weights,
            'power_conferences': self.power_conferences
        }
        
        return enriched
    
    def run_predictive_analysis(self, num_simulations=1000):
        """
        Run multiple simulations to predict tournament outcomes with confidence intervals.
        
        Args:
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary of prediction results
        """
        print(f"Running {num_simulations} simulations for predictive analysis...")
        
        # Initialize prediction tracking
        predictions = {
            'champions': defaultdict(int),
            'final_four': defaultdict(int),
            'elite_eight': defaultdict(int),
            'sweet_sixteen': defaultdict(int),
            'upsets': defaultdict(list),
            'conference_champions': defaultdict(int),
            'average_seed_by_round': {
                'champion': [],
                'championship': [],
                'semifinals': [],
                'elite_eight': [],
                'sweet_sixteen': []
            }
        }
        
        # Run multiple simulations
        for i in range(num_simulations):
            # Reset the bracket for a fresh simulation
            self.bracket = self._initialize_bracket()
            
            # Run the simulation
            self.simulate_tournament()
            
            # Track champion
            champion = self.bracket['final_four']['champion']
            predictions['champions'][champion] += 1
            
            # Track conference of champion
            if champion in self.team_dict:
                champion_conf = self.team_dict[champion].get('Conference', '')
                if champion_conf:
                    predictions['conference_champions'][champion_conf] += 1
            
            # Track Final Four teams
            for semifinal in self.bracket['final_four']['semifinals']:
                if semifinal:
                    if semifinal['team1']:
                        predictions['final_four'][semifinal['team1']] += 1
                    if semifinal['team2']:
                        predictions['final_four'][semifinal['team2']] += 1
            
            # Track average seed by round
            if champion in self.team_dict:
                predictions['average_seed_by_round']['champion'].append(self.team_dict[champion].get('SeedNum', 0))
            
            # Track championship game participants
            championship = self.bracket['final_four']['championship']
            if championship:
                if championship['team1'] in self.team_dict:
                    predictions['average_seed_by_round']['championship'].append(
                        self.team_dict[championship['team1']].get('SeedNum', 0))
                if championship['team2'] in self.team_dict:
                    predictions['average_seed_by_round']['championship'].append(
                        self.team_dict[championship['team2']].get('SeedNum', 0))
            
            # Track Elite Eight and Sweet Sixteen teams and their seeds
            for region in self.regions:
                # Elite Eight
                elite_eight = self.bracket['regions'][region]['elite_eight']
                if elite_eight:
                    if elite_eight['team1']:
                        predictions['elite_eight'][elite_eight['team1']] += 1
                        if elite_eight['team1'] in self.team_dict:
                            predictions['average_seed_by_round']['elite_eight'].append(
                                self.team_dict[elite_eight['team1']].get('SeedNum', 0))
                    if elite_eight['team2']:
                        predictions['elite_eight'][elite_eight['team2']] += 1
                        if elite_eight['team2'] in self.team_dict:
                            predictions['average_seed_by_round']['elite_eight'].append(
                                self.team_dict[elite_eight['team2']].get('SeedNum', 0))
                
                # Sweet Sixteen
                sweet_sixteen = self.bracket['regions'][region]['sweet_sixteen']
                for matchup in sweet_sixteen:
                    if matchup:
                        if matchup['team1']:
                            predictions['sweet_sixteen'][matchup['team1']] += 1
                            if matchup['team1'] in self.team_dict:
                                predictions['average_seed_by_round']['sweet_sixteen'].append(
                                    self.team_dict[matchup['team1']].get('SeedNum', 0))
                        if matchup['team2']:
                            predictions['sweet_sixteen'][matchup['team2']] += 1
                            if matchup['team2'] in self.team_dict:
                                predictions['average_seed_by_round']['sweet_sixteen'].append(
                                    self.team_dict[matchup['team2']].get('SeedNum', 0))
            
            # Track upsets in first round
            for region in self.regions:
                first_round = self.bracket['regions'][region]['first_round']
                for matchup in first_round:
                    if matchup:
                        team1 = self.team_dict.get(matchup['team1'], {})
                        team2 = self.team_dict.get(matchup['team2'], {})
                        
                        seed1 = team1.get('SeedNum', 0)
                        seed2 = team2.get('SeedNum', 0)
                        
                        # Check for upset (lower seed beats higher seed)
                        if seed1 < seed2 and matchup['winner'] == matchup['team2']:
                            upset_key = f"{seed2} over {seed1}"
                            predictions['upsets'][upset_key].append((matchup['team2'], matchup['team1']))
                        elif seed2 < seed1 and matchup['winner'] == matchup['team1']:
                            upset_key = f"{seed1} over {seed2}"
                            predictions['upsets'][upset_key].append((matchup['team1'], matchup['team2']))
            
            # Progress update
            if (i + 1) % 100 == 0 or i == num_simulations - 1:
                print(f"Completed {i + 1}/{num_simulations} simulations")
        
        # Calculate averages and percentages
        results = {
            'champions': {},
            'final_four': {},
            'elite_eight': {},
            'sweet_sixteen': {},
            'most_likely_upsets': {},
            'conference_champions': {},
            'average_seed_by_round': {}
        }
        
        # Calculate champion probabilities
        for team, count in predictions['champions'].items():
            results['champions'][team] = {
                'probability': count / num_simulations,
                'seed': self.team_dict.get(team, {}).get('SeedNum', 'N/A'),
                'conference': self.team_dict.get(team, {}).get('Conference', 'N/A')
            }
        
        # Calculate Final Four probabilities
        for team, count in predictions['final_four'].items():
            results['final_four'][team] = {
                'probability': count / num_simulations,
                'seed': self.team_dict.get(team, {}).get('SeedNum', 'N/A'),
                'conference': self.team_dict.get(team, {}).get('Conference', 'N/A')
            }
        
        # Calculate Elite Eight probabilities
        for team, count in predictions['elite_eight'].items():
            results['elite_eight'][team] = {
                'probability': count / num_simulations,
                'seed': self.team_dict.get(team, {}).get('SeedNum', 'N/A'),
                'conference': self.team_dict.get(team, {}).get('Conference', 'N/A')
            }
        
        # Calculate Sweet Sixteen probabilities
        for team, count in predictions['sweet_sixteen'].items():
            results['sweet_sixteen'][team] = {
                'probability': count / num_simulations,
                'seed': self.team_dict.get(team, {}).get('SeedNum', 'N/A'),
                'conference': self.team_dict.get(team, {}).get('Conference', 'N/A')
            }
        
        # Calculate upset probabilities
        for upset_key, instances in predictions['upsets'].items():
            results['most_likely_upsets'][upset_key] = {
                'probability': len(instances) / num_simulations,
                'examples': instances[:5]  # Include a few examples
            }
        
        # Calculate conference champion probabilities
        for conf, count in predictions['conference_champions'].items():
            results['conference_champions'][conf] = {
                'probability': count / num_simulations
            }
        
        # Calculate average seed by round
        for round_name, seeds in predictions['average_seed_by_round'].items():
            if seeds:
                results['average_seed_by_round'][round_name] = {
                    'average': sum(seeds) / len(seeds),
                    'min': min(seeds),
                    'max': max(seeds)
                }
        
        # Sort results by probability
        for category in ['champions', 'final_four', 'elite_eight', 'sweet_sixteen']:
            results[category] = dict(sorted(
                results[category].items(), 
                key=lambda x: x[1]['probability'], 
                reverse=True
            ))
        
        # Export prediction results
        prediction_file = 'data/tournament_predictions.json'
        os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
        
        with open(prediction_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Predictive analysis complete. Results saved to {prediction_file}")
        
        # Create visualizations of predictions
        try:
            self._visualize_predictions(results, num_simulations)
        except Exception as e:
            print(f"Error creating prediction visualizations: {e}")
        
        return results
    
    def _visualize_predictions(self, results, num_simulations):
        """Create visualizations of tournament predictions"""
        viz_dir = 'data/visualizations'
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Champion Probabilities
        plt.figure(figsize=(12, 6))
        top_champions = list(results['champions'].items())[:10]  # Top 10 most likely champions
        
        teams = [t[0] for t in top_champions]
        probs = [t[1]['probability'] * 100 for t in top_champions]  # Convert to percentage
        seeds = [t[1]['seed'] for t in top_champions]
        
        bars = plt.bar(teams, probs, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Probability (%)')
        plt.title(f'Championship Probability (Based on {num_simulations} Simulations)')
        
        # Add seed numbers to the bars
        for i, (team, prob) in enumerate(zip(teams, probs)):
            seed = results['champions'][team]['seed']
            plt.text(i, prob + 0.5, f"Seed {seed}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'champion_predictions.png'), dpi=300)
        plt.close()
        
        # 2. Final Four Probabilities
        plt.figure(figsize=(14, 6))
        top_final_four = list(results['final_four'].items())[:15]  # Top 15 most likely Final Four teams
        
        teams = [t[0] for t in top_final_four]
        probs = [t[1]['probability'] * 100 for t in top_final_four]
        
        bars = plt.bar(teams, probs, color='lightgreen')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Probability (%)')
        plt.title(f'Final Four Probability (Based on {num_simulations} Simulations)')
        
        # Add seed numbers
        for i, team in enumerate(teams):
            seed = results['final_four'][team]['seed']
            plt.text(i, probs[i] + 0.5, f"Seed {seed}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'final_four_predictions.png'), dpi=300)
        plt.close()
        
        # 3. Upset Probabilities
        plt.figure(figsize=(10, 6))
        
        if results['most_likely_upsets']:
            top_upsets = list(results['most_likely_upsets'].items())[:8]  # Top 8 most likely upsets
            
            upset_labels = [u[0] for u in top_upsets]
            upset_probs = [u[1]['probability'] * 100 for u in top_upsets]
            
            bars = plt.barh(upset_labels, upset_probs, color='salmon')
            plt.xlabel('Probability (%)')
            plt.title(f'Most Likely First Round Upsets (Based on {num_simulations} Simulations)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'upset_predictions.png'), dpi=300)
            plt.close()
        
        # 4. Conference Champion Distribution
        plt.figure(figsize=(12, 6))
        
        conf_data = list(results['conference_champions'].items())
        conf_data.sort(key=lambda x: x[1]['probability'], reverse=True)
        
        confs = [c[0] for c in conf_data]
        conf_probs = [c[1]['probability'] * 100 for c in conf_data]
        
        colors = []
        for conf in confs:
            if conf in self.power_conferences and self.power_conferences[conf] > 1.2:
                colors.append('darkred')  # Highlight strong power conferences
            elif conf in self.power_conferences:
                colors.append('salmon')  # Other power conferences
            else:
                colors.append('lightblue')  # Non-power conferences
        
        bars = plt.bar(confs, conf_probs, color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Probability (%)')
        plt.title(f'Championship Probability by Conference (Based on {num_simulations} Simulations)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'conference_predictions.png'), dpi=300)
        plt.close()
        
        # 5. Average Seed by Round
        if results['average_seed_by_round']:
            plt.figure(figsize=(10, 6))
            
            rounds = ['sweet_sixteen', 'elite_eight', 'semifinals', 'championship', 'champion']
            round_labels = ['Sweet 16', 'Elite 8', 'Final Four', 'Championship', 'Champion']
            
            avg_seeds = []
            min_seeds = []
            max_seeds = []
            
            for r in rounds:
                if r in results['average_seed_by_round']:
                    avg_seeds.append(results['average_seed_by_round'][r]['average'])
                    min_seeds.append(results['average_seed_by_round'][r]['min'])
                    max_seeds.append(results['average_seed_by_round'][r]['max'])
                else:
                    avg_seeds.append(0)
                    min_seeds.append(0)
                    max_seeds.append(0)
            
            plt.plot(round_labels, avg_seeds, 'o-', label='Average Seed')
            plt.fill_between(round_labels, min_seeds, max_seeds, alpha=0.2, label='Min-Max Range')
            
            plt.ylabel('Seed Number')
            plt.title(f'Average Seed by Tournament Round (Based on {num_simulations} Simulations)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'seed_by_round.png'), dpi=300)
            plt.close()
    
    def generate_future_prediction(self, team1, team2, round_name='championship'):
        """
        Generate a prediction for a hypothetical future matchup.
        
        Args:
            team1: First team name
            team2: Second team name
            round_name: Tournament round for the matchup
            
        Returns:
            Dictionary with prediction details
        """
        # Ensure both teams exist in our data
        if team1 not in self.team_dict or team2 not in self.team_dict:
            return {
                'error': 'One or both teams not found in tournament data',
                'teams_found': [t for t in [team1, team2] if t in self.team_dict]
            }
        
        # Run multiple simulations of this specific matchup
        num_simulations = 1000
        team1_wins = 0
        team2_wins = 0
        score_margins = []
        key_factors_counter = defaultdict(int)
        
        for _ in range(num_simulations):
            winner, margin, key_factors = self.simulate_game(team1, team2, round_name)
            
            if winner == team1:
                team1_wins += 1
            else:
                team2_wins += 1
            
            score_margins.append(margin)
            
            for factor in key_factors:
                key_factors_counter[factor] += 1
        
        # Calculate probabilities and statistics
        team1_prob = team1_wins / num_simulations
        team2_prob = team2_wins / num_simulations
        
        avg_margin = sum(score_margins) / len(score_margins) if score_margins else 0
        
        # Find the most common key factors
        common_factors = sorted(key_factors_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare the prediction result
        prediction = {
            'matchup': f"{team1} vs {team2}",
            'round': round_name,
            'simulations': num_simulations,
            'team1': {
                'name': team1,
                'win_probability': team1_prob,
                'seed': self.team_dict[team1].get('SeedNum', 'N/A'),
                'conference': self.team_dict[team1].get('Conference', 'N/A')
            },
            'team2': {
                'name': team2,
                'win_probability': team2_prob,
                'seed': self.team_dict[team2].get('SeedNum', 'N/A'),
                'conference': self.team_dict[team2].get('Conference', 'N/A')
            },
            'average_margin': avg_margin,
            'key_factors': [{'factor': f[0], 'frequency': f[1]/num_simulations} for f in common_factors[:5]]
        }
        
        return prediction

def simulate_single_tournament(input_csv, output_json):
    """
    Run a single tournament simulation with the enhanced learning algorithm and export the results.
    
    Args:
        input_csv: Path to the CSV file containing team data
        output_json: Path to save the simulation results
        
    Returns:
        Path to the output file
    """
    simulator = AdvancedBasketballSimulator(input_csv)
    simulator.simulate_tournament()
    output_path = simulator.export_results(output_json)
    return output_path

def run_advanced_simulation(input_csv, output_dir='data', num_simulations=10, predictive_analysis=True):
    """
    Run multiple tournament simulations with learning between simulations.
    
    Args:
        input_csv: Path to the CSV file containing team data
        output_dir: Directory to save outputs
        num_simulations: Number of tournament simulations to run
        predictive_analysis: Whether to run a final predictive analysis
        
    Returns:
        Dictionary with simulation results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the simulator
    simulator = AdvancedBasketballSimulator(input_csv)
    
    # Run multiple simulations, learning from each
    for i in range(num_simulations):
        print(f"\nRunning simulation {i+1}/{num_simulations}...")
        
        # Run the simulation
        simulator.simulate_tournament()
        
        # Export results for this simulation
        output_json = os.path.join(output_dir, f"tournament_results_sim{i+1}.json")
        simulator.export_results(output_json)
        
        print(f"Simulation {i+1} completed. Results saved to {output_json}")
    
    # Export the final simulation as the main result
    final_output = os.path.join(output_dir, "tournament_results.json")
    simulator.export_results(final_output)
    
    # Run a comprehensive predictive analysis if requested
    if predictive_analysis:
        print("\nRunning predictive analysis...")
        prediction_results = simulator.run_predictive_analysis(num_simulations=1000)
        
        # Print the top champion predictions
        print("\nTop 5 Championship Predictions:")
        for i, (team, data) in enumerate(list(prediction_results['champions'].items())[:5]):
            print(f"{i+1}. {team} (Seed {data['seed']}): {data['probability']*100:.1f}%")
    
    return {
        'final_results': final_output,
        'simulator': simulator,
        'simulations_run': num_simulations
    }

if __name__ == "__main__":
    # Example usage
    input_csv = "data/tournament_teams.csv"
    
    # For a single simulation
    output_json = "data/tournament_results.json"
    results_file = simulate_single_tournament(input_csv, output_json)
    print(f"Tournament simulation completed. Results saved to {results_file}")
    
    # For advanced simulation with learning
    advanced_results = run_advanced_simulation(
        input_csv, 
        output_dir='data/advanced_simulation',
        num_simulations=10,
        predictive_analysis=True
    )
    print("Advanced simulation completed.")
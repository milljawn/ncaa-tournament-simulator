#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import pickle
from collections import defaultdict

class BasketballSimulator:
    def __init__(self, teams_data_path):
        """
        Initialize the basketball tournament simulator.
        
        Args:
            teams_data_path: Path to the CSV file containing team data
        """
        self.teams_df = pd.read_csv(teams_data_path)
        self.teams = self.teams_df.to_dict('records')
        self.team_dict = {team['Team']: team for team in self.teams}
        self.regions = sorted(self.teams_df['Region'].unique())
        
        # Group teams by region and seed
        region_teams = {}
        for region in self.regions:
            region_teams[region] = self.teams_df[self.teams_df['Region'] == region].sort_values('SeedNum').to_dict('records')
        self.region_teams = region_teams
        
        # Set up bracket structure
        self.bracket = self._initialize_bracket()
        
        # Load historical results if available
        self.historical_results = self._load_historical_results()
        
        # Define the statistic category weights (from most to least important)
        self.stat_weights = {
            'Tempo': 1.0,
            'TempoRank': 0.9,
            'ORating': 0.8,
            'DRating': 0.7,
            'ORank': 0.6,
            'DRank': 0.5,
            'NetRating': 0.4,
            'Seed': 0.3,
            'SeedNum': 0.2,
            'Record': 0.1,
            'Region': 0.05,
            'Team': 0.01,
            'Conference': 0.01
        }
    
    def _load_historical_results(self):
        """Load historical tournament results for learning."""
        history_file = 'data/tournament_history.pkl'
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading historical data: {e}")
                return self._initialize_historical_results()
        else:
            return self._initialize_historical_results()
    
    def _initialize_historical_results(self):
        """Initialize the historical results structure."""
        return {
            'matchups': defaultdict(lambda: defaultdict(int)),  # Store team vs team results
            'team_stats': defaultdict(lambda: {
                'wins': 0,
                'losses': 0,
                'round_reached': defaultdict(int)
            }),
            'simulation_count': 0
        }
    
    def _save_historical_results(self):
        """Save the historical results to disk."""
        history_file = 'data/tournament_history.pkl'
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        try:
            with open(history_file, 'wb') as f:
                pickle.dump(self.historical_results, f)
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
                        'winner': None
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
    
    def calculate_team_strength(self, team_name, round_name='first_round'):
        """
        Calculate team strength based on weighted statistics and historical performance.
        
        Args:
            team_name: The name of the team
            round_name: The current tournament round, affects the weighting
        """
        team = self.team_dict[team_name]
        
        # Base strength from weighted statistics
        strength = 0
        
        # Apply weights to each statistic (note: lower ranks are better)
        if 'Tempo' in team:
            strength += team['Tempo'] * self.stat_weights['Tempo']
        
        if 'TempoRank' in team and team['TempoRank'] is not None:
            # Convert rank (lower is better) to a positive contribution
            strength += (400 - team['TempoRank']) / 400 * self.stat_weights['TempoRank']
        
        if 'ORating' in team:
            strength += team['ORating'] * self.stat_weights['ORating'] / 100
        
        if 'DRating' in team:
            # Lower defensive rating is better
            strength += (120 - team['DRating']) * self.stat_weights['DRating'] / 100
        
        if 'ORank' in team and team['ORank'] is not None:
            # Convert rank to a positive contribution
            strength += (400 - team['ORank']) / 400 * self.stat_weights['ORank']
        
        if 'DRank' in team and team['DRank'] is not None:
            # Convert rank to a positive contribution
            strength += (400 - team['DRank']) / 400 * self.stat_weights['DRank']
        
        if 'NetRating' in team:
            # NetRating can be negative, so we adjust to ensure positive contribution for good teams
            strength += (team['NetRating'] + 50) * self.stat_weights['NetRating'] / 100
        
        if 'SeedNum' in team:
            # Lower seed is better, so invert
            strength += (17 - team['SeedNum']) * self.stat_weights['SeedNum'] / 16
        
        # Record contribution (win percentage)
        if 'Record' in team:
            try:
                w, l = map(int, team['Record'].split('-'))
                win_pct = w / (w + l)
                strength += win_pct * self.stat_weights['Record']
            except:
                pass
        
        # Conference strength could be factored in but requires additional data
        # For now, we'll use a small constant for conferences
        if 'Conference' in team:
            # Could be enhanced with conference strength data
            power_conferences = ['B12', 'SEC', 'B10', 'ACC', 'BE', 'MWC']
            conf_factor = 0.05 if team['Conference'] in power_conferences else 0.02
            strength += conf_factor * self.stat_weights['Conference']
        
        # Historical performance adjustment
        if self.historical_results['simulation_count'] > 0:
            team_history = self.historical_results['team_stats'].get(team_name, {'wins': 0, 'losses': 0})
            
            # Calculate win percentage from historical simulations
            total_games = team_history['wins'] + team_history['losses']
            if total_games > 0:
                historical_win_pct = team_history['wins'] / total_games
                
                # Weight historical performance more in later rounds
                round_weights = {
                    'first_round': 0.05,
                    'second_round': 0.1,
                    'sweet_sixteen': 0.15,
                    'elite_eight': 0.2,
                    'semifinals': 0.25,
                    'championship': 0.3
                }
                
                # Add historical performance contribution
                strength += historical_win_pct * round_weights.get(round_name, 0.1)
                
                # Add bonus for teams that historically go far in the tournament
                deep_run_bonus = 0
                for round_reached, count in team_history['round_reached'].items():
                    if round_reached in ['elite_eight', 'semifinals', 'championship', 'champion']:
                        round_value = {
                            'elite_eight': 0.05,
                            'semifinals': 0.1,
                            'championship': 0.15,
                            'champion': 0.2
                        }
                        deep_run_bonus += count * round_value[round_reached]
                
                # Cap the deep run bonus and add it to strength
                deep_run_bonus = min(deep_run_bonus, 0.3)
                strength += deep_run_bonus
        
        return strength
    
    def simulate_game(self, team1_name, team2_name, round_name='first_round'):
        """
        Simulate a single game between two teams, returning the winner.
        
        The simulation uses weighted statistics and historical learning to determine the winner.
        """
        # Handle None values
        if team1_name is None:
            return team2_name
        if team2_name is None:
            return team1_name
        
        # Calculate team strengths based on weighted statistics
        team1_strength = self.calculate_team_strength(team1_name, round_name)
        team2_strength = self.calculate_team_strength(team2_name, round_name)
        
        # Add randomness factor (more randomness in earlier rounds)
        randomness_factor = {
            'first_round': 0.3,
            'second_round': 0.25,
            'sweet_sixteen': 0.2,
            'elite_eight': 0.15,
            'semifinals': 0.1,
            'championship': 0.05
        }.get(round_name, 0.2)
        
        team1_strength += np.random.normal(0, randomness_factor * team1_strength)
        team2_strength += np.random.normal(0, randomness_factor * team2_strength)
        
        # Check historical head-to-head results if available
        if self.historical_results['simulation_count'] > 0:
            matchup_key = tuple(sorted([team1_name, team2_name]))
            matchup_history = self.historical_results['matchups'][matchup_key]
            
            if team1_name in matchup_history and team2_name in matchup_history:
                team1_wins = matchup_history[team1_name]
                team2_wins = matchup_history[team2_name]
                total_matchups = team1_wins + team2_wins
                
                if total_matchups > 0:
                    # Weight the historical matchup more in later rounds
                    history_weight = {
                        'first_round': 0.1,
                        'second_round': 0.15,
                        'sweet_sixteen': 0.2,
                        'elite_eight': 0.25,
                        'semifinals': 0.3,
                        'championship': 0.35
                    }.get(round_name, 0.15)
                    
                    # Adjust strengths based on head-to-head history
                    team1_advantage = (team1_wins / total_matchups - 0.5) * 2  # Range: -1 to 1
                    team1_strength += team1_advantage * history_weight * team1_strength
                    team2_strength -= team1_advantage * history_weight * team2_strength
        
        # Factor in seed-based upset potential
        team1 = self.team_dict[team1_name]
        team2 = self.team_dict[team2_name]
        
        seed_diff = abs(team1['SeedNum'] - team2['SeedNum'])
        if seed_diff > 0:
            higher_seed = team1 if team1['SeedNum'] < team2['SeedNum'] else team2
            lower_seed = team2 if higher_seed == team1 else team1
            
            # Calculate upset factor - decreases as the tournament progresses
            round_upset_factor = {
                'first_round': 0.8,
                'second_round': 0.6,
                'sweet_sixteen': 0.4,
                'elite_eight': 0.3,
                'semifinals': 0.2,
                'championship': 0.1
            }.get(round_name, 0.5)
            
            upset_factor = min(5, seed_diff) * np.random.random() * round_upset_factor
            
            # Apply upset adjustment
            if higher_seed['Team'] == team1_name:
                team2_strength += upset_factor
            else:
                team1_strength += upset_factor
        
        # Determine winner
        if team1_strength > team2_strength:
            return team1_name
        else:
            return team2_name
    
    def simulate_round(self, round_type, region=None):
        """
        Simulate a round of the tournament in a specific region or the Final Four.
        
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
                    winner = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                    matchup['winner'] = winner
                    
                    # Update historical results
                    self._update_historical_results(matchup['team1'], matchup['team2'], winner, round_type)
                    
                    # Set up the next round
                    if i % 2 == 0:
                        self.bracket['regions'][region][next_round][next_round_idx] = {
                            'team1': winner,
                            'team2': None,  # Will be filled by the next matchup
                            'winner': None
                        }
                    else:
                        self.bracket['regions'][region][next_round][next_round_idx]['team2'] = winner
                        next_round_idx += 1
                
            elif round_type == 'second_round':
                matchups = self.bracket['regions'][region][round_type]
                next_round = 'sweet_sixteen'
                next_round_idx = 0
                
                for i, matchup in enumerate(matchups):
                    winner = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                    matchup['winner'] = winner
                    
                    # Update historical results
                    self._update_historical_results(matchup['team1'], matchup['team2'], winner, round_type)
                    
                    # Set up the Sweet 16
                    if i % 2 == 0:
                        self.bracket['regions'][region][next_round][next_round_idx] = {
                            'team1': winner,
                            'team2': None,
                            'winner': None
                        }
                    else:
                        self.bracket['regions'][region][next_round][next_round_idx]['team2'] = winner
                        next_round_idx += 1
                
            elif round_type == 'sweet_sixteen':
                matchups = self.bracket['regions'][region][round_type]
                
                for i, matchup in enumerate(matchups):
                    winner = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                    matchup['winner'] = winner
                    
                    # Update historical results
                    self._update_historical_results(matchup['team1'], matchup['team2'], winner, round_type)
                    
                    # Set up the Elite 8
                    if i == 0:
                        self.bracket['regions'][region]['elite_eight'] = {
                            'team1': winner,
                            'team2': None,
                            'winner': None
                        }
                    else:
                        self.bracket['regions'][region]['elite_eight']['team2'] = winner
                
            elif round_type == 'elite_eight':
                matchup = self.bracket['regions'][region][round_type]
                winner = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                matchup['winner'] = winner
                
                # Update historical results
                self._update_historical_results(matchup['team1'], matchup['team2'], winner, round_type)
                
                # Set the regional champion
                self.bracket['regions'][region]['regional_champion'] = winner
                
                # Update team's round reached
                self._update_round_reached(winner, round_type)
                
                # Determine which Final Four spot to fill
                region_idx = self.regions.index(region)
                if region_idx == 0:
                    if not self.bracket['final_four']['semifinals'][0]:
                        self.bracket['final_four']['semifinals'][0] = {'team1': winner, 'team2': None, 'winner': None}
                    else:
                        self.bracket['final_four']['semifinals'][0]['team2'] = winner
                elif region_idx == 1:
                    if not self.bracket['final_four']['semifinals'][0]:
                        self.bracket['final_four']['semifinals'][0] = {'team1': None, 'team2': winner, 'winner': None}
                    else:
                        self.bracket['final_four']['semifinals'][0]['team2'] = winner
                elif region_idx == 2:
                    if not self.bracket['final_four']['semifinals'][1]:
                        self.bracket['final_four']['semifinals'][1] = {'team1': winner, 'team2': None, 'winner': None}
                    else:
                        self.bracket['final_four']['semifinals'][1]['team2'] = winner
                elif region_idx == 3:
                    if not self.bracket['final_four']['semifinals'][1]:
                        self.bracket['final_four']['semifinals'][1] = {'team1': None, 'team2': winner, 'winner': None}
                    else:
                        self.bracket['final_four']['semifinals'][1]['team2'] = winner
        
        else:  # Final Four
            if round_type == 'semifinals':
                for i, matchup in enumerate(self.bracket['final_four']['semifinals']):
                    if matchup and matchup['team1'] and matchup['team2']:
                        winner = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                        matchup['winner'] = winner
                        
                        # Update historical results
                        self._update_historical_results(matchup['team1'], matchup['team2'], winner, round_type)
                        
                        # Update team's round reached
                        self._update_round_reached(winner, round_type)
                        
                        # Set up the championship game
                        if i == 0:
                            self.bracket['final_four']['championship'] = {
                                'team1': winner,
                                'team2': None,
                                'winner': None
                            }
                        else:
                            if self.bracket['final_four']['championship']:
                                self.bracket['final_four']['championship']['team2'] = winner
                            else:
                                self.bracket['final_four']['championship'] = {
                                    'team1': None,
                                    'team2': winner,
                                    'winner': None
                                }
            
            elif round_type == 'championship':
                matchup = self.bracket['final_four']['championship']
                if matchup and matchup['team1'] and matchup['team2']:
                    winner = self.simulate_game(matchup['team1'], matchup['team2'], round_type)
                    matchup['winner'] = winner
                    
                    # Update historical results
                    self._update_historical_results(matchup['team1'], matchup['team2'], winner, round_type)
                    
                    # Update team's round reached
                    self._update_round_reached(winner, 'champion')
                    
                    # Set the champion
                    self.bracket['final_four']['champion'] = winner
    
    def _update_historical_results(self, team1, team2, winner, round_name):
        """Update historical results with the outcome of a game."""
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
    
    def _update_round_reached(self, team, round_name):
        """Update the furthest round reached by a team."""
        self.historical_results['team_stats'][team]['round_reached'][round_name] += 1
    
    def simulate_tournament(self):
        """Simulate the entire tournament from start to finish."""
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
        self.historical_results['simulation_count'] += 1
        
        # Save the updated historical results
        self._save_historical_results()
        
        return self.bracket
    
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
                                'winner': matchup['winner']
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
                            'winner': matchups['winner']
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
                    'winner': semifinal['winner']
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
                'winner': bracket['final_four']['championship']['winner']
            }
        
        enriched['final_four']['champion'] = bracket['final_four']['champion']
        
        # Add simulation metadata
        enriched['metadata'] = {
            'simulation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'simulation_id': np.random.randint(1000000, 9999999),
            'historical_simulations': self.historical_results['simulation_count']
        }
        
        return enriched

def simulate_single_tournament(input_csv, output_json):
    """Run a single tournament simulation and export the results"""
    simulator = BasketballSimulator(input_csv)
    simulator.simulate_tournament()
    output_path = simulator.export_results(output_json)
    return output_path

if __name__ == "__main__":
    # Example usage
    input_csv = "data/tournament_teams.csv"
    output_json = "data/tournament_results.json"
    
    results_file = simulate_single_tournament(input_csv, output_json)
    print(f"Tournament simulation completed. Results saved to {results_file}")
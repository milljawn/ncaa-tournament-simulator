#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

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
    
    def simulate_game(self, team1_name, team2_name):
        """
        Simulate a single game between two teams, returning the winner.
        
        The simulation uses offensive and defensive ratings, along with some randomness,
        to determine the winner.
        """
        team1 = self.team_dict[team1_name]
        team2 = self.team_dict[team2_name]
        
        # Use the net ratings and add some randomness
        team1_strength = team1['NetRating'] + np.random.normal(0, 5)  # Add some noise
        team2_strength = team2['NetRating'] + np.random.normal(0, 5)
        
        # Incorporate upset probability based on seed differences
        seed_diff = abs(team1['SeedNum'] - team2['SeedNum'])
        
        # Lower seeds have a better chance of upsets in closer seed matchups
        if seed_diff > 0:
            higher_seed = team1 if team1['SeedNum'] < team2['SeedNum'] else team2
            lower_seed = team2 if higher_seed == team1 else team1
            
            # Calculate upset factor - decreases as the tournament progresses
            upset_factor = min(5, seed_diff) * np.random.random() * 2
            
            # Apply upset adjustment
            if higher_seed == team1:
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
                    winner = self.simulate_game(matchup['team1'], matchup['team2'])
                    matchup['winner'] = winner
                    
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
                    winner = self.simulate_game(matchup['team1'], matchup['team2'])
                    matchup['winner'] = winner
                    
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
                    winner = self.simulate_game(matchup['team1'], matchup['team2'])
                    matchup['winner'] = winner
                    
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
                winner = self.simulate_game(matchup['team1'], matchup['team2'])
                matchup['winner'] = winner
                
                # Set the regional champion
                self.bracket['regions'][region]['regional_champion'] = winner
                
                # Determine which Final Four spot to fill
                region_idx = self.regions.index(region)
                if region_idx < 2:
                    self.bracket['final_four']['semifinals'][0] = {
                        'team1': winner if region_idx == 0 else None,
                        'team2': winner if region_idx == 1 else self.bracket['final_four']['semifinals'][0]['team1'] if self.bracket['final_four']['semifinals'][0] else None,
                        'winner': None
                    }
                else:
                    self.bracket['final_four']['semifinals'][1] = {
                        'team1': winner if region_idx == 2 else None,
                        'team2': winner if region_idx == 3 else self.bracket['final_four']['semifinals'][1]['team1'] if self.bracket['final_four']['semifinals'][1] else None,
                        'winner': None
                    }
        
        else:  # Final Four
            if round_type == 'semifinals':
                for i, matchup in enumerate(self.bracket['final_four']['semifinals']):
                    winner = self.simulate_game(matchup['team1'], matchup['team2'])
                    matchup['winner'] = winner
                    
                    # Set up the championship game
                    if i == 0:
                        self.bracket['final_four']['championship'] = {
                            'team1': winner,
                            'team2': None,
                            'winner': None
                        }
                    else:
                        self.bracket['final_four']['championship']['team2'] = winner
            
            elif round_type == 'championship':
                matchup = self.bracket['final_four']['championship']
                winner = self.simulate_game(matchup['team1'], matchup['team2'])
                matchup['winner'] = winner
                
                # Set the champion
                self.bracket['final_four']['champion'] = winner
    
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
            'simulation_id': np.random.randint(1000000, 9999999)
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
    input_csv = "tournament_teams.csv"
    output_json = "tournament_results.json"
    
    results_file = simulate_single_tournament(input_csv, output_json)
    print(f"Tournament simulation completed. Results saved to {results_file}")
#!/usr/bin/env python3
import pandas as pd
from bs4 import BeautifulSoup
import os
import re
import numpy as np

def parse_kenpom_html(html_file):
    """
    Parse the KenPom HTML file and extract relevant team data.
    """
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Find the ratings table
    table = soup.find('table', {'id': 'ratings-table'})
    
    # Extract data
    teams_data = []
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        if len(cells) >= 21:  # Ensure we have enough columns
            rank = cells[0].text.strip()
            team_cell = cells[1]
            team_name = team_cell.find('a').text.strip()
            
            # Check if there's a seed number
            seed_span = team_cell.find('span', {'class': 'seed'})
            seed = seed_span.text.strip() if seed_span else 'NIT'
            
            conference = cells[2].find('a').text.strip()
            record = cells[3].text.strip()
            
            # Extract efficiency metrics
            net_rating = cells[4].text.strip()
            o_rating = cells[5].text.strip()
            o_rank = cells[6].find('span', {'class': 'seed'}).text.strip() if cells[6].find('span', {'class': 'seed'}) else 'N/A'
            d_rating = cells[7].text.strip()
            d_rank = cells[8].find('span', {'class': 'seed'}).text.strip() if cells[8].find('span', {'class': 'seed'}) else 'N/A'
            tempo = cells[9].text.strip()
            tempo_rank = cells[10].find('span', {'class': 'seed'}).text.strip() if cells[10].find('span', {'class': 'seed'}) else 'N/A'
            
            teams_data.append({
                'Rank': int(rank),
                'Team': team_name,
                'Seed': seed if seed != 'NIT' else None,
                'Conference': conference,
                'Record': record,
                'NetRating': float(net_rating),
                'ORating': float(o_rating),
                'ORank': int(o_rank) if o_rank != 'N/A' else None,
                'DRating': float(d_rating),
                'DRank': int(d_rank) if d_rank != 'N/A' else None,
                'Tempo': float(tempo),
                'TempoRank': int(tempo_rank) if tempo_rank != 'N/A' else None
            })
    
    return pd.DataFrame(teams_data)

def select_tournament_teams(df):
    """
    Select the 64 teams that would make the NCAA tournament.
    In a real scenario, this would be based on selection committee criteria,
    but for simplicity, we'll use KenPom rankings with some seed logic.
    """
    # First, select teams with actual seeds (these are automatically in)
    seeded_teams = df[df['Seed'].notna()].copy()
    
    # Convert seeds to integers
    seeded_teams['SeedNum'] = seeded_teams['Seed'].astype(int)
    
    # Count how many teams we have per seed
    seed_counts = seeded_teams['SeedNum'].value_counts().to_dict()
    
    # Create a list to track which seeds we need more teams for
    needed_seeds = []
    for seed in range(1, 17):  # NCAA tournament seeds 1-16
        needed = 4 - seed_counts.get(seed, 0)  # Each seed should have 4 teams
        if needed > 0:
            needed_seeds.extend([seed] * needed)
    
    # Select additional teams based on ranking
    unseeded_teams = df[df['Seed'].isna()].copy()
    additional_teams = unseeded_teams.head(len(needed_seeds)).copy()
    
    # Assign seeds to additional teams
    additional_teams['SeedNum'] = needed_seeds[:len(additional_teams)]
    
    # Combine seeded and additional teams
    tournament_teams = pd.concat([seeded_teams, additional_teams])
    
    # Ensure we have exactly 64 teams
    tournament_teams = tournament_teams.head(64)
    
    # Sort by seed and then by ranking within seed
    tournament_teams = tournament_teams.sort_values(['SeedNum', 'Rank'])
    
    return tournament_teams

def create_regions(tournament_teams):
    """
    Divide the 64 teams into 4 regions (East, West, South, Midwest).
    Each region gets 16 teams, with seeds 1-16.
    """
    regions = ['East', 'West', 'South', 'Midwest']
    
    # Group teams by seed
    seed_groups = tournament_teams.groupby('SeedNum')
    
    # Create an empty DataFrame to hold region assignments
    regional_teams = []
    
    # For each seed 1-16, distribute the 4 teams across the 4 regions
    for seed in range(1, 17):
        if seed in seed_groups.groups:
            seed_teams = seed_groups.get_group(seed).copy()
            
            # Make sure we have exactly 4 teams for this seed
            # If we have fewer, duplicate the last team to fill in
            while len(seed_teams) < 4:
                seed_teams = pd.concat([seed_teams, seed_teams.tail(1)])
            
            # Take only the first 4 teams for this seed
            seed_teams = seed_teams.head(4)
            
            # Assign regions to these teams
            seed_teams['Region'] = regions
            
            regional_teams.append(seed_teams)
    
    # Combine all teams with their region assignments
    regional_df = pd.concat(regional_teams)
    
    # Sort by region and seed for a clean bracket view
    regional_df = regional_df.sort_values(['Region', 'SeedNum'])
    
    return regional_df

def save_tournament_data(df, output_file):
    """
    Save the tournament team data to a CSV file.
    """
    df.to_csv(output_file, index=False)
    print(f"Tournament data saved to {output_file}")

def main():
    html_file = '2025PomeroyCollegeBasketballRatings.html'
    output_file = 'tournament_teams.csv'
    
    # Parse the KenPom data
    df = parse_kenpom_html(html_file)
    print(f"Parsed {len(df)} teams from KenPom ratings")
    
    # Select tournament teams
    tournament_teams = select_tournament_teams(df)
    print(f"Selected {len(tournament_teams)} teams for the tournament")
    
    # Create regions
    regional_teams = create_regions(tournament_teams)
    print(f"Divided teams into regions: {', '.join(regional_teams['Region'].unique())}")
    
    # Save tournament data
    save_tournament_data(regional_teams, output_file)

if __name__ == "__main__":
    main()
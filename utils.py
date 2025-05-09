#!/usr/bin/env python3
from functools import wraps
import re
import unicodedata

def register_jinja_filters(app):
    """Register custom Jinja2 filters for the Flask app."""
    
    @app.template_filter('slugify')
    def slugify(value):
        """
        Convert a string to a URL-friendly slug.
        
        Example:
            "North Carolina State" -> "north-carolina-state"
        """
        # Normalize unicode characters
        value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode('ascii')
        # Convert to lowercase
        value = value.lower()
        # Replace spaces with hyphens
        value = re.sub(r'[^\w\s-]', '', value).strip()
        value = re.sub(r'[\s-]+', '-', value)
        return value
    
    @app.template_filter('seed')
    def get_team_seed(team_name, results):
        """Get the seed number for a team."""
        # Check all regions
        for region_name, region_data in results['regions'].items():
            # Check all rounds for the team
            for round_name, matchups in region_data.items():
                if round_name == 'regional_champion':
                    continue  # Skip the regional champion, which is just a string
                
                if isinstance(matchups, list):
                    for matchup in matchups:
                        if matchup and matchup['team1']['name'] == team_name:
                            return matchup['team1']['details']['SeedNum']
                        if matchup and matchup['team2']['name'] == team_name:
                            return matchup['team2']['details']['SeedNum']
                elif matchups:
                    if matchups['team1']['name'] == team_name:
                        return matchups['team1']['details']['SeedNum']
                    if matchups['team2']['name'] == team_name:
                        return matchups['team2']['details']['SeedNum']
        
        # Check Final Four
        for semifinal in results['final_four']['semifinals']:
            if semifinal:
                if semifinal['team1']['name'] == team_name:
                    return semifinal['team1']['details']['SeedNum']
                if semifinal['team2']['name'] == team_name:
                    return semifinal['team2']['details']['SeedNum']
        
        if results['final_four']['championship']:
            if results['final_four']['championship']['team1']['name'] == team_name:
                return results['final_four']['championship']['team1']['details']['SeedNum']
            if results['final_four']['championship']['team2']['name'] == team_name:
                return results['final_four']['championship']['team2']['details']['SeedNum']
        
        # If we get here, we couldn't find the team
        return 'N/A'
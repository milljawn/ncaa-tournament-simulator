document.addEventListener('DOMContentLoaded', function() {
    // Get the simulate button
    const simulateButton = document.getElementById('simulate-button');
    
    // Add loading spinner to the body
    const loadingSpinner = document.createElement('div');
    loadingSpinner.className = 'loading-spinner';
    document.body.appendChild(loadingSpinner);
    
    // Add click event listener to the simulate button
    simulateButton.addEventListener('click', function() {
        // Show loading state
        document.body.classList.add('loading');
        
        // Send request to the server
        fetch('/simulate', {
            method: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Update the bracket with the new results
            updateBracket(data);
            
            // Remove loading state
            document.body.classList.remove('loading');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during simulation. Please try again.');
            
            // Remove loading state
            document.body.classList.remove('loading');
        });
    });
    
    // Function to update the bracket with new results
    function updateBracket(results) {
        // Rather than manually updating every element,
        // we'll just reload the page to get the new results
        location.reload();
    }
    
    // Add tooltip functionality for team info
    const teamElements = document.querySelectorAll('.team');
    
    teamElements.forEach(team => {
        team.addEventListener('mouseover', function() {
            // Get team name
            const teamName = this.querySelector('.name').textContent;
            
            // Check if tooltip already exists
            if (!this.querySelector('.team-tooltip')) {
                // Create tooltip
                const tooltip = document.createElement('div');
                tooltip.className = 'team-tooltip';
                
                // Fetch team data
                fetch(`/team-info?name=${encodeURIComponent(teamName)}`)
                    .then(response => response.json())
                    .then(data => {
                        // Populate tooltip
                        tooltip.innerHTML = `
                            <p>Record: ${data.Record}</p>
                            <p>Conference: ${data.Conference}</p>
                            <p>KenPom Rank: ${data.Rank}</p>
                            <p>ORtg: ${data.ORating} (${data.ORank})</p>
                            <p>DRtg: ${data.DRating} (${data.DRank})</p>
                        `;
                    })
                    .catch(error => {
                        console.error('Error fetching team info:', error);
                        tooltip.textContent = 'Error loading team info';
                    });
                
                this.appendChild(tooltip);
            }
        });
        
        team.addEventListener('mouseout', function() {
            // Remove tooltip
            const tooltip = this.querySelector('.team-tooltip');
            if (tooltip) {
                tooltip.remove();
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Press 'R' to run a new simulation
        if (event.key === 'r' || event.key === 'R') {
            simulateButton.click();
        }
    });
});
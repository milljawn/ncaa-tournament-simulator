document.addEventListener('DOMContentLoaded', function() {
    // Get the simulate button
    const simulateButton = document.getElementById('simulate-button');
    const resetButton = document.getElementById('reset-button');
    
    // Add loading spinner to the body
    const loadingSpinner = document.createElement('div');
    loadingSpinner.className = 'loading-spinner';
    document.body.appendChild(loadingSpinner);
    
    // Initialize the bracket modifications tracking
    let bracketModifications = {
        teamSwaps: []
    };
    
    // Set up drag and drop functionality
    setupDragAndDrop();
    
    // Add click event listener to the simulate button
    simulateButton.addEventListener('click', function() {
        // Show loading state
        document.body.classList.add('loading');
        
        // Send request to the server with the modifications
        fetch('/simulate', {
            method: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ modifications: bracketModifications })
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
            
            // Reset modifications tracking
            bracketModifications.teamSwaps = [];
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during simulation. Please try again.');
            
            // Remove loading state
            document.body.classList.remove('loading');
        });
    });
    
    // Add reset button functionality
    resetButton.addEventListener('click', function() {
        if (confirm('Are you sure you want to reset the bracket to its original state?')) {
            window.location.reload();
        }
    });
    
    // Function to set up drag and drop functionality
    function setupDragAndDrop() {
        const draggableElements = document.querySelectorAll('.draggable');
        let draggedElement = null;
        
        draggableElements.forEach(element => {
            // Drag start
            element.addEventListener('dragstart', function(e) {
                draggedElement = this;
                this.classList.add('dragging');
                
                // Store the original matchup and position
                e.dataTransfer.setData('text/plain', JSON.stringify({
                    teamId: this.getAttribute('data-team-id'),
                    teamSeed: this.getAttribute('data-team-seed'),
                    region: this.getAttribute('data-region'),
                    position: this.getAttribute('data-position'),
                    matchupIndex: this.getAttribute('data-matchup-index')
                }));
            });
            
            // Drag end
            element.addEventListener('dragend', function() {
                this.classList.remove('dragging');
                draggedElement = null;
                
                // Remove any drag-over classes
                document.querySelectorAll('.drag-over').forEach(el => {
                    el.classList.remove('drag-over');
                });
            });
            
            // Drag over
            element.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.classList.add('drag-over');
            });
            
            // Drag leave
            element.addEventListener('dragleave', function() {
                this.classList.remove('drag-over');
            });
            
            // Drop
            element.addEventListener('drop', function(e) {
                e.preventDefault();
                this.classList.remove('drag-over');
                
                if (draggedElement !== this) {
                    // Get the dragged team data
                    const draggedData = JSON.parse(e.dataTransfer.getData('text/plain'));
                    
                    // Get the target team data
                    const targetData = {
                        teamId: this.getAttribute('data-team-id'),
                        teamSeed: this.getAttribute('data-team-seed'),
                        region: this.getAttribute('data-region'),
                        position: this.getAttribute('data-position'),
                        matchupIndex: this.getAttribute('data-matchup-index')
                    };
                    
                    // Only allow swapping teams in the first round
                    if (this.closest('.round-of-64') && draggedElement.closest('.round-of-64')) {
                        swapTeams(draggedElement, this, draggedData, targetData);
                        
                        // Track this swap for the server
                        bracketModifications.teamSwaps.push({
                            team1: {
                                name: draggedElement.querySelector('.name').textContent,
                                region: draggedData.region,
                                position: draggedData.position,
                                matchupIndex: parseInt(draggedData.matchupIndex)
                            },
                            team2: {
                                name: this.querySelector('.name').textContent,
                                region: targetData.region,
                                position: targetData.position,
                                matchupIndex: parseInt(targetData.matchupIndex)
                            }
                        });
                    }
                }
            });
        });
    }
    
    // Function to swap two teams
    function swapTeams(team1Element, team2Element, team1Data, team2Data) {
        // Swap the name content
        const team1Name = team1Element.querySelector('.name').textContent;
        const team2Name = team2Element.querySelector('.name').textContent;
        team1Element.querySelector('.name').textContent = team2Name;
        team2Element.querySelector('.name').textContent = team1Name;
        
        // Swap the seed numbers
        const team1Seed = team1Element.querySelector('.seed').textContent;
        const team2Seed = team2Element.querySelector('.seed').textContent;
        team1Element.querySelector('.seed').textContent = team2Seed;
        team2Element.querySelector('.seed').textContent = team1Seed;
        
        // Update data attributes
        team1Element.setAttribute('data-team-id', team2Data.teamId);
        team2Element.setAttribute('data-team-id', team1Data.teamId);
        
        team1Element.setAttribute('data-team-seed', team2Data.teamSeed);
        team2Element.setAttribute('data-team-seed', team1Data.teamSeed);
        
        // Remove any winner classes as they're no longer relevant
        team1Element.classList.remove('winner');
        team2Element.classList.remove('winner');
        
        // Clear any subsequent rounds (they should be recalculated)
        clearSubsequentRounds();
    }
    
    // Function to clear subsequent rounds after the first round
    function clearSubsequentRounds() {
        const subsequentRounds = document.querySelectorAll('.round-of-32, .sweet-16, .elite-8, .final-four-qualifier, .final-four');
        
        subsequentRounds.forEach(round => {
            const teams = round.querySelectorAll('.team');
            teams.forEach(team => {
                // Keep the team divs but clear their content
                const seedElement = team.querySelector('.seed');
                const nameElement = team.querySelector('.name');
                
                if (seedElement) seedElement.textContent = '?';
                if (nameElement) nameElement.textContent = 'TBD';
                
                // Remove any winner class
                team.classList.remove('winner');
            });
        });
    }
    
    // Function to update the bracket with new results
    function updateBracket(results) {
        location.reload();
    }
    
    // Add tooltip functionality for team info
    const teamElements = document.querySelectorAll('.team');
    
    teamElements.forEach(team => {
        team.addEventListener('mouseover', function(e) {
            // Avoid handling hover on drag handles
            if (e.target.classList.contains('drag-handle')) return;
            
            // Get team name
            const teamName = this.querySelector('.name').textContent;
            
            // Don't create tooltips for TBD teams
            if (teamName === 'TBD') return;
            
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
        // Press 'Escape' to reset the bracket
        if (event.key === 'Escape') {
            resetButton.click();
        }
    });
});
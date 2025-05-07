// Function to create a rating distribution chart
function createRatingDistributionChart(chartData) {
    const ctx = document.getElementById('ratingDistributionChart').getContext('2d');
    
    // Get ratings and counts
    const ratings = Object.keys(chartData).map(Number).sort();
    const counts = ratings.map(rating => chartData[rating]);
    
    // Create chart
    const ratingChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ratings,
            datasets: [{
                label: 'Number of Ratings',
                data: counts,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(255, 159, 64, 0.7)',
                    'rgba(255, 205, 86, 0.7)',
                    'rgba(75, 192, 192, 0.7)',
                    'rgba(54, 162, 235, 0.7)'
                ],
                borderColor: [
                    'rgb(255, 99, 132)',
                    'rgb(255, 159, 64)',
                    'rgb(255, 205, 86)',
                    'rgb(75, 192, 192)',
                    'rgb(54, 162, 235)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Rating Distribution',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat().format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Ratings'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Rating Value'
                    }
                }
            }
        }
    });
}

// Function to load dataset statistics from API
function loadDatasetStats() {
    fetch('/api/dataset_stats')
        .then(response => response.json())
        .then(data => {
            if (data.rating_distribution) {
                createRatingDistributionChart(data.rating_distribution);
            }
        })
        .catch(error => {
            console.error('Error fetching dataset stats:', error);
        });
}

// Function to handle sample ID click
function handleSampleIdClick(inputId) {
    return function(e) {
        const sampleId = e.target.textContent;
        document.getElementById(inputId).value = sampleId;
    };
}

// Initialize page-specific scripts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set up click handlers for sample IDs
    const sampleUserElements = document.querySelectorAll('.sample-user-id');
    sampleUserElements.forEach(element => {
        element.addEventListener('click', handleSampleIdClick('user_id'));
    });
    
    const sampleProductElements = document.querySelectorAll('.sample-product-id');
    sampleProductElements.forEach(element => {
        element.addEventListener('click', handleSampleIdClick('product_id'));
    });
    
    // Set up click handlers for deep learning sample IDs
    const dlSampleUserElements = document.querySelectorAll('.dl-sample-user-id');
    dlSampleUserElements.forEach(element => {
        element.addEventListener('click', handleSampleIdClick('user_id'));
    });
    
    // Create charts if on the dataset_stats page
    if (document.getElementById('ratingDistributionChart')) {
        loadDatasetStats();
    }
    
    // Set up form submission loading indicator
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const spinner = document.querySelector('.loading-spinner');
            if (spinner) {
                spinner.style.display = 'block';
            }
        });
    });
});

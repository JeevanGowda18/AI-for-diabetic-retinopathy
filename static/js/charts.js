/**
 * DR Vision AI - Charts & Visualizations
 * Powered by Chart.js
 */

function initDashboardCharts(gradeCounts, trendData) {
    // 1. Severity Distribution Doughnut Chart
    const ctxDist = document.getElementById('severityDoughnutChart');
    if (ctxDist) {
        new Chart(ctxDist, {
            type: 'doughnut',
            data: {
                labels: ['Grade 0 (No DR)', 'Grade 1 (Mild)', 'Grade 2 (Moderate)', 'Grade 3 (Severe)', 'Grade 4 (PDR)'],
                datasets: [{
                    data: [
                        gradeCounts[0] || 0,
                        gradeCounts[1] || 0,
                        gradeCounts[2] || 0,
                        gradeCounts[3] || 0,
                        gradeCounts[4] || 0
                    ],
                    backgroundColor: [
                        '#10B981', // Emerald Normal
                        '#0284C7', // Sky Mild
                        '#F59E0B', // Amber Moderate
                        '#EF4444', // Red Severe
                        '#8B5CF6'  // Purple PDR
                    ],
                    borderWidth: 2,
                    borderColor: '#FFFFFF'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 12,
                            font: { size: 11, family: 'Inter' }
                        }
                    }
                },
                cutout: '68%'
            }
        });
    }

    // 2. Activity Trends Line Chart
    const ctxTrend = document.getElementById('activityTrendChart');
    if (ctxTrend && trendData) {
        const labels = trendData.map(d => d.date);
        const counts = trendData.map(d => d.count);

        new Chart(ctxTrend, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Screenings',
                    data: counts,
                    borderColor: '#0D9488',
                    backgroundColor: 'rgba(13, 148, 136, 0.12)',
                    fill: true,
                    tension: 0.35,
                    pointBackgroundColor: '#0D9488',
                    pointRadius: 4,
                    borderWidth: 2.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 1, precision: 0 },
                        grid: { color: '#F1F5F9' }
                    },
                    x: {
                        grid: { display: false }
                    }
                }
            }
        });
    }
}

function initProbabilityChart(canvasId, probabilities) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [
                'Grade 0: No DR',
                'Grade 1: Mild NPDR',
                'Grade 2: Mod NPDR',
                'Grade 3: Sev NPDR',
                'Grade 4: PDR'
            ],
            datasets: [{
                label: 'Prediction Probability',
                data: probabilities.map(p => (p * 100).toFixed(1)),
                backgroundColor: [
                    '#10B981',
                    '#0284C7',
                    '#F59E0B',
                    '#EF4444',
                    '#8B5CF6'
                ],
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => ` ${context.raw}% confidence`
                    }
                }
            },
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    ticks: {
                        callback: (val) => `${val}%`
                    },
                    grid: { color: '#F1F5F9' }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

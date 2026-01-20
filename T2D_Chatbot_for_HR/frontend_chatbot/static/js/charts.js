```javascript
// Chart utilities and configurations

// Default Plotly layout
const defaultPlotlyLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
        family: 'Inter, sans-serif',
        size: 12
    },
    margin: { t: 40, r: 20, b: 40, l: 50 }
};

// Default Chart.js options
const defaultChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: true,
            position: 'bottom'
        }
    }
};

// Create radar chart
function createRadarChart(elementId, data, labels) {
    const trace = {
        type: 'scatterpolar',
        r: data,
        theta: labels,
        fill: 'toself',
        fillcolor: 'rgba(59, 130, 246, 0.3)',
        line: {
            color: 'rgb(59, 130, 246)',
            width: 2
        }
    };

    const layout = {
        ...defaultPlotlyLayout,
        polar: {
            radialaxis: {
                visible: true,
                range: [0, 5]
            }
        },
        showlegend: false
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true });
}

// Create line chart
function createLineChart(elementId, xData, yData, title = '') {
    const trace = {
        x: xData,
        y: yData,
        type: 'scatter',
        mode: 'lines+markers',
        line: {
            color: 'rgb(139, 92, 246)',
            width: 3
        },
        marker: {
            size: 8,
            color: 'rgb(139, 92, 246)'
        }
    };

    const layout = {
        ...defaultPlotlyLayout,
        title: title,
        xaxis: { title: 'Période' },
        yaxis: { title: 'Score Moyen', range: [0, 5] }
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true });
}

// Create bar chart
function createBarChart(elementId, xData, yData, title = '', orientation = 'v') {
    const trace = {
        x: orientation === 'v' ? xData : yData,
        y: orientation === 'v' ? yData : xData,
        type: 'bar',
        orientation: orientation,
        marker: {
            color: 'rgb(59, 130, 246)'
        }
    };

    const layout = {
        ...defaultPlotlyLayout,
        title: title
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true });
}

// Create pie chart
function createPieChart(elementId, labels, values, title = '') {
    const trace = {
        labels: labels,
        values: values,
        type: 'pie',
        hole: 0.4,
        marker: {
            colors: ['#10b981', '#ef4444', '#f59e0b', '#6b7280']
        }
    };

    const layout = {
        ...defaultPlotlyLayout,
        title: title
    };

    Plotly.newPlot(elementId, [trace], layout, { responsive: true });
}

// Export functions
window.createRadarChart = createRadarChart;
window.createLineChart = createLineChart;
window.createBarChart = createBarChart;
window.createPieChart = createPieChart;
document.addEventListener("DOMContentLoaded", function () {
    // Mental Health Data
    const mentalHealthData = [
        { day: "Mon", score: 65 },
        { day: "Tue", score: 59 },
        { day: "Wed", score: 80 },
        { day: "Thu", score: 81 },
        { day: "Fri", score: 56 },
        { day: "Sat", score: 55 },
        { day: "Sun", score: 40 },
    ];

    // Populate Recommended Programs
    const recommendedPrograms = [
        { name: "Guided Meditation", duration: "10 min" },
        { name: "Stress Management", duration: "15 min" },
        { name: "Sleep Improvement", duration: "20 min" },
    ];

    const programsList = document.getElementById("programs-list");
    recommendedPrograms.forEach(program => {
        let li = document.createElement("li");
        li.innerHTML = `<strong>${program.name}</strong> <span>(${program.duration})</span>`;

        programsList.appendChild(li);
    });

    // Render Chart
    const ctx = document.getElementById("mentalHealthChart").getContext("2d");
    new Chart(ctx, {
        type: "bar",
        data: {
            labels: mentalHealthData.map(d => d.day),
            datasets: [{
                label: "Mental Health Score",
                data: mentalHealthData.map(d => d.score),
                backgroundColor: "#8884d8"
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });

    // Update Score on Button Click
    document.getElementById("new-assessment").addEventListener("click", function () {
        const newScore = Math.floor(Math.random() * 100);
        document.getElementById("score").textContent = newScore;
        document.getElementById("progress-bar").value = newScore;
    });
});

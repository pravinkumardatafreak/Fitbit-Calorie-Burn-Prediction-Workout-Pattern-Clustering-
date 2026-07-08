/* 
   Fitbit Dashboard Frontend Controller
   Integrates sliding inputs, performs API fetches, and triggers animations.
*/

document.addEventListener("DOMContentLoaded", () => {
    // Form and input elements
    const form = document.getElementById("workout-form");
    const genderSelect = document.getElementById("gender");
    const workoutTypeSelect = document.getElementById("workout_type");
    
    // Sliders
    const weightSlider = document.getElementById("weight");
    const heightSlider = document.getElementById("height");
    const avgBpmSlider = document.getElementById("avg_bpm");
    const sessionDurationSlider = document.getElementById("session_duration");
    const hrIntensitySlider = document.getElementById("hr_intensity");
    
    // Slider badges
    const weightVal = document.getElementById("weight-val");
    const heightVal = document.getElementById("height-val");
    const avgBpmVal = document.getElementById("avg_bpm-val");
    const sessionDurationVal = document.getElementById("session_duration-val");
    const hrIntensityVal = document.getElementById("hr_intensity-val");

    // Advanced section elements
    const toggleAdvancedBtn = document.getElementById("toggle-advanced");
    const advancedFields = document.getElementById("advanced-fields");
    const restingBpmInput = document.getElementById("resting_bpm");
    const fatPercentageInput = document.getElementById("fat_percentage");
    const waterIntakeInput = document.getElementById("water_intake");
    const workoutFrequencyInput = document.getElementById("workout_frequency");
    const experienceLevelSelect = document.getElementById("experience_level");

    // Output DOM elements
    const calorieNum = document.getElementById("calorie-num");
    const calorieProgressBar = document.getElementById("calorie-progress");
    const calorieExplanation = document.getElementById("calorie-explanation");
    const zoneBadge = document.getElementById("zone-badge");
    const zoneDescription = document.getElementById("zone-description");
    const bmiVal = document.getElementById("bmi-val");
    const pcaCoordsVal = document.getElementById("pca-coords");

    // State
    let currentCalorieVal = 0;
    let debounceTimer = null;

    // Initialize slider values
    updateSliderBadges();

    // Toggle advanced section
    toggleAdvancedBtn.addEventListener("click", () => {
        toggleAdvancedBtn.classList.toggle("active");
        advancedFields.classList.toggle("hidden");
        const chevron = toggleAdvancedBtn.querySelector("span");
        if (toggleAdvancedBtn.classList.contains("active")) {
            chevron.textContent = "▼";
        } else {
            chevron.textContent = "▶";
        }
    });

    // Listen to changes across all inputs
    form.addEventListener("input", () => {
        updateSliderBadges();
        // Debounce API calls by 100ms to avoid overwhelming local server during rapid sliding
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(calculateMetabolicMetrics, 100);
    });

    // Update textual badges next to range inputs
    function updateSliderBadges() {
        weightVal.textContent = weightSlider.value + " kg";
        heightVal.textContent = parseFloat(heightSlider.value).toFixed(2) + " m";
        avgBpmVal.textContent = avgBpmSlider.value + " BPM";
        sessionDurationVal.textContent = sessionDurationSlider.value + " min";
        hrIntensityVal.textContent = parseFloat(hrIntensitySlider.value).toFixed(2);
    }

    // Call predictions and cluster APIs
    async function calculateMetabolicMetrics() {
        const payload = {
            gender: genderSelect.value,
            weight: parseFloat(weightSlider.value),
            height: parseFloat(heightSlider.value),
            avg_bpm: parseFloat(avgBpmSlider.value),
            resting_bpm: parseFloat(restingBpmInput.value),
            session_duration: parseFloat(sessionDurationSlider.value) / 60, // Convert minutes to hours
            fat_percentage: parseFloat(fatPercentageInput.value),
            water_intake: parseFloat(waterIntakeInput.value),
            workout_frequency: parseInt(workoutFrequencyInput.value),
            experience_level: parseInt(experienceLevelSelect.value),
            workout_type: workoutTypeSelect.value,
            hr_intensity: parseFloat(hrIntensitySlider.value)
        };

        // Local calculations (BMI)
        const bmi = payload.weight / (payload.height ** 2);
        bmiVal.textContent = bmi.toFixed(1);

        try {
            // Call both API endpoints in parallel
            const [predictRes, clusterRes] = await Promise.all([
                fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                }),
                fetch("/api/cluster", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                })
            ]);

            if (!predictRes.ok || !clusterRes.ok) {
                throw new Error("Failed to receive metrics from model microservices");
            }

            const predictData = await predictRes.json();
            const clusterData = await clusterRes.json();

            // 1. Update Calorie Burn display with roll-up animation
            animateCalorieRollup(predictData.predicted_calories);
            calorieExplanation.textContent = predictData.explanation;

            // Update calorie progress bar (scale relative to 1000 kcal max)
            const percentage = Math.min((predictData.predicted_calories / 1000) * 100, 100);
            calorieProgressBar.style.width = percentage + "%";

            // 2. Update Workout Cluster Zone styling and information
            const zone = clusterData.zone;
            zoneBadge.textContent = zone.name;
            zoneBadge.style.color = "#ffffff";
            zoneBadge.style.backgroundColor = zone.color;
            zoneBadge.style.boxShadow = `0 4px 15px ${zone.color}40`;
            zoneBadge.style.borderColor = zone.color;
            
            // Adjust card header glow
            const zoneCard = document.querySelector(".zone-card");
            zoneCard.style.borderTopColor = zone.color;
            
            zoneDescription.textContent = zone.description;

            // Display coordinates
            pcaCoordsVal.textContent = `${clusterData.pca_x.toFixed(2)}, ${clusterData.pca_y.toFixed(2)}`;

        } catch (error) {
            console.error("API error:", error);
            calorieExplanation.textContent = "Server offline. Ensure uvicorn is running.";
            zoneDescription.textContent = "Unable to classify metabolic zone.";
            pcaCoordsVal.textContent = "Error";
            bmiVal.textContent = "--";
        }
    }

    // Number roll-up effect for smooth premium feel
    function animateCalorieRollup(targetVal) {
        const start = currentCalorieVal;
        const end = targetVal;
        if (start === end) return;

        const duration = 300; // ms
        const startTime = performance.now();

        function updateNumber(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Ease out quad formula
            const easeProgress = progress * (2 - progress);
            const val = start + (end - start) * easeProgress;
            
            calorieNum.textContent = Math.round(val);

            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            } else {
                calorieNum.textContent = Math.round(end);
                currentCalorieVal = end;
            }
        }

        requestAnimationFrame(updateNumber);
    }

    // Trigger initial calculation on page load
    calculateMetabolicMetrics();
});

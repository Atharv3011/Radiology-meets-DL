document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImage = document.getElementById('previewImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultContainer = document.getElementById('resultContainer');
    const resultMessage = document.getElementById('resultMessage');
    const progressBar = document.querySelector('.progress-bar');

    let selectedFile = null;

    // =============================
    // Preview uploaded image
    // =============================
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = function(event) {
                previewImage.src = event.target.result;
                imagePreview.style.display = 'block';
                analyzeBtn.disabled = false;
                resultContainer.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    });

    // =============================
    // Call Flask backend
    // =============================
    analyzeBtn.addEventListener('click', async function() {
        if (!selectedFile) {
            alert("Please select an image first!");
            return;
        }

        // Show loading spinner
        const btn = this;
        btn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Analyzing...
        `;
        btn.disabled = true;

        let formData = new FormData();
        formData.append("image", selectedFile); // ✅ must match Flask key

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error("Server error: " + response.status);
            }

            const result = await response.json();
            console.log(result);

            resultContainer.style.display = 'block';
            const confidencePercent = result.confidence.toFixed(2);

            if (result.fracture_detected) {
                resultMessage.className = "alert alert-danger";
                resultMessage.innerHTML = `
                    <strong><i class="fas fa-times-circle me-2"></i> Fracture Detected!</strong><br>
                    <b>Type:</b> ${result.fracture_type}<br>
                    <b>Confidence:</b> ${confidencePercent}%
                `;
                progressBar.className = "progress-bar bg-danger";
            } else {
                resultMessage.className = "alert alert-success";
                resultMessage.innerHTML = `
                    <strong><i class="fas fa-check-circle me-2"></i> No Fracture Detected.</strong><br>
                    <b>Confidence:</b> ${confidencePercent}%
                `;
                progressBar.className = "progress-bar bg-success";
            }

            progressBar.style.width = `${confidencePercent}%`;
            progressBar.innerText = `${confidencePercent}% Confidence`;

            resultContainer.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error("Error:", error);
            alert("⚠️ Error connecting to backend. Make sure Flask is running on port 5500.");
        }

        // Reset button
        btn.innerHTML = '<i class="fas fa-search me-2"></i> Analyze for Fractures';
        btn.disabled = false;
    });
});

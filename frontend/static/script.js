document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const inputArea = document.getElementById("features-input");

    // UI Elements
    const btnText = document.querySelector(".btn-text");
    const spinner = document.getElementById("loading-spinner");
    const checkBtn = document.getElementById("check-fraud-btn");

    // Result Elements
    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultContent = document.getElementById("result-content");
    const predictionBadge = document.getElementById("prediction-badge");
    const probPercentage = document.getElementById("prob-percentage");
    const probFill = document.getElementById("prob-fill");
    const detailsBox = document.getElementById("details-box");

    // Demo Normal Data (Class 0 from creditcard.csv approx)
    const demoNormal = "0, -1.359, -0.0727, 2.536, 1.378, -0.338, 0.462, 0.239, 0.098, 0.363, 0.09, -0.551, -0.617, -0.991, -0.311, 1.468, -0.47, 0.207, 0.025, 0.403, 0.251, -0.018, 0.277, -0.11, 0.066, 0.128, -0.189, 0.133, -0.021, 149.62";

    // Demo Fraud Data (Class 1 approx)
    const demoFraud = "406, -2.312, 1.951, -1.609, 3.997, -0.522, -1.426, -2.537, 1.391, -2.77, -2.772, 3.202, -2.899, -0.595, -4.289, 0.389, -1.14, -2.815, -1.268, 0.416, 0.126, 0.517, -0.035, -0.465, 0.32, 0.044, 0.177, 0.261, -0.143, 0.0";

    // Auto-fill logic based on URL params
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('autofill') === 'true') {
        const card = urlParams.get('card');
        const name = urlParams.get('name');
        const cvc = urlParams.get('cvc');
        const ip = urlParams.get('ip');

        if (card) {
            // Map payment details into the 30-feature space
            inputArea.value = generateFeaturesFromCard(card, cvc, name, ip);

            // Show a visual confirmation metric
            const subtitle = document.querySelector(".subtitle");
            if (subtitle) {
                let ipDisplay = ip ? ` from IP <strong>${ip}</strong>` : "";
                subtitle.innerHTML = `Features synthetically generated mapping payment data from Card ending in <strong>${card.slice(-4)}</strong>${ipDisplay}.`;
                subtitle.style.color = "var(--primary-color, #4361ee)";
                subtitle.style.fontWeight = "600";
            }
        } else {
            inputArea.value = demoNormal; // Fallback
        }
    }

    function generateFeaturesFromCard(cardNumber, cvc, name, ip) {
        const seedString = (cardNumber || "") + (cvc || "") + (name || "") + (ip || "");

        let hash = 0;
        for (let i = 0; i < seedString.length; i++) {
            hash = ((hash << 5) - hash) + seedString.charCodeAt(i);
            hash |= 0;
        }

        const seededRandom = function () {
            hash = Math.sin(hash) * 10000;
            return hash - Math.floor(hash);
        }

        seededRandom();

        // Randomly select Normal or Fraud
        const isFraud = Math.random() > 0.5;
        const baseFeatures = (isFraud ? demoFraud : demoNormal).split(",").map(Number);

        const features = [];
        for (let i = 0; i < 30; i++) {
            if (i === 0) {
                features.push(Math.floor(Math.abs(seededRandom()) * 170000));
            } else if (i === 29) {
                features.push((Math.abs(seededRandom()) * 500).toFixed(2));
            } else {
                const perturbation = (seededRandom() * 1.5) - 0.75;
                features.push((baseFeatures[i] + perturbation).toFixed(4));
            }
        }
        return features.join(", ");
    }

    document.getElementById("fill-demo-btn").addEventListener("click", () => {
        inputArea.value = demoNormal;
    });

    document.getElementById("fill-demo-fraud").addEventListener("click", () => {
        inputArea.value = demoFraud;
    });

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Parse input
        const rawInput = inputArea.value;
        const featureStrArray = rawInput.split(",").map(item => item.trim());

        if (featureStrArray.length !== 30) {
            alert(`Please provide exactly 30 numerical features separated by commas. You provided ${featureStrArray.length}.`);
            return;
        }

        const features = featureStrArray.map(item => parseFloat(item));
        if (features.some(isNaN)) {
            alert("All features must be valid numerical values.");
            return;
        }

        // Show loading state
        setLoading(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ features: features, payer_ip: urlParams.get('ip') || "Unknown IP" })
            });

            const data = await response.json();

            if (response.ok) {
                renderResult(data.prediction, data.fraud_probability);
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error(error);
            alert("Failed to connect to the prediction server.");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        checkBtn.disabled = isLoading;
        if (isLoading) {
            btnText.classList.add("hidden");
            spinner.classList.remove("hidden");
            resultPlaceholder.classList.remove("hidden");
            resultContent.classList.add("hidden");
        } else {
            btnText.classList.remove("hidden");
            spinner.classList.add("hidden");
        }
    }

    function renderResult(prediction, probability) {
        resultPlaceholder.classList.add("hidden");
        resultContent.classList.remove("hidden");

        const probPct = (probability * 100).toFixed(2);
        probPercentage.innerText = `${probPct}%`;
        probFill.style.width = `${probPct}%`;

        // Reset classes
        predictionBadge.className = "status-badge";
        probFill.style.backgroundColor = "";

        if (prediction === "Fraud") {
            predictionBadge.textContent = "FRAUD DETECTED";
            predictionBadge.classList.add("fraud");
            probFill.style.backgroundColor = "var(--danger-color)";
            detailsBox.innerHTML = `
                <p style="color: var(--danger-color); font-weight:600;">High Risk Transaction Warning</p>
                <p>The deployed Neural Network model strongly flags this transaction as fraudulent. Recommend immediate blockage and verification.</p>
            `;
            detailsBox.style.borderLeftColor = "var(--danger-color)";
        } else {
            predictionBadge.textContent = "NORMAL TRANSACTION";
            predictionBadge.classList.add("normal");
            probFill.style.backgroundColor = "var(--success-color)";
            detailsBox.innerHTML = `
                <p style="color: var(--success-color); font-weight:600;">Safe Transaction</p>
                <p>The transaction parameters match normal behavior patterns. No suspicious activities detected within the deep learning architecture.</p>
            `;
            detailsBox.style.borderLeftColor = "var(--success-color)";
        }
    }
});

// ============================================================================
// IMDB Sentiment Analysis
// Front-End Controller
// ============================================================================

// ============================================================================
// Elements
// ============================================================================

const reviewInput = document.getElementById("review");

const predictButton = document.getElementById("predict-button");
const buttonText = document.getElementById("button-text");
const loadingSpinner = document.getElementById("loading-spinner");

const resultCard = document.getElementById("result-card");

const sentimentBadge = document.getElementById("sentiment-badge");
const sentimentResult = document.getElementById("sentiment-result");

const confidenceResult = document.getElementById("confidence-result");
const confidenceFill = document.getElementById("confidence-fill");

const labelResult = document.getElementById("label-result");
const modelResult = document.getElementById("model-result");
const timeResult = document.getElementById("time-result");

const charCounter = document.getElementById("char-count");

const positiveExampleButton =
    document.getElementById("positive-example");

const negativeExampleButton =
    document.getElementById("negative-example");

const clearButton =
    document.getElementById("clear-review");

const copyButton =
    document.getElementById("copy-result");

const errorBox =
    document.getElementById("error-box");

const successBox =
    document.getElementById("success-box");

const modelButtons =
    document.querySelectorAll(".model-option");

// ============================================================================
// State
// ============================================================================

let selectedModel = "logistic_regression";

let predictionRunning = false;

// ============================================================================
// Initial State
// ============================================================================

initialize();

// ============================================================================

function initialize() {

    hideMessages();

    hideResult();

    hideSpinner();

    charCounter.textContent =
        reviewInput.value.length;

}

// ============================================================================
// Character Counter
// ============================================================================

reviewInput.addEventListener("input", () => {

    charCounter.textContent =
        reviewInput.value.length;

});

// ============================================================================
// Model Selection
// ============================================================================

modelButtons.forEach(button => {

    button.addEventListener("click", () => {

        if (predictionRunning) {

            return;

        }

        modelButtons.forEach(item => {

            item.classList.remove("active");

        });

        button.classList.add("active");

        selectedModel =
            button.dataset.model;

    });

});

// ============================================================================
// Example Reviews
// ============================================================================

positiveExampleButton.addEventListener("click", () => {

    reviewInput.value =
        "This movie was absolutely amazing. The acting was brilliant, the story was engaging, and I enjoyed every minute of it.";

    charCounter.textContent =
        reviewInput.value.length;

    hideMessages();

});

negativeExampleButton.addEventListener("click", () => {

    reviewInput.value =
        "This was one of the worst movies I have ever seen. The plot was boring, the acting was terrible, and it was a complete waste of time.";

    charCounter.textContent =
        reviewInput.value.length;

    hideMessages();

});

// ============================================================================
// Clear Review
// ============================================================================

clearButton.addEventListener("click", () => {

    reviewInput.value = "";

    charCounter.textContent = 0;

    hideMessages();

    hideResult();

});

// ============================================================================
// Copy Result
// ============================================================================

copyButton.addEventListener("click", async () => {

    const text =

`Sentiment : ${sentimentResult.textContent}

Confidence : ${confidenceResult.textContent}

Model : ${modelResult.textContent}

Prediction Time : ${timeResult.textContent}`;

    try {

        await navigator.clipboard.writeText(text);

        showSuccess(
            "Result copied successfully."
        );

    }

    catch {

        showError(
            "Cannot copy the result."
        );

    }

});

// ============================================================================
// Message Functions
// ============================================================================

function hideMessages() {

    errorBox.classList.add("hidden");

    successBox.classList.add("hidden");

}

function showError(message) {

    successBox.classList.add("hidden");

    errorBox.textContent = message;

    errorBox.classList.remove("hidden");

}

function showSuccess(message) {

    errorBox.classList.add("hidden");

    successBox.textContent = message;

    successBox.classList.remove("hidden");

}

// ============================================================================
// Spinner
// ============================================================================

function showSpinner() {

    loadingSpinner.classList.remove("hidden");

}

function hideSpinner() {

    loadingSpinner.classList.add("hidden");

}

// ============================================================================
// Loading State
// ============================================================================

function setLoading(isLoading) {

    predictionRunning = isLoading;

    predictButton.disabled = isLoading;

    if (isLoading) {

        buttonText.textContent = "Predicting...";

        showSpinner();

    }

    else {

        buttonText.textContent = "Predict Sentiment";

        hideSpinner();

    }

}

// ============================================================================
// Result Card
// ============================================================================

function hideResult() {

    resultCard.classList.add("hidden");

}

function showResultCard() {

    resultCard.classList.remove("hidden");

}

// ============================================================================
// Confidence Bar
// ============================================================================

function updateConfidence(confidence) {

    confidenceResult.textContent =
        `${confidence}%`;

    confidenceFill.style.width =
        `${confidence}%`;

    if (confidence >= 80) {

        confidenceFill.style.background =
            "linear-gradient(90deg,#22c55e,#16a34a)";

    }

    else if (confidence >= 60) {

        confidenceFill.style.background =
            "linear-gradient(90deg,#eab308,#ca8a04)";

    }

    else {

        confidenceFill.style.background =
            "linear-gradient(90deg,#ef4444,#dc2626)";

    }

}

// ============================================================================
// Sentiment Badge
// ============================================================================

function updateSentiment(sentiment) {

    sentimentBadge.classList.remove(

        "positive",
        "negative"

    );

    sentimentBadge.classList.add(

        sentiment

    );

    sentimentResult.textContent =

        sentiment.charAt(0).toUpperCase() +
        sentiment.slice(1);

}

// ============================================================================
// Fill Result
// ============================================================================

function renderPrediction(data) {

    updateSentiment(
        data.sentiment
    );

    updateConfidence(
        data.confidence
    );

    labelResult.textContent =
        data.label;

    modelResult.textContent =
        data.model;

    timeResult.textContent =
        `${data.prediction_time} sec`;

    showResultCard();

    resultCard.scrollIntoView({

        behavior: "smooth",

        block: "start"

    });

}

// ============================================================================
// Validation
// ============================================================================

function validateReview() {

    const review =

        reviewInput.value.trim();

    if (review.length === 0) {

        showError(
            "Please enter a review."
        );

        return null;

    }

    return review;

}

// ============================================================================
// Request Builder
// ============================================================================

function createPayload(review) {

    return {

        review: review,

        model: selectedModel

    };

}

// ============================================================================
// Fetch Prediction
// ============================================================================

async function requestPrediction(payload) {

    const response = await fetch(

        "/predict",

        {

            method: "POST",

            headers: {

                "Content-Type":
                    "application/json"

            },

            body: JSON.stringify(
                payload
            )

        }

    );

    const data =

        await response.json();

    if (!response.ok) {

        throw new Error(

            data.error ||

            "Prediction failed."

        );

    }

    return data;

}

// ============================================================================
// Predict Button
// ============================================================================

predictButton.addEventListener("click", async () => {


    hideMessages();

    hideResult();

    hideSpinner();

    const review = validateReview();

    if (review === null) {

        return;

    }

    setLoading(true);

    try {

        const payload = createPayload(review);

        const result = await requestPrediction(payload);

        renderPrediction(result);

        showSuccess(
            "Prediction completed successfully."
        );

    }

    catch (error) {

        showError(
            error.message ||
            "Prediction failed."
        );

    }

    finally {

        setLoading(false);

    }

});

// ============================================================================
// Hide Messages While User Is Typing
// ============================================================================

reviewInput.addEventListener("focus", () => {

    hideMessages();

});

reviewInput.addEventListener("keydown", () => {

    hideMessages();

});

// ============================================================================
// Keyboard Shortcut (Ctrl + Enter)
// ============================================================================

reviewInput.addEventListener("keydown", event => {

    if (

        event.ctrlKey &&
        event.key === "Enter"

    ) {

        event.preventDefault();

        predictButton.click();

    }

});

// ============================================================================
// Prevent Double Click
// ============================================================================

predictButton.addEventListener("dblclick", event => {

    event.preventDefault();

});
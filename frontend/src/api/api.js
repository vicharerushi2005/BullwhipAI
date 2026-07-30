const API_URL = "http://127.0.0.1:8000";

async function fetchJSON(endpoint) {
    const response = await fetch(`${API_URL}${endpoint}`);

    if (!response.ok) {
        throw new Error(`Failed to fetch ${endpoint}`);
    }

    return response.json();
}

export function getStatus() {
    return fetchJSON("/");
}

export function getPrediction() {
    return fetchJSON("/prediction");
}

export function getSummary() {
    return fetchJSON("/summary");
}

export function getExplanation() {
    return fetchJSON("/explanation");
}

export function getRecommendation() {
    return fetchJSON("/recommendation");
}

export function getInventory() {
    return fetchJSON("/inventory");
}
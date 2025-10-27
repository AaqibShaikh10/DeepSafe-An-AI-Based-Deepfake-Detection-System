/**
 * DeepSafe Frontend JavaScript
 * Handles file upload, validation, API communication, and results display
 */

// Global Variables
const apiBaseUrl = '/api/';
let currentFile = null;
let isProcessing = false;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const analyzeButton = document.getElementById('analyzeButton');
const analyzeAudioCheckbox = document.getElementById('analyzeAudioCheckbox');
const analyzeAudioContainer = document.getElementById('analyzeAudioContainer');
const buttonText = document.getElementById('buttonText');
const loadingSpinner = document.getElementById('loadingSpinner');
const progressText = document.getElementById('progressText');
const selectedFileInfo = document.getElementById('selectedFileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // File input change event
    fileInput.addEventListener('change', handleFileSelect);

    // Analyze button click event
    analyzeButton.addEventListener('click', handleAnalyzeClick);

    // Drag and drop events
    const uploadArea = document.querySelector('.upload-area');
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
});

/**
 * Handle file selection from file input
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processSelectedFile(file);
    }
}

/**
 * Handle drag over event
 */
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.style.borderColor = '#764ba2';
}

/**
 * Handle drag leave event
 */
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.style.borderColor = '#667eea';
}

/**
 * Handle drop event
 */
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.style.borderColor = '#667eea';

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        processSelectedFile(files[0]);
    }
}

/**
 * Process selected file
 */
function processSelectedFile(file) {
    currentFile = file;

    // Validate file
    const validation = validateFile(file);

    if (!validation.valid) {
        displayError({ error: validation.error, error_type: 'validation_error' });
        analyzeButton.disabled = true;
        selectedFileInfo.style.display = 'none';
        return;
    }

    // Display file info
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    selectedFileInfo.style.display = 'block';

    // Show audio checkbox for video files
    const fileExt = getFileExtension(file.name);
    const videoExtensions = ['.mp4', '.avi', '.mov', '.webm'];
    if (videoExtensions.includes(fileExt)) {
        analyzeAudioContainer.style.display = 'block';
    } else {
        analyzeAudioContainer.style.display = 'none';
        analyzeAudioCheckbox.checked = false;
    }

    // Enable analyze button
    analyzeButton.disabled = false;

    // Hide previous results and errors
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

/**
 * Handle analyze button click
 */
function handleAnalyzeClick() {
    if (!currentFile || isProcessing) {
        return;
    }

    analyzeFile();
}

/**
 * Analyze the selected file
 */
async function analyzeFile() {
    isProcessing = true;

    // Update UI to show loading state
    analyzeButton.disabled = true;
    buttonText.textContent = 'Analyzing...';
    loadingSpinner.style.display = 'inline-block';
    progressText.style.display = 'block';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';

    try {
        // Create FormData
        const formData = new FormData();

        // Determine file type and endpoint
        const fileExt = getFileExtension(currentFile.name);
        const videoExtensions = ['.mp4', '.avi', '.mov', '.webm'];
        const audioExtensions = ['.mp3', '.wav', '.m4a', '.ogg'];

        let endpoint;
        if (videoExtensions.includes(fileExt)) {
            formData.append('video', currentFile);
            endpoint = apiBaseUrl + 'detect/video/';

            // Add analyze_audio parameter if checkbox is checked
            if (analyzeAudioCheckbox.checked) {
                formData.append('analyze_audio', 'true');
            }
        } else if (audioExtensions.includes(fileExt)) {
            formData.append('audio', currentFile);
            endpoint = apiBaseUrl + 'detect/audio/';
        } else {
            throw new Error('Unsupported file type');
        }

        // Get CSRF token
        const csrfToken = document.getElementById('csrfToken').value;

        // Make API request
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': csrfToken
            }
        });

        const data = await response.json();

        if (response.ok && data.success) {
            displayResults(data);
        } else {
            displayError(data);
        }

    } catch (error) {
        displayError({
            error: 'Network error - please check your connection and try again',
            error_type: 'network_error'
        });
    } finally {
        // Reset UI
        isProcessing = false;
        analyzeButton.disabled = false;
        buttonText.textContent = 'Analyze for Deepfakes';
        loadingSpinner.style.display = 'none';
        progressText.style.display = 'none';
    }
}

/**
 * Display analysis results
 */
function displayResults(data) {
    // Hide error section
    errorSection.style.display = 'none';

    // Show results section
    resultsSection.style.display = 'block';

    // Reset all result cards
    document.getElementById('videoResults').style.display = 'none';
    document.getElementById('audioResults').style.display = 'none';
    document.getElementById('combinedResults').style.display = 'none';

    // Display video results if present
    if (data.predictions.video) {
        displayVideoResults(data.predictions.video, data.processing_time);
    }

    // Display audio results if present
    if (data.predictions.audio) {
        displayAudioResults(data.predictions.audio, data.processing_time);
    }

    // Display combined results if present
    if (data.predictions.combined) {
        displayCombinedResults(data.predictions.combined);
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Display video analysis results
 */
function displayVideoResults(videoData, processingTime) {
    const videoResults = document.getElementById('videoResults');
    const videoPredictionBadge = document.getElementById('videoPredictionBadge');
    const videoProbabilityBar = document.getElementById('videoProbabilityBar');
    const videoFramesAnalyzed = document.getElementById('videoFramesAnalyzed');
    const videoProcessingTime = document.getElementById('videoProcessingTime');

    // Set prediction badge
    videoPredictionBadge.textContent = videoData.prediction.toUpperCase();
    videoPredictionBadge.className = 'prediction-badge ' + videoData.prediction;

    // Create probability bar
    videoProbabilityBar.innerHTML = createProbabilityBar(
        videoData.real_probability,
        videoData.fake_probability
    );

    // Set metadata
    videoFramesAnalyzed.textContent = videoData.frames_analyzed || 'N/A';
    videoProcessingTime.textContent = processingTime.toFixed(2);

    // Show video results
    videoResults.style.display = 'block';
}

/**
 * Display audio analysis results
 */
function displayAudioResults(audioData, processingTime) {
    const audioResults = document.getElementById('audioResults');
    const audioPredictionBadge = document.getElementById('audioPredictionBadge');
    const audioProbabilityBar = document.getElementById('audioProbabilityBar');
    const audioSegmentsAnalyzed = document.getElementById('audioSegmentsAnalyzed');
    const audioDuration = document.getElementById('audioDuration');

    // Set prediction badge
    audioPredictionBadge.textContent = audioData.prediction.toUpperCase();
    audioPredictionBadge.className = 'prediction-badge ' + audioData.prediction;

    // Create probability bar
    audioProbabilityBar.innerHTML = createProbabilityBar(
        audioData.real_probability,
        audioData.fake_probability
    );

    // Set metadata
    audioSegmentsAnalyzed.textContent = audioData.segments_analyzed || 'N/A';
    audioDuration.textContent = audioData.duration_seconds ? audioData.duration_seconds.toFixed(1) : 'N/A';

    // Show audio results
    audioResults.style.display = 'block';
}

/**
 * Display combined analysis results
 */
function displayCombinedResults(combinedData) {
    const combinedResults = document.getElementById('combinedResults');
    const combinedPredictionBadge = document.getElementById('combinedPredictionBadge');
    const combinedProbabilityBar = document.getElementById('combinedProbabilityBar');
    const finalVerdict = document.getElementById('finalVerdict');

    // Set prediction badge
    combinedPredictionBadge.textContent = combinedData.prediction.toUpperCase();
    combinedPredictionBadge.className = 'prediction-badge ' + combinedData.prediction;

    // Create probability bar
    combinedProbabilityBar.innerHTML = createProbabilityBar(
        combinedData.real_probability,
        combinedData.fake_probability
    );

    // Set final verdict
    finalVerdict.textContent = combinedData.prediction.toUpperCase();
    finalVerdict.style.color = combinedData.prediction === 'real' ? '#4caf50' : '#f44336';

    // Show combined results
    combinedResults.style.display = 'block';
}

/**
 * Create probability bar HTML
 */
function createProbabilityBar(realProb, fakeProb) {
    const realPercent = (realProb * 100).toFixed(1);
    const fakePercent = (fakeProb * 100).toFixed(1);

    return `
        <div class="probability-bar real" style="width: ${realPercent}%;">
            <span class="probability-text">Real: ${realPercent}%</span>
        </div>
        <div class="probability-bar fake" style="width: ${fakePercent}%; position: absolute; right: 0;">
            <span class="probability-text">Fake: ${fakePercent}%</span>
        </div>
    `;
}

/**
 * Display error message
 */
function displayError(errorData) {
    // Hide results section
    resultsSection.style.display = 'none';

    // Show error section
    errorSection.style.display = 'block';

    // Set error message
    let message = errorData.error || 'An unknown error occurred';

    // Add user-friendly context based on error type
    if (errorData.error_type === 'validation_error') {
        message = '❌ Validation Error: ' + message;
    } else if (errorData.error_type === 'processing_error') {
        message = '⚠ Processing Error: ' + message;
    } else if (errorData.error_type === 'server_error') {
        message = '🔧 Server Error: ' + message;
    } else if (errorData.error_type === 'network_error') {
        message = '🌐 Network Error: ' + message;
    }

    errorMessage.textContent = message;

    // Scroll to error
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Validate file
 */
function validateFile(file) {
    if (!file) {
        return { valid: false, error: 'No file selected' };
    }

    // Get file extension
    const ext = getFileExtension(file.name);

    // Check file format
    const videoExtensions = ['.mp4', '.avi', '.mov', '.webm'];
    const audioExtensions = ['.mp3', '.wav', '.m4a', '.ogg'];
    const allExtensions = [...videoExtensions, ...audioExtensions];

    if (!allExtensions.includes(ext)) {
        return {
            valid: false,
            error: 'Unsupported file format. Please select a video (MP4, AVI, MOV, WEBM) or audio (MP3, WAV, M4A, OGG) file.'
        };
    }

    // Check file size
    const isVideo = videoExtensions.includes(ext);
    const maxSize = isVideo ? 100 * 1024 * 1024 : 50 * 1024 * 1024; // 100MB for video, 50MB for audio

    if (file.size > maxSize) {
        const maxSizeMB = isVideo ? 100 : 50;
        return {
            valid: false,
            error: `File size exceeds ${maxSizeMB}MB limit. Please select a smaller file.`
        };
    }

    return { valid: true };
}

/**
 * Get file extension
 */
function getFileExtension(filename) {
    return filename.substring(filename.lastIndexOf('.')).toLowerCase();
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Reset UI to initial state
 */
function resetUI() {
    currentFile = null;
    fileInput.value = '';
    selectedFileInfo.style.display = 'none';
    analyzeAudioContainer.style.display = 'none';
    analyzeAudioCheckbox.checked = false;
    analyzeButton.disabled = true;
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    progressText.style.display = 'none';
}

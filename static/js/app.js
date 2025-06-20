// ViT-GIF Highlight - Interactive Web Application
// Main JavaScript application logic

class ViTGIFApp {
    constructor() {
        this.currentStep = 1;
        this.jobId = null;
        this.videoData = null;
        this.websocket = null;
        this.videoDuration = 0;
        this.startTime = 0;
        this.duration = 5;
        this.isDragging = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.setupTimelineControls();
        this.setupRangeSliders();
        this.showStep(1);
    }
    
    // Event Listeners Setup
    setupEventListeners() {
        // File input
        const videoInput = document.getElementById('videoInput');
        videoInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Manual time inputs
        const startTimeInput = document.getElementById('startTimeInput');
        const durationInput = document.getElementById('durationInput');
        
        startTimeInput.addEventListener('input', (e) => this.updateStartTime(parseFloat(e.target.value)));
        durationInput.addEventListener('input', (e) => this.updateDuration(parseFloat(e.target.value)));
        
        // Video player events
        const videoPlayer = document.getElementById('videoPlayer');
        videoPlayer.addEventListener('loadedmetadata', () => this.onVideoLoaded());
        videoPlayer.addEventListener('timeupdate', () => this.onVideoTimeUpdate());
    }
    
    // Drag and Drop Setup
    setupDragAndDrop() {
        const uploadArea = document.getElementById('uploadArea');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
        });
        
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e), false);
        uploadArea.addEventListener('click', () => document.getElementById('videoInput').click());
    }
    
    // Timeline Controls Setup
    setupTimelineControls() {
        const timeline = document.getElementById('timeline');
        const startHandle = document.getElementById('startHandle');
        const endHandle = document.getElementById('endHandle');
        
        // Timeline click
        timeline.addEventListener('click', (e) => this.onTimelineClick(e));
        
        // Handle dragging
        startHandle.addEventListener('mousedown', (e) => this.startDragging(e, 'start'));
        endHandle.addEventListener('mousedown', (e) => this.startDragging(e, 'end'));
        
        document.addEventListener('mousemove', (e) => this.onDragging(e));
        document.addEventListener('mouseup', () => this.stopDragging());
        
        // Touch events for mobile
        startHandle.addEventListener('touchstart', (e) => this.startDragging(e, 'start'));
        endHandle.addEventListener('touchstart', (e) => this.startDragging(e, 'end'));
        document.addEventListener('touchmove', (e) => this.onDragging(e));
        document.addEventListener('touchend', () => this.stopDragging());
    }
    
    // Range Sliders Setup
    setupRangeSliders() {
        const ranges = [
            { id: 'fpsRange', valueId: 'fpsValue' },
            { id: 'maxFramesRange', valueId: 'maxFramesValue' },
            { id: 'overlayIntensity', valueId: 'overlayIntensityValue' },
            { id: 'optimizationLevel', valueId: 'optimizationValue' }
        ];
        
        ranges.forEach(({ id, valueId }) => {
            const range = document.getElementById(id);
            const valueDisplay = document.getElementById(valueId);
            
            range.addEventListener('input', () => {
                valueDisplay.textContent = range.value;
                this.updateSummary();
            });
        });
        
        // Model select
        document.getElementById('modelSelect').addEventListener('change', () => {
            this.updateModelInfo();
            this.updateSummary();
        });
        
        // Overlay style
        document.getElementById('overlayStyle').addEventListener('change', () => {
            this.updateSummary();
        });
    }
    
    // File Handling
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }
    
    async processFile(file) {
        // Validate file
        if (!this.validateFile(file)) {
            return;
        }
        
        // Show upload progress
        this.showUploadProgress();
        
        try {
            // Create FormData
            const formData = new FormData();
            formData.append('file', file);
            
            // Upload file
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            this.jobId = result.job_id;
            this.videoData = result;
            
            // Load video in player
            const videoPlayer = document.getElementById('videoPlayer');
            const videoBlob = new Blob([file], { type: file.type });
            const videoURL = URL.createObjectURL(videoBlob);
            videoPlayer.src = videoURL;
            
            this.hideUploadProgress();
            this.displayVideoInfo(result);
            this.goToStep(2);
            
            this.showToast('Video cargado exitosamente', 'success');
            
        } catch (error) {
            this.hideUploadProgress();
            this.showToast(`Error al cargar video: ${error.message}`, 'error');
        }
    }
    
    validateFile(file) {
        const maxSize = 100 * 1024 * 1024; // 100MB
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm'];
        
        if (file.size > maxSize) {
            this.showToast('El archivo es demasiado grande (máx. 100MB)', 'error');
            return false;
        }
        
        if (!allowedTypes.includes(file.type)) {
            this.showToast('Formato de archivo no soportado', 'error');
            return false;
        }
        
        return true;
    }
    
    showUploadProgress() {
        document.getElementById('uploadProgress').style.display = 'block';
        document.getElementById('uploadArea').style.display = 'none';
    }
    
    hideUploadProgress() {
        document.getElementById('uploadProgress').style.display = 'none';
        document.getElementById('uploadArea').style.display = 'block';
    }
    
    displayVideoInfo(data) {
        const info = data.metadata;
        
        document.getElementById('videoDuration').textContent = this.formatTime(info.duration);
        document.getElementById('videoResolution').textContent = `${info.width}x${info.height}`;
        document.getElementById('videoFPS').textContent = info.fps.toFixed(1);
        document.getElementById('videoSize').textContent = this.formatFileSize(info.file_size);
        
        this.videoDuration = info.duration;
        this.updateTimeline();
        
        // Set timeline end label
        document.getElementById('timelineEnd').textContent = this.formatTime(info.duration);
    }
    
    onVideoLoaded() {
        const videoPlayer = document.getElementById('videoPlayer');
        this.videoDuration = videoPlayer.duration;
        this.updateTimeline();
    }
    
    onVideoTimeUpdate() {
        // Update video player position relative to timeline
        const videoPlayer = document.getElementById('videoPlayer');
        const currentTime = videoPlayer.currentTime;
        
        // You could add visual feedback here
    }
    
    // Timeline Controls
    onTimelineClick(e) {
        if (this.isDragging) return;
        
        const timeline = document.getElementById('timeline');
        const rect = timeline.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const timelineWidth = rect.width - 32; // Account for padding
        const clickRatio = (clickX - 16) / timelineWidth;
        const clickTime = Math.max(0, Math.min(this.videoDuration, clickRatio * this.videoDuration));
        
        // Move start time to click position
        this.updateStartTime(clickTime);
    }
    
    startDragging(e, handle) {
        this.isDragging = true;
        this.dragHandle = handle;
        e.preventDefault();
    }
    
    onDragging(e) {
        if (!this.isDragging) return;
        
        const timeline = document.getElementById('timeline');
        const rect = timeline.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        const dragX = clientX - rect.left;
        const timelineWidth = rect.width - 32;
        const dragRatio = Math.max(0, Math.min(1, (dragX - 16) / timelineWidth));
        const dragTime = dragRatio * this.videoDuration;
        
        if (this.dragHandle === 'start') {
            const maxStart = this.videoDuration - this.duration;
            this.updateStartTime(Math.max(0, Math.min(maxStart, dragTime)));
        } else if (this.dragHandle === 'end') {
            const newEndTime = Math.max(this.startTime + 1, Math.min(this.videoDuration, dragTime));
            this.updateDuration(newEndTime - this.startTime);
        }
        
        e.preventDefault();
    }
    
    stopDragging() {
        this.isDragging = false;
        this.dragHandle = null;
    }
    
    updateStartTime(newStartTime) {
        const maxStart = this.videoDuration - this.duration;
        this.startTime = Math.max(0, Math.min(maxStart, newStartTime));
        
        document.getElementById('startTimeInput').value = this.startTime.toFixed(1);
        this.updateTimelineDisplay();
        this.updateSummary();
        
        // Update video player
        const videoPlayer = document.getElementById('videoPlayer');
        videoPlayer.currentTime = this.startTime;
    }
    
    updateDuration(newDuration) {
        const maxDuration = Math.min(60, this.videoDuration - this.startTime);
        this.duration = Math.max(1, Math.min(maxDuration, newDuration));
        
        document.getElementById('durationInput').value = this.duration.toFixed(1);
        this.updateTimelineDisplay();
        this.updateSummary();
    }
    
    updateTimeline() {
        this.updateTimelineDisplay();
    }
    
    updateTimelineDisplay() {
        if (this.videoDuration === 0) return;
        
        const startRatio = this.startTime / this.videoDuration;
        const durationRatio = this.duration / this.videoDuration;
        
        const selection = document.getElementById('timelineSelection');
        const startHandle = document.getElementById('startHandle');
        const endHandle = document.getElementById('endHandle');
        
        const startPercent = startRatio * 100;
        const widthPercent = durationRatio * 100;
        const endPercent = (startRatio + durationRatio) * 100;
        
        selection.style.left = `${16 + startPercent * 0.01 * (selection.parentElement.clientWidth - 32)}px`;
        selection.style.width = `${widthPercent * 0.01 * (selection.parentElement.clientWidth - 32)}px`;
        
        startHandle.style.left = `${16 + startPercent * 0.01 * (startHandle.parentElement.clientWidth - 32)}px`;
        endHandle.style.left = `${16 + endPercent * 0.01 * (endHandle.parentElement.clientWidth - 32)}px`;
        
        // Update info display
        document.getElementById('startTimeDisplay').textContent = this.formatTime(this.startTime);
        document.getElementById('durationDisplay').textContent = `${this.duration.toFixed(1)}s`;
        document.getElementById('endTimeDisplay').textContent = this.formatTime(this.startTime + this.duration);
    }
    
    // Step Navigation
    goToStep(step) {
        // Hide current step
        const currentSection = document.querySelector('section.active');
        if (currentSection) {
            currentSection.classList.remove('active');
        }
        
        // Show new step
        const sections = {
            1: 'uploadSection',
            2: 'videoSection',
            3: 'controlsSection',
            4: 'settingsSection',
            5: 'generateSection'
        };
        
        const newSection = document.getElementById(sections[step]);
        if (newSection) {
            newSection.classList.add('active');
        }
        
        // Update progress indicator
        document.querySelectorAll('.progress-indicator .step').forEach((el, index) => {
            el.classList.remove('active', 'completed');
            if (index + 1 === step) {
                el.classList.add('active');
            } else if (index + 1 < step) {
                el.classList.add('completed');
            }
        });
        
        this.currentStep = step;
        
        // Special handling for certain steps
        if (step === 4) {
            this.updateModelInfo();
            this.updateSummary();
        } else if (step === 5) {
            this.updateSummary();
            this.updateTimeEstimate();
        }
    }
    
    // Duration Buttons
    setDuration(seconds) {
        // Update active button
        document.querySelectorAll('.duration-buttons .btn').forEach(btn => {
            btn.classList.remove('active');
        });
        event.target.classList.add('active');
        
        this.updateDuration(seconds);
    }
    
    // Settings
    updateModelInfo() {
        const modelSelect = document.getElementById('modelSelect');
        const modelInfo = document.getElementById('modelInfo');
        
        const modelDescriptions = {
            '': '📊 El modelo se seleccionará automáticamente basado en las características de tu video.',
            'videomae-base': '⚡ VideoMAE Base: Rápido y eficiente, ideal para videos cortos. Memoria: ~2-3GB GPU.',
            'videomae-large': '🎯 VideoMAE Large: Mejor calidad de atención, más lento. Memoria: ~4-6GB GPU.',
            'timesformer-base': '🚀 TimeSformer: Muy eficiente para videos largos. Memoria: ~2-4GB GPU.'
        };
        
        modelInfo.innerHTML = `<p>${modelDescriptions[modelSelect.value]}</p>`;
    }
    
    updateSummary() {
        document.getElementById('summarySegment').textContent = 
            `${this.formatTime(this.startTime)} - ${this.formatTime(this.startTime + this.duration)} (${this.duration.toFixed(1)}s)`;
        
        document.getElementById('summaryFPS').textContent = document.getElementById('fpsRange').value;
        
        const styleSelect = document.getElementById('overlayStyle');
        const styleText = styleSelect.options[styleSelect.selectedIndex].text;
        document.getElementById('summaryStyle').textContent = styleText;
        
        const modelSelect = document.getElementById('modelSelect');
        const modelText = modelSelect.value || 'Automático';
        document.getElementById('summaryModel').textContent = modelText;
    }
    
    updateTimeEstimate() {
        const duration = this.duration;
        const fps = parseInt(document.getElementById('fpsRange').value);
        const model = document.getElementById('modelSelect').value || 'videomae-base';
        
        // Rough time estimation
        let baseTime = duration * 2; // Base processing time
        
        if (model.includes('large')) {
            baseTime *= 1.5;
        } else if (model.includes('huge')) {
            baseTime *= 2.5;
        }
        
        baseTime += fps * 2; // More FPS = more processing
        
        document.getElementById('timeEstimate').textContent = 
            `Tiempo estimado: ~${Math.ceil(baseTime)} segundos`;
    }
    
    // GIF Generation
    async generateGIF() {
        if (!this.jobId) {
            this.showToast('No hay video cargado', 'error');
            return;
        }
        
        // Get settings
        const settings = {
            job_id: this.jobId,
            start_time: this.startTime,
            duration: this.duration,
            fps: parseInt(document.getElementById('fpsRange').value),
            max_frames: parseInt(document.getElementById('maxFramesRange').value),
            overlay_style: document.getElementById('overlayStyle').value,
            overlay_intensity: parseFloat(document.getElementById('overlayIntensity').value),
            optimization_level: parseInt(document.getElementById('optimizationLevel').value),
            model_name: document.getElementById('modelSelect').value || null
        };
        
        try {
            // Start processing
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });
            
            if (!response.ok) {
                throw new Error(`Processing failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            // Show processing status
            this.showProcessingStatus();
            
            // Connect to WebSocket for real-time updates
            this.connectWebSocket();
            
            // Start polling for status
            this.pollStatus();
            
        } catch (error) {
            this.showError(`Error al iniciar procesamiento: ${error.message}`);
        }
    }
    
    showProcessingStatus() {
        document.getElementById('generateBtn').style.display = 'none';
        document.getElementById('processingStatus').style.display = 'block';
        
        // Reset processing steps
        document.querySelectorAll('.processing-steps .step').forEach(step => {
            step.classList.remove('active', 'completed');
            step.querySelector('.step-status').textContent = '⏳';
        });
    }
    
    connectWebSocket() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/ws/${this.jobId}`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateProcessingProgress(data);
        };
        
        this.websocket.onerror = (error) => {
            console.warn('WebSocket error:', error);
        };
        
        this.websocket.onclose = () => {
            this.websocket = null;
        };
    }
    
    async pollStatus() {
        const maxAttempts = 60; // 5 minutes max
        let attempts = 0;
        
        const poll = async () => {
            if (attempts >= maxAttempts) {
                this.showError('Timeout: El procesamiento tardó demasiado');
                return;
            }
            
            try {
                const response = await fetch(`/api/status/${this.jobId}`);
                if (!response.ok) {
                    throw new Error('Status check failed');
                }
                
                const status = await response.json();
                this.updateProcessingProgress(status);
                
                if (status.status === 'completed') {
                    this.showResults(status);
                } else if (status.status === 'failed') {
                    this.showError(status.error || 'Procesamiento fallido');
                } else {
                    attempts++;
                    setTimeout(poll, 5000); // Poll every 5 seconds
                }
                
            } catch (error) {
                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(poll, 5000);
                } else {
                    this.showError('Error de conexión');
                }
            }
        };
        
        poll();
    }
    
    updateProcessingProgress(data) {
        // Update progress bar
        const progressFill = document.getElementById('processingProgressFill');
        const progressPercent = document.getElementById('processingProgressPercent');
        const progressMessage = document.getElementById('processingProgressMessage');
        
        progressFill.style.width = `${data.progress}%`;
        progressPercent.textContent = `${data.progress}%`;
        progressMessage.textContent = data.message;
        
        // Update processing steps
        const steps = document.querySelectorAll('.processing-steps .step');
        
        // Determine current step based on progress
        let currentStepIndex = 0;
        if (data.progress > 25) currentStepIndex = 1;
        if (data.progress > 50) currentStepIndex = 2;
        if (data.progress > 75) currentStepIndex = 3;
        
        steps.forEach((step, index) => {
            step.classList.remove('active', 'completed');
            const statusEl = step.querySelector('.step-status');
            
            if (index < currentStepIndex) {
                step.classList.add('completed');
                statusEl.textContent = '✅';
            } else if (index === currentStepIndex) {
                step.classList.add('active');
                statusEl.textContent = '⚡';
            } else {
                statusEl.textContent = '⏳';
            }
        });
    }
    
    showResults(result) {
        document.getElementById('processingStatus').style.display = 'none';
        document.getElementById('resultsContainer').style.display = 'block';
        
        // Show result GIF
        const resultGIF = document.getElementById('resultGIF');
        const downloadBtn = document.getElementById('downloadBtn');
        
        resultGIF.src = result.result_url;
        downloadBtn.href = result.result_url;
        downloadBtn.download = `vitgif_${this.jobId}.gif`;
        
        // Show result info
        this.displayResultInfo(result);
        
        this.showToast('¡GIF generado exitosamente!', 'success');
    }
    
    displayResultInfo(result) {
        const resultInfo = document.getElementById('resultInfo');
        
        // This would be populated with actual result data
        resultInfo.innerHTML = `
            <h4>📊 Estadísticas del GIF</h4>
            <div class="info-grid">
                <div class="info-item">
                    <span class="label">Tiempo de procesamiento:</span>
                    <span>${result.processing_time || 'N/A'}</span>
                </div>
                <div class="info-item">
                    <span class="label">Frames seleccionados:</span>
                    <span>${result.selected_frames || 'N/A'}</span>
                </div>
                <div class="info-item">
                    <span class="label">Tamaño del archivo:</span>
                    <span>${result.file_size || 'N/A'}</span>
                </div>
                <div class="info-item">
                    <span class="label">Ratio de compresión:</span>
                    <span>${result.compression_ratio || 'N/A'}</span>
                </div>
            </div>
        `;
    }
    
    showError(message) {
        document.getElementById('processingStatus').style.display = 'none';
        document.getElementById('errorContainer').style.display = 'block';
        document.getElementById('errorMessage').textContent = message;
        
        this.showToast(message, 'error');
    }
    
    retryGeneration() {
        document.getElementById('errorContainer').style.display = 'none';
        document.getElementById('generateBtn').style.display = 'block';
    }
    
    resetApp() {
        // Reset all state
        this.currentStep = 1;
        this.jobId = null;
        this.videoData = null;
        this.videoDuration = 0;
        this.startTime = 0;
        this.duration = 5;
        
        // Reset UI
        document.getElementById('videoInput').value = '';
        document.getElementById('videoPlayer').src = '';
        document.getElementById('processingStatus').style.display = 'none';
        document.getElementById('resultsContainer').style.display = 'none';
        document.getElementById('errorContainer').style.display = 'none';
        document.getElementById('generateBtn').style.display = 'block';
        
        // Reset form values
        document.getElementById('startTimeInput').value = '0';
        document.getElementById('durationInput').value = '5';
        document.getElementById('fpsRange').value = '5';
        document.getElementById('fpsValue').textContent = '5';
        
        this.goToStep(1);
        this.showToast('Aplicación reiniciada', 'success');
    }
    
    shareGIF() {
        if (navigator.share) {
            navigator.share({
                title: 'Mi GIF generado con ViT-GIF Highlight',
                text: 'Mira este GIF inteligente que creé',
                url: document.getElementById('resultGIF').src
            });
        } else {
            // Fallback: copy URL to clipboard
            navigator.clipboard.writeText(document.getElementById('resultGIF').src);
            this.showToast('URL copiada al portapapeles', 'success');
        }
    }
    
    // Utility Functions
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    formatFileSize(bytes) {
        const units = ['B', 'KB', 'MB', 'GB'];
        let size = bytes;
        let unitIndex = 0;
        
        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }
        
        return `${size.toFixed(1)} ${units[unitIndex]}`;
    }
    
    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        container.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }
}

// Global functions for HTML onclick handlers
function goToStep(step) {
    app.goToStep(step);
}

function setDuration(seconds) {
    app.setDuration(seconds);
}

function generateGIF() {
    app.generateGIF();
}

function retryGeneration() {
    app.retryGeneration();
}

function resetApp() {
    app.resetApp();
}

function shareGIF() {
    app.shareGIF();
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new ViTGIFApp();
});

// Handle window resize for timeline
window.addEventListener('resize', () => {
    if (app) {
        app.updateTimelineDisplay();
    }
}); 

/**
 * DR Vision AI - Core Frontend Logic
 * Rural Health Centre Retinal Screening System
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Sidebar Toggle for Mobile & Tablets
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.app-sidebar');
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('show');
        });
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth < 992 && sidebar && sidebar.classList.contains('show')) {
            if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
                sidebar.classList.remove('show');
            }
        }
    });

    // 2. Auto-dismiss alerts after 6 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        }, 6000);
    });

    // 3. Screening Upload & Drag-and-Drop Handler
    setupScreeningUploader();
});

function setupScreeningUploader() {
    const dropzone = document.getElementById('retinalDropzone');
    const fileInput = document.getElementById('fundusFileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImg = document.getElementById('previewImg');
    const qualityBox = document.getElementById('qualityAssessmentBox');
    const tempFilenameInput = document.getElementById('tempFilenameInput');
    const submitBtn = document.getElementById('runAiBtn');

    if (!dropzone || !fileInput) return;

    // Dropzone click triggers input
    dropzone.addEventListener('click', () => fileInput.click());

    // Drag-over styling
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    // Handle standard browse selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file (JPG, JPEG, or PNG).');
            return;
        }

        // Display local preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            dropzone.classList.add('d-none');
            previewContainer.classList.remove('d-none');
        };
        reader.readAsDataURL(file);

        // Run automated quality assessment via AJAX
        performQualityAssessment(file);
    }

    function performQualityAssessment(file) {
        if (!qualityBox) return;

        qualityBox.classList.remove('d-none');
        qualityBox.innerHTML = `
            <div class="p-3 text-center text-muted">
                <div class="spinner-border spinner-border-sm text-teal me-2" role="status"></div>
                Analyzing image focus, illumination, and retinal field coverage...
            </div>
        `;

        const formData = new FormData();
        formData.append('image', file);

        fetch('/screening/upload-quality', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (tempFilenameInput) {
                    tempFilenameInput.value = data.temp_filename;
                }
                renderQualityResults(data.quality);
            } else {
                qualityBox.innerHTML = `
                    <div class="alert alert-danger p-2 mb-0">
                        <i class="bi bi-exclamation-triangle-fill me-2"></i> ${data.error || 'Quality analysis failed.'}
                    </div>
                `;
            }
        })
        .catch(err => {
            console.error(err);
            qualityBox.innerHTML = `
                <div class="alert alert-warning p-2 mb-0">
                    <i class="bi bi-exclamation-circle me-2"></i> Quality check server unavailable. Proceeding with caution.
                </div>
            `;
        });
    }

    function renderQualityResults(q) {
        let badgeClass = 'bg-success';
        let statusIcon = 'bi-check-circle-fill';
        let barColor = '#10B981';

        if (q.status === 'Warning') {
            badgeClass = 'bg-warning text-dark';
            statusIcon = 'bi-exclamation-triangle-fill';
            barColor = '#F59E0B';
        } else if (q.status === 'Fail') {
            badgeClass = 'bg-danger';
            statusIcon = 'bi-x-circle-fill';
            barColor = '#EF4444';
            if (submitBtn) submitBtn.disabled = true;
        } else {
            if (submitBtn) submitBtn.disabled = false;
        }

        let issuesHtml = '';
        if (q.issues && q.issues.length > 0) {
            issuesHtml = `
                <div class="mt-2 text-danger small">
                    <b>Detected Issues:</b>
                    <ul class="mb-1 ps-3">
                        ${q.issues.map(i => `<li>${i}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        let recsHtml = '';
        if (q.recommendations && q.recommendations.length > 0) {
            recsHtml = `
                <div class="mt-1 text-primary small">
                    <b>Guidance for Health Worker:</b>
                    <ul class="mb-0 ps-3">
                        ${q.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        qualityBox.innerHTML = `
            <div class="border rounded p-3 bg-white">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="fw-semibold">
                        <i class="bi ${statusIcon} me-1"></i> Quality Status: <span class="badge ${badgeClass}">${q.status}</span>
                    </span>
                    <span class="fw-bold">${q.score} / 100</span>
                </div>
                <div class="quality-gauge mb-2">
                    <div class="quality-fill" style="width: ${q.score}%; background-color: ${barColor};"></div>
                </div>
                <div class="row g-2 text-muted small text-center">
                    <div class="col-4">Focus Var: <b>${q.metrics.laplacian_var}</b></div>
                    <div class="col-4">Luminosity: <b>${q.metrics.mean_brightness}</b></div>
                    <div class="col-4">Retina FOV: <b>${Math.round(q.metrics.fov_ratio * 100)}%</b></div>
                </div>
                ${issuesHtml}
                ${recsHtml}
            </div>
        `;
    }

    // Reset button to pick another image
    const resetBtn = document.getElementById('resetUploadBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            fileInput.value = '';
            if (tempFilenameInput) tempFilenameInput.value = '';
            previewContainer.classList.add('d-none');
            dropzone.classList.remove('d-none');
            if (qualityBox) {
                qualityBox.classList.add('d-none');
                qualityBox.innerHTML = '';
            }
            if (submitBtn) submitBtn.disabled = false;
        });
    }
}

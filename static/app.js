/**
 * Bilingual DOCX Translation — Frontend Logic
 */
(function () {
    "use strict";

    // --- DOM References ---
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const fileInfo = document.getElementById("file-info");
    const fileName = document.getElementById("file-name");
    const translateBtn = document.getElementById("translate-btn");

    const progressSection = document.getElementById("progress-section");
    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");
    const progressSteps = document.getElementById("progress-steps");

    const resultSection = document.getElementById("result-section");
    const resultStats = document.getElementById("result-stats");
    const downloadLink = document.getElementById("download-link");
    const translateAnother = document.getElementById("translate-another");

    const errorSection = document.getElementById("error-section");
    const errorMessage = document.getElementById("error-message");
    const retryBtn = document.getElementById("retry-btn");

    let selectedFile = null;

    // --- Drag & Drop ---
    uploadZone.addEventListener("click", function () {
        fileInput.click();
    });

    uploadZone.addEventListener("dragover", function (e) {
        e.preventDefault();
        uploadZone.classList.add("drag-over");
    });

    uploadZone.addEventListener("dragleave", function (e) {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
    });

    uploadZone.addEventListener("drop", function (e) {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.name.toLowerCase().endsWith(".docx")) {
            alert("Please select a .docx file.");
            return;
        }
        selectedFile = file;
        fileName.textContent = file.name;
        fileInfo.classList.remove("hidden");
        uploadZone.classList.add("hidden");
        resetState();
    }

    // --- Translate ---
    translateBtn.addEventListener("click", startTranslation);

    function startTranslation() {
        if (!selectedFile) return;

        fileInfo.classList.add("hidden");
        progressSection.classList.remove("hidden");
        setProgress(5, "Uploading document...");
        setStepActive("extract");

        const formData = new FormData();
        formData.append("file", selectedFile);

        setProgress(20, "Extracting content...");
        setStepActive("extract");

        fetch("/api/translate", {
            method: "POST",
            body: formData,
        })
            .then(function (response) {
                // Simulate progress since we can't stream
                setProgress(60, "Translating text...");
                setStepActive("translate");
                simulateProgress(60, 90, "Translating text...");

                return response.json();
            })
            .then(function (data) {
                clearInterval(window._progressInterval);
                if (data.error) {
                    showError(data.error);
                    return;
                }

                setProgress(95, "Building document...");
                setStepActive("insert");

                setTimeout(function () {
                    setProgress(100, "Complete!");
                    setStepDone("extract");
                    setStepDone("translate");
                    setStepDone("insert");
                    setStepActive("done");

                    setTimeout(function () {
                        showResult(data);
                    }, 400);
                }, 600);
            })
            .catch(function (err) {
                clearInterval(window._progressInterval);
                showError("Network error: " + err.message);
            });
    }

    function simulateProgress(from, to, text) {
        let val = from;
        window._progressInterval = setInterval(function () {
            val += (to - val) * 0.15;
            if (val >= to - 1) {
                clearInterval(window._progressInterval);
                return;
            }
            setProgress(Math.round(val), text);
            if (val > 80) {
                setStepActive("insert");
            }
        }, 800);
    }

    // --- Result ---
    function showResult(data) {
        progressSection.classList.add("hidden");
        resultSection.classList.remove("hidden");

        const p = data.stats.paragraphs_translated || 0;
        const t = data.stats.table_cells_translated || 0;
        resultStats.textContent =
            p + " paragraphs and " + t + " table cells translated.";

        downloadLink.href = "/api/download/" + data.job_id + "/" + data.filename;
    }

    // --- Error ---
    function showError(msg) {
        progressSection.classList.add("hidden");
        errorSection.classList.remove("hidden");
        errorMessage.textContent = msg;
    }

    retryBtn.addEventListener("click", function () {
        resetState();
        errorSection.classList.add("hidden");
        uploadZone.classList.remove("hidden");
    });

    // --- Translate Another ---
    translateAnother.addEventListener("click", function () {
        resetState();
        resultSection.classList.add("hidden");
        uploadZone.classList.remove("hidden");
        selectedFile = null;
    });

    // --- Helpers ---
    function setProgress(pct, text) {
        progressBar.style.width = pct + "%";
        if (text) progressText.textContent = text;
    }

    function setStepActive(name) {
        const steps = progressSteps.querySelectorAll(".step");
        steps.forEach(function (s) {
            s.classList.remove("active", "done");
            if (s.dataset.step === name) {
                s.classList.add("active");
            }
        });
    }

    function setStepDone(name) {
        const step = progressSteps.querySelector('[data-step="' + name + '"]');
        if (step) {
            step.classList.remove("active");
            step.classList.add("done");
        }
    }

    function resetState() {
        progressSection.classList.add("hidden");
        resultSection.classList.add("hidden");
        errorSection.classList.add("hidden");
        progressBar.style.width = "0%";
        const steps = progressSteps.querySelectorAll(".step");
        steps.forEach(function (s) {
            s.classList.remove("active", "done");
        });
        clearInterval(window._progressInterval);
    }
})();

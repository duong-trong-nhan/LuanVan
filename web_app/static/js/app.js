/**
 * Heritage XAI - Main Application Controller & GSAP UI Choreography
 * ------------------------------------------------------------------
 * Điều phối toàn bộ luồng người dùng:
 * 1. Chuyển đổi 3 chế độ đầu vào (Video, Audio, Live Mic Recorder).
 * 2. Kết nối Web Audio API với 3D Particle Ribbon.
 * 3. Hiển thị kết quả qua GSAP Timeline (Staggered Reveal, Count-up animation).
 * 4. Phân đoạn Timeline (Chunks 3-5s) & Filmstrip tương tác.
 * 5. Tương tác CBM Concept Pills (Counterfactual Intervention qua /api/intervene).
 * 6. Liên kết 2D DOM -> 3D Semantic Graph Camera Fly-to.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Khởi tạo các thành phần đồ họa 3D & Audio
    const audioWave = new AudioWaveRibbon("wave-canvas-container");
    const graph3d = new SemanticGraph3D(
        "graph-canvas-container",
        "graph-tooltip",
    );
    const recorder = new LiveAudioRecorder({
        timerId: "mic-timer",
        ringId: "mic-timer-ring",
        statusId: "mic-status-text",
    });

    // Lưu trữ tham chiếu toàn cục
    window.audioWave = audioWave;
    window.graph3d = graph3d;

    // Bảng màu 6 Di sản Văn hóa Âm nhạc Việt Nam chuẩn hóa đồng bộ với Backend
    const GENRE_COLORS = {
        "Ca trù": "#C84B31", // Đỏ gạch nung (Vermilion)
        Chèo: "#2E5A88", // Xanh chàm đình làng (Indigo)
        "Chầu văn": "#8C3B88", // Tím khăn chầu áo ngự (Ceremonial Purple)
        "Cải lương": "#C97A1B", // Vàng đồng / Hổ phách (Amber Ochre)
        "Hát xẩm": "#7A6248", // Nâu đất chiếu xẩm (Earth Brown)
        "Quan họ": "#1E6B5B", // Xanh ngọc quan họ (Jade Emerald)
    };

    // State quản lý kết quả hiện tại & Nguồn đầu vào duy nhất (Single Active Input)
    let currentAnalysis = null;
    let currentActiveTab = "media"; // "media" | "mic" | "url"
    let activeMediaType = null; // "video" | "audio" | null (Chỉ cho phép 1 input tồn tại tại 1 thời điểm)

    // State quản lý phân đoạn thời gian, gộp đoạn thông minh & XAI
    let currentTimelineMode = "raw"; // "raw" (mặc định từng chunk) | "merged"
    let currentRawChunks = [];
    let currentMergedSegments = [];
    let currentTimelineSeries = [];
    let currentOverallExplanation = null;
    let currentOverallModalities = null;
    let activeSelectedSegmentId = null;
    let playbackSyncHandler = null;
    let currentTimelineSubtab = "filmstrip";
    let currentContrastiveData = null;
    let initialContrastiveProfiles = null;
    let selectedContrastGenre = "";
    let currentContrastFilter = "all";
    let areAllContrastDetailsExpanded = false;

    // State quản lý Interactive Multimodal Analysis Workspace (6 Khu vực)
    let currentWorkspaceMode = "scroll"; // "scroll" | "focus"
    let currentRightColumnMode = "verdict"; // "verdict" | "evidence"
    let currentMultimodalTab = "audio"; // "audio" | "visual" | "fused"
    let activeSelectedRuleIdx = 0;
    let currentExpExplorerTab = "aspect"; // "aspect" | "groups" | "whatif"
    let selectedAspectKey = null; // null -> chọn tiêu chí đầu tiên; "__ALL__" -> hiển thị tất cả tiêu chí

    // Tham chiếu các phần tử DOM chính
    const mediaInput =
        document.getElementById("media-file-input") ||
        document.getElementById("video-file-input");
    const mediaDropzone =
        document.getElementById("media-dropzone") ||
        document.getElementById("video-dropzone");
    const videoPlayer = document.getElementById("main-video-player");
    const audioPlayer = document.getElementById("main-audio-player");
    const videoContainer = document.getElementById("video-player-container");
    const audioContainer = document.getElementById("audio-player-container");
    const btnChangeVideo = document.getElementById("btn-change-video");
    const btnChangeAudio = document.getElementById("btn-change-audio");
    const videoFileName = document.getElementById("video-file-name");
    const audioFileName = document.getElementById("audio-file-name");
    const videoAspectBadge = document.getElementById("video-aspect-badge");

    const micRecordBtn = document.getElementById("mic-record-btn");
    const micStopBtn = document.getElementById("mic-stop-btn");
    const loadingOverlay = document.getElementById("loading-overlay");
    const toastBox = document.getElementById("toast-box");

    // Gắn sự kiện phản hồi âm thanh bền vững cho trình phát Video & Audio với Quả cầu 3D Brand Identity
    function bindMediaElementToBrandOrb(mediaEl) {
        if (!mediaEl) return;
        mediaEl.addEventListener("play", () => {
            audioWave.connectMediaElement(mediaEl);
        });
        mediaEl.addEventListener("playing", () => {
            audioWave.connectMediaElement(mediaEl);
        });
        mediaEl.addEventListener("pause", () => {
            audioWave.disconnect();
        });
        mediaEl.addEventListener("ended", () => {
            audioWave.disconnect();
        });
        mediaEl.addEventListener("emptied", () => {
            audioWave.disconnect();
        });
    }
    bindMediaElementToBrandOrb(videoPlayer);
    bindMediaElementToBrandOrb(audioPlayer);

    /**
     * Nguyên tắc thiết kế cốt lõi: CHỈ ĐƯỢC PHÉP TỒN TẠI 1 INPUT TẠI 1 THỜI ĐIỂM.
     * Dọn dẹp triệt để phương tiện đối ứng (video hoặc audio), ngắt kết nối âm thanh và giải phóng bộ nhớ.
     */
    function clearAllInputsExcept(keepType) {
        if (keepType !== "video") {
            if (videoPlayer) {
                videoPlayer.pause();
                videoPlayer.removeAttribute("src");
                videoPlayer.load();
            }
            if (videoContainer) videoContainer.style.display = "none";
            if (videoFileName) videoFileName.innerHTML = "";
        }

        if (keepType !== "audio") {
            if (audioPlayer) {
                audioPlayer.pause();
                audioPlayer.removeAttribute("src");
                audioPlayer.load();
            }
            if (audioContainer) audioContainer.style.display = "none";
            if (audioFileName) audioFileName.innerHTML = "";
        }

        if (mediaInput) mediaInput.value = "";
        if (mediaDropzone) {
            mediaDropzone.style.display = keepType === null ? "block" : "none";
        }

        // Nếu đang ghi âm từ microphone mà nạp input khác -> dừng thu âm
        if (recorder && recorder.isRecording) {
            if (micStopBtn) micStopBtn.click();
        }

        audioWave.disconnect();
        activeMediaType = keepType;

        if (keepType === null) {
            resetAnalysisUI();
        }
    }

    function resetAnalysisUI() {
        currentAnalysis = null;
        currentTimelineMode = "raw";
        currentRawChunks = [];
        currentMergedSegments = [];
        currentTimelineSeries = [];
        currentOverallExplanation = null;
        currentOverallModalities = null;
        activeSelectedSegmentId = null;

        if (audioWave && typeof audioWave.setState === "function") {
            audioWave.setState("idle", { resetCompletion: true });
        }

        const titleEl = document.getElementById("verdict-genre-title");
        const confEl = document.getElementById("confidence-number");
        const durEl = document.getElementById("verdict-duration");

        if (titleEl) {
            titleEl.innerText = "Chưa có dữ liệu";
            titleEl.style.color = "var(--text-tertiary)";
        }
        if (confEl) confEl.innerText = "--.-%";
        if (durEl) durEl.innerText = "Thời lượng phân tích: --";

        const stickyGenre = document.getElementById("sticky-verdict-genre");
        const stickyConf = document.getElementById("sticky-verdict-conf");
        const stickyDot = document.getElementById("sticky-verdict-dot");
        if (stickyGenre) stickyGenre.innerText = "Chưa có dữ liệu";
        if (stickyConf) stickyConf.innerText = "--%";
        if (stickyDot) stickyDot.style.backgroundColor = "var(--text-tertiary)";

        const winnerWrap = document.getElementById("prob-top-winner-wrap");
        if (winnerWrap) {
            winnerWrap.innerHTML = `
        <div class="prob-empty-notice" style="font-size: 12px; color: var(--text-tertiary); font-style: italic; padding: 6px 0;">
          <i class="fa-solid fa-chart-simple" style="margin-right: 6px;"></i>Chưa có phân phối xác suất
        </div>
      `;
        }

        const otherList = document.getElementById("probabilities-other-list");
        if (otherList) otherList.innerHTML = "";

        const headlineEl = document.getElementById("xai-headline");
        const bodyEl = document.getElementById("xai-body-text");
        const previewEl = document.getElementById("overview-insight-preview");
        if (headlineEl) headlineEl.innerText = "Chưa có kết quả phân tích";
        if (bodyEl)
            bodyEl.innerText =
                "Hãy tải lên video/audio hoặc thu âm trực tiếp để hệ thống thực hiện suy luận mờ 2 tầng và sinh lời lý giải khoa học.";
        if (previewEl)
            previewEl.innerText =
                "Hãy tải lên video/audio hoặc chọn một mẫu di sản chuẩn phía trên để hệ thống thực hiện suy luận mờ 2 tầng và tổng hợp nhận định chính.";

        const xaiBanner = document.getElementById("xai-segment-banner");
        if (xaiBanner) xaiBanner.style.display = "none";
        const xaiModsGrid = document.getElementById("xai-modalities-grid");
        if (xaiModsGrid) xaiModsGrid.style.display = "none";

        renderConceptPills([]);

        const rulesContainer = document.getElementById("rules-list-container");
        if (rulesContainer) {
            rulesContainer.innerHTML =
                "<span style='font-size: 12px; color: var(--text-tertiary);'>Chưa có luật kích hoạt.</span>";
        }
        const ruleCountBadge = document.getElementById("rule-explorer-count-badge");
        if (ruleCountBadge) ruleCountBadge.innerText = "0 luật kích hoạt";
        const ruleDetailEl = document.getElementById("rule-explorer-detail");
        if (ruleDetailEl) {
            ruleDetailEl.innerHTML = `
              <div class="rule-detail-empty">
                <i class="fa-solid fa-scale-balanced"></i>
                <h4>Chọn một luật suy luận bên trái để xem chi tiết</h4>
                <p>Mỗi luật biểu diễn chuỗi suy diễn từ các khái niệm trung gian (CBM Concepts) đến kết luận thể loại di sản cùng độ tin cậy và số mẫu huấn luyện hỗ trợ.</p>
              </div>
            `;
        }

        const stripContainer = document.getElementById("timeline-scroll-strip");
        if (stripContainer) {
            stripContainer.innerHTML =
                "<p style='color: var(--text-secondary); font-size: 12px; grid-column: 1 / -1; padding: 12px 0;'>Chưa có dữ liệu phân đoạn. Hãy chọn một mẫu phía trên hoặc tải lên file để phân tích.</p>";
        }
        const modeBar = document.getElementById("timeline-mode-bar");
        if (modeBar) modeBar.style.display = "none";
        const ribbonWrap = document.getElementById(
            "genre-timeline-ribbon-wrap",
        );
        if (ribbonWrap) ribbonWrap.style.display = "none";
        const svgChart = document.getElementById("timeline-svg-chart");
        if (svgChart) svgChart.innerHTML = "";
        const specContainer = document.getElementById(
            "spectrogram-grid-container",
        );
        if (specContainer) specContainer.innerHTML = "";
        const radarContainer = document.getElementById("radar-bars-container");
        if (radarContainer) radarContainer.innerHTML = "";

        const mmAudioPill = document.getElementById("mm-pill-audio-count");
        const mmVisualPill = document.getElementById("mm-pill-visual-count");
        const mmFusedPill = document.getElementById("mm-pill-fused-count");
        if (mmAudioPill) mmAudioPill.innerText = "7 trục";
        if (mmVisualPill) mmVisualPill.innerText = "0 đoạn";
        if (mmFusedPill) mmFusedPill.innerText = "Late Fusion";

        const tableBody = document.getElementById("contrastive-table-body");
        if (tableBody) {
            tableBody.innerHTML = `
        <tr>
          <td colspan="3" style="text-align: center; padding: 36px 20px; color: var(--text-tertiary);">
            <i class="fa-solid fa-arrows-split-up-and-left" style="font-size: 24px; margin-bottom: 8px; display: block; opacity: 0.5;"></i>
            Vui lòng chọn một mẫu âm thanh/video hoặc tải lên tệp để kích hoạt phần Giải thích đối chứng (Contrastive Explanation).
          </td>
        </tr>
      `;
        }
        const aspectList = document.getElementById("aspect-selector-list");
        if (aspectList) {
            aspectList.innerHTML = `<div class="aspect-empty-hint">Chưa có tiêu chí đối chiếu</div>`;
        }
        const aspectBadge = document.getElementById("aspect-count-badge");
        if (aspectBadge) aspectBadge.innerText = "0";
        selectedAspectKey = null;

        const metaWinner = document.getElementById("meta-winner-tag");
        const metaRunner = document.getElementById("meta-runnerup-tag");
        if (metaWinner) metaWinner.innerText = "Dự đoán: Đang chờ";
        if (metaRunner) metaRunner.innerText = "Top 2: Đang chờ";
        const contrastSelect = document.getElementById("contrast-genre-select");
        if (contrastSelect) {
            contrastSelect.innerHTML = `<option value="">Đối chứng (Hạng 2): Đang chờ</option>`;
            contrastSelect.disabled = true;
        }
        const threeGroups = document.getElementById("contrastive-three-groups");
        if (threeGroups) threeGroups.style.display = "none";
        const filterBar = document.getElementById("contrastive-filter-bar");
        if (filterBar) filterBar.style.display = "none";
        const thTop1Prob = document.getElementById("th-top1-prob");
        const thTop2Prob = document.getElementById("th-top2-prob");
        if (thTop1Prob) thTop1Prob.innerText = "--%";
        if (thTop2Prob) thTop2Prob.innerText = "--%";
        currentContrastiveData = null;
        initialContrastiveProfiles = null;
        selectedContrastGenre = "";

        // Reset khu vực 6 (Tổng kết Key Findings)
        const tkVerdictVal = document.getElementById("takeaway-verdict-val");
        const tkVerdictSub = document.getElementById("takeaway-verdict-sub");
        const tkModVal = document.getElementById("takeaway-modality-val");
        const tkModSub = document.getElementById("takeaway-modality-sub");
        const tkRuleVal = document.getElementById("takeaway-rule-val");
        const tkRuleSub = document.getElementById("takeaway-rule-sub");
        const tkDiscVal = document.getElementById("takeaway-disc-val");
        const tkDiscSub = document.getElementById("takeaway-disc-sub");
        if (tkVerdictVal) tkVerdictVal.innerText = "Chưa có dữ liệu";
        if (tkVerdictSub) tkVerdictSub.innerText = "Độ tin cậy: --%";
        if (tkModVal) tkModVal.innerText = "Đang chờ phân tích";
        if (tkModSub) tkModSub.innerText = "Âm thanh & Hình ảnh";
        if (tkRuleVal) tkRuleVal.innerText = "0 luật kích hoạt";
        if (tkRuleSub) tkRuleSub.innerText = "Suy luận mờ 2 tầng";
        if (tkDiscVal) tkDiscVal.innerText = "Chưa đối chiếu";
        if (tkDiscSub) tkDiscSub.innerText = "Khoảng cách dẫn trước: --%";

        // Xóa trạng thái và dữ liệu Bản đồ bằng chứng đa phương thức (Evidence Map)
        clearEvidenceMap();
    }

    // Nút thu nhỏ / mở rộng danh sách xác suất ô dự đoán
    let isVerdictExpanded = false;
    const btnToggleVerdict = document.getElementById("btn-toggle-verdict");
    const probsCollapseWrapper = document.getElementById(
        "probs-collapsible-wrapper",
    );
    const verdictToggleIcon = document.getElementById("verdict-toggle-icon");

    if (btnToggleVerdict && probsCollapseWrapper && verdictToggleIcon) {
        btnToggleVerdict.addEventListener("click", () => {
            isVerdictExpanded = !isVerdictExpanded;
            if (isVerdictExpanded) {
                probsCollapseWrapper.classList.remove("collapsed");
                btnToggleVerdict.classList.add("expanded");
                btnToggleVerdict.title = "Thu nhỏ danh sách xác suất";
                verdictToggleIcon.className = "fa-solid fa-chevron-up";
            } else {
                probsCollapseWrapper.classList.add("collapsed");
                btnToggleVerdict.classList.remove("expanded");
                btnToggleVerdict.title =
                    "Mở rộng toàn bộ xác suất (5 thể loại khác)";
                verdictToggleIcon.className = "fa-solid fa-chevron-down";
            }
        });
    }

    // =========================================================================
    // 2. KHỞI TẠO TABS CHẾ ĐỘ ĐẦU VÀO
    // =========================================================================
    const tabTriggers = document.querySelectorAll(".tab-trigger");
    const tabContents = document.querySelectorAll(".tab-content");

    function switchTab(targetTab) {
        currentActiveTab = targetTab;
        tabTriggers.forEach((t) => {
            if (t.getAttribute("data-tab") === targetTab) {
                t.classList.add("active");
            } else {
                t.classList.remove("active");
            }
        });
        tabContents.forEach((c) => {
            if (c.id === `tab-${targetTab}`) {
                c.classList.add("active");
            } else {
                c.classList.remove("active");
            }
        });
    }

    tabTriggers.forEach((trigger) => {
        trigger.addEventListener("click", () => {
            const targetTab = trigger.getAttribute("data-tab");
            switchTab(targetTab);

            // Dừng media đang phát nếu người dùng tự bấm chuyển tab
            if (videoPlayer) videoPlayer.pause();
            if (audioPlayer) audioPlayer.pause();
            audioWave.disconnect();
        });
    });

    // Sự kiện nút "Thay đổi tệp" -> Xóa media hiện tại và mở lại dropzone
    if (btnChangeVideo) {
        btnChangeVideo.addEventListener("click", () => {
            clearAllInputsExcept(null);
        });
    }
    if (btnChangeAudio) {
        btnChangeAudio.addEventListener("click", () => {
            clearAllInputsExcept(null);
        });
    }

    // =========================================================================
    // 3. NẠP DANH SÁCH MẪU SẴN CÓ (/api/sample_files)
    // =========================================================================
    fetchSampleFiles();

    async function fetchSampleFiles() {
        try {
            const res = await fetch("/api/sample_files");
            const data = await res.json();
            if (data.status === "success" && data.samples) {
                window.samplesList = data.samples;
                renderSampleChips(data.samples);
                // Không tự động nạp mẫu khi mở trang theo yêu cầu người dùng
            }
        } catch (e) {
            console.warn("Không thể tải danh sách mẫu:", e);
        }
    }

    function renderSampleChips(samples) {
        const container = document.getElementById("sample-chips-container");
        if (!container) return;

        container.innerHTML = "";
        samples.forEach((s) => {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.id = `sample-chip-${s.genre.replace(/\s+/g, "_")}`;
            chip.className = "chip-btn";
            chip.title = `Chạy thử mẫu ${s.title}`;
            chip.innerHTML = `
        <span class="chip-dot" style="background-color: ${s.color};"></span>
        <span style="font-weight: 600;">${s.genre}</span>
        <span style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: 10px;">${s.duration}</span>
      `;

            chip.onclick = (e) => {
                if (e) e.preventDefault();
                analyzeSampleFile(s);
            };

            container.appendChild(chip);
        });
    }

    window.analyzeSampleFile = analyzeSampleFile;

    async function analyzeSampleFile(sample) {
        clearAllInputsExcept(sample.mode === "video" ? "video" : "audio");
        showLoading(
            `Đang phân tích mẫu ${sample.genre}...`,
            "Trích xuất chuỗi phân đoạn 3-5s & 7 trục âm học...",
        );

        try {
            const res = await fetch("/api/analyze_sample", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    sample_path: sample.file_path,
                    mode: sample.mode,
                }),
            });

            if (!res.ok) throw new Error("Phân tích mẫu thất bại");
            const data = await res.json();
            if (!data.file_name)
                data.file_name = sample.title || sample.file_path;
            handleAnalysisResult(data);
            showToast(`Đã phân tích thành công mẫu: ${sample.title}`);
        } catch (err) {
            console.error(err);
            showToast("Lỗi khi nạp mẫu thử: " + err.message);
        } finally {
            hideLoading();
        }
    }

    // =========================================================================
    // 4. XỬ LÝ UPLOAD TỆP ĐA PHƯƠNG TIỆN (VIDEO & AUDIO GỘP CHUNG - TỰ ĐỘNG PHÂN LOẠI)
    // =========================================================================
    function handleMediaFileSelection(file) {
        if (!file) return;
        const isVideo =
            file.type.startsWith("video/") ||
            /\.(mp4|avi|mov|mkv|webm|flv|m4v)$/i.test(file.name);
        const targetMode = isVideo ? "video" : "audio";
        uploadAndAnalyze(file, targetMode);
    }

    setupDropzone("media-dropzone", mediaInput, handleMediaFileSelection);

    function setupDropzone(dropzoneId, inputEl, onFileSelected) {
        const dropzone = document.getElementById(dropzoneId);
        if (!dropzone || !inputEl) return;

        dropzone.addEventListener("click", () => inputEl.click());

        inputEl.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                onFileSelected(e.target.files[0]);
            }
        });

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.classList.remove("dragover");
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                onFileSelected(e.dataTransfer.files[0]);
            }
        });
    }

    async function uploadAndAnalyze(file, mode) {
        let localBlobUrl = null;
        const isVideo =
            mode === "video" ||
            file.type.startsWith("video/") ||
            /\.(mp4|avi|mov|mkv|webm|flv|m4v)$/i.test(file.name);
        const targetMode = isVideo ? "video" : "audio";
        clearAllInputsExcept(targetMode);

        if (targetMode === "video") {
            switchTab("media");
            localBlobUrl = URL.createObjectURL(file);
            videoPlayer.src = localBlobUrl;
            videoPlayer.load();
            if (mediaDropzone) mediaDropzone.style.display = "none";
            if (videoContainer) videoContainer.style.display = "block";
            if (audioContainer) audioContainer.style.display = "none";
            if (videoFileName)
                videoFileName.innerHTML = `<i class="fa-solid fa-film"></i> ${file.name}`;
        } else {
            if (mode === "mic") {
                switchTab("mic");
            } else {
                switchTab("media");
            }
            localBlobUrl = URL.createObjectURL(file);
            audioPlayer.src = localBlobUrl;
            audioPlayer.load();
            if (mediaDropzone) mediaDropzone.style.display = "none";
            if (audioContainer) audioContainer.style.display = "block";
            if (videoContainer) videoContainer.style.display = "none";
            const displayName =
                mode === "mic" ? "Bản ghi âm trực tiếp" : file.name;
            if (audioFileName)
                audioFileName.innerHTML = `<i class="fa-solid fa-file-audio"></i> ${displayName}`;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("mode", targetMode);

        showLoading(
            "Đang tải tệp & xử lý phân đoạn...",
            `Tệp: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`,
        );

        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                body: formData,
            });

            if (!res.ok) throw new Error("Máy chủ trả về mã lỗi " + res.status);
            const data = await res.json();
            if (!data.file_name) data.file_name = file.name;
            if (localBlobUrl) data._localBlobUrl = localBlobUrl;
            handleAnalysisResult(data);
            showToast(
                `Đã hoàn tất phân loại: ${data.prediction} (${data.confidence}%)`,
            );
        } catch (err) {
            console.error(err);
            showToast("Lỗi phân tích tệp: " + err.message);
        } finally {
            hideLoading();
        }
    }

    // =========================================================================
    // XỬ LÝ NHẬP LIÊN KẾT YOUTUBE / VIDEO URL
    // =========================================================================
    const urlVideoInput = document.getElementById("url-video-input");
    const btnAnalyzeUrl = document.getElementById("btn-analyze-url");
    const urlChips = document.querySelectorAll(".url-chip");

    if (btnAnalyzeUrl && urlVideoInput) {
        btnAnalyzeUrl.addEventListener("click", () => {
            const url = urlVideoInput.value.trim();
            if (!url) {
                showToast("Vui lòng nhập hoặc dán liên kết video / YouTube!");
                urlVideoInput.focus();
                return;
            }
            analyzeVideoUrl(url);
        });

        urlVideoInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                btnAnalyzeUrl.click();
            }
        });
    }

    urlChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            const u = chip.getAttribute("data-url");
            if (u && urlVideoInput) {
                urlVideoInput.value = u;
                analyzeVideoUrl(u);
            }
        });
    });

    async function analyzeVideoUrl(url) {
        clearAllInputsExcept("video");
        showLoading(
            "Đang nạp video từ YouTube / URL...",
            "Sử dụng yt-dlp tải luồng & trích xuất phân đoạn 3-5s...",
        );

        try {
            const res = await fetch("/api/analyze_url", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url, mode: "video" }),
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(
                    errData.detail || "Máy chủ trả về mã lỗi " + res.status,
                );
            }

            const data = await res.json();
            handleAnalysisResult(data);
            showToast(
                `Đã phân tích thành công: ${data.file_name || data.prediction}`,
            );
        } catch (err) {
            console.error(err);
            showToast("Lỗi khi tải hoặc phân tích video URL: " + err.message);
        } finally {
            hideLoading();
        }
    }

    // =========================================================================
    // 5. GHI ÂM TRỰC TIẾP TỪ MICROPHONE (LIVE MIC RECORDER)
    // =========================================================================
    if (micRecordBtn) {
        micRecordBtn.addEventListener("click", async () => {
            micRecordBtn.style.display = "none";
            if (micStopBtn) micStopBtn.style.display = "inline-flex";

            const started = await recorder.startRecording();

            if (!started) {
                micRecordBtn.style.display = "inline-flex";
                if (micStopBtn) micStopBtn.style.display = "none";
            }
        });
    }

    if (micStopBtn) {
        micStopBtn.addEventListener("click", async () => {
            micStopBtn.style.display = "none";
            if (micRecordBtn) micRecordBtn.style.display = "inline-flex";

            audioWave.disconnect();
            const recordResult = recorder.stopRecording();

            if (!recordResult) return;

            if (recordResult.duration < 5.0) {
                showToast(
                    "Bản ghi ngắn hơn 5s. Khuyến nghị ghi 6-15s để đạt độ tin cậy chuẩn!",
                );
            }

            // Gửi buffer WAV lên backend phân tích
            uploadAndAnalyze(recordResult.file, "mic");
        });
    }

    // =========================================================================
    // 6. XỬ LÝ & HIỂN THỊ KẾT QUẢ VỚI GSAP CHOREOGRAPHY
    // =========================================================================
    function handleAnalysisResult(result) {
        currentAnalysis = result;
        currentAnalysis.base_probs = Object.assign({}, result.probs);
        currentAnalysis.base_rules = result.triggered_rules
            ? JSON.parse(JSON.stringify(result.triggered_rules))
            : [];
        currentOverallExplanation = result.explanation;
        currentOverallModalities = result.modalities || null;
        currentRawChunks = result.chunks || [];
        currentMergedSegments = result.merged_segments || [];
        currentTimelineSeries = result.timeline_series || [];

        // 1. Kích hoạt media player và gắn Web Audio API
        setupMediaPlayer(result);

        // Mặc định hiển thị Chế độ A (Nhận diện) ở cột phải khi đưa video mới vào
        if (typeof switchRightColumnMode === "function") {
            switchRightColumnMode("verdict");
        }

        // 2. Điền thông tin Verdict chính
        renderVerdict(result);

        // 3. Animate 6 thanh xác suất bằng GSAP
        animateProbabilityBars(result.probs, result.prediction);

        // 4. Render lời lý giải tự nhiên XAI
        renderXaiExplanation(result.explanation);

        // 5. Render hiển thị theo từng phương thức tổng thể (Requirement 2)
        if (result.modalities) {
            updateModalitiesGrid(
                result.modalities,
                result.prediction,
                result.confidence,
                false,
            );
        }

        // 6. Render CBM Concept Pills tương tác
        renderConceptPills(result.active_concepts);

        // 7. Render Triggered Symbolic Rules
        renderTriggeredRules(result.triggered_rules);

        // 8. Render Segment Timeline & Filmstrip (Gộp thông minh, Ribbon & Biểu đồ thời gian)
        renderTimelineAndSegments(
            result.chunks,
            result.mode,
            result.merged_segments,
        );
        renderGenreTimelineRibbon(result.chunks, result.duration_seconds);
        renderTimelineChart(
            result.timeline_series || result.chunks,
            result.duration_seconds,
        );

        // 9. Render Ma trận Đối sánh Phân biệt Di sản & Thử nghiệm What-If (Phương án 3)
        if (result.contrastive_analysis) {
            renderContrastiveMatrix(
                result.contrastive_analysis,
                result.probs,
                result.prediction,
            );
        }

        // 10. Render Multimodal Evidence Map (Bản đồ bằng chứng đa phương thức)
        if (result.evidence_map) {
            renderEvidenceMap(result.evidence_map);
        }

        // 11. Render Tổng kết Key Findings (Khu vực 6)
        renderExecutiveSummary(result);

        // 12. Chuyển trạng thái Quả cầu 3D Brand Identity sang Completed (hoặc Audio-Reactive nếu đang phát)
        if (audioWave && typeof audioWave.setCompletedState === "function") {
            audioWave.setCompletedState(
                result.genre_color || GENRE_COLORS[result.prediction],
            );
        }
    }

    function setupMediaPlayer(result) {
        if (!result.media_url && !result._localBlobUrl) return;

        if (result.mode === "video") {
            clearAllInputsExcept("video");
            switchTab("media");

            // Ẩn dropzone kéo thả, hiển thị trình phát video
            if (mediaDropzone) mediaDropzone.style.display = "none";
            if (videoContainer) videoContainer.style.display = "block";
            if (audioContainer) audioContainer.style.display = "none";

            const displayName =
                result.file_name || result.prediction || "Video biểu diễn";
            if (videoFileName) {
                videoFileName.innerHTML = `<i class="fa-solid fa-film"></i> ${displayName}`;
            }

            // Ưu tiên blob URL cục bộ nếu có
            const mediaSrc = result._localBlobUrl || result.media_url;
            if (videoPlayer.src !== mediaSrc) {
                videoPlayer.src = mediaSrc;
                videoPlayer.load();
            }

            // Tự động nhận diện aspect ratio và cập nhật badge
            videoPlayer.onloadedmetadata = () => {
                const vw = videoPlayer.videoWidth;
                const vh = videoPlayer.videoHeight;
                if (vw && vh && videoAspectBadge) {
                    const ratio = vw / vh;
                    let label = `${vw}×${vh}`;
                    if (Math.abs(ratio - 16 / 9) < 0.15) label = "16:9 HD";
                    else if (Math.abs(ratio - 4 / 3) < 0.15)
                        label = "4:3 Classic";
                    else if (Math.abs(ratio - 1.0) < 0.08) label = "1:1 Vuông";
                    videoAspectBadge.innerText = label;
                }
            };

            videoPlayer.onplay = () => {
                audioWave.connectMediaElement(videoPlayer);
            };
            videoPlayer.onpause = () => {
                audioWave.disconnect();
            };
            videoPlayer.onended = () => {
                audioWave.disconnect();
            };

            // Đồng bộ thời gian phát với thanh Ribbon & Biểu đồ diễn biến
            setupPlaybackTimeSync(videoPlayer, result.duration_seconds);
        } else {
            clearAllInputsExcept("audio");
            if (result.mode === "mic") {
                switchTab("mic");
            } else {
                switchTab("media");
            }

            // Ẩn dropzone kéo thả, hiển thị trình phát audio
            if (mediaDropzone) mediaDropzone.style.display = "none";
            if (audioContainer) audioContainer.style.display = "block";
            if (videoContainer) videoContainer.style.display = "none";

            const displayName =
                result.file_name ||
                (result.mode === "mic"
                    ? "Bản ghi âm trực tiếp"
                    : result.prediction) ||
                "Tệp âm thanh";
            if (audioFileName) {
                audioFileName.innerHTML = `<i class="fa-solid fa-file-audio"></i> ${displayName}`;
            }

            const mediaSrc = result._localBlobUrl || result.media_url;
            if (audioPlayer.src !== mediaSrc) {
                audioPlayer.src = mediaSrc;
                audioPlayer.load();
            }

            audioPlayer.onplay = () => {
                audioWave.connectMediaElement(audioPlayer);
            };
            audioPlayer.onpause = () => {
                audioWave.disconnect();
            };
            audioPlayer.onended = () => {
                audioWave.disconnect();
            };

            // Đồng bộ thời gian phát với thanh Ribbon & Biểu đồ diễn biến
            setupPlaybackTimeSync(audioPlayer, result.duration_seconds);
        }
    }

    function renderVerdict(result) {
        const titleEl = document.getElementById("verdict-genre-title");
        const confEl = document.getElementById("confidence-number");
        const durEl = document.getElementById("verdict-duration");
        const gColor = result.genre_color || GENRE_COLORS[result.prediction] || "#C84B31";

        if (titleEl) {
            titleEl.innerText = result.prediction;
            titleEl.style.color = gColor;
        }

        if (durEl) {
            durEl.innerText = `${result.duration_seconds}s (${result.mode.toUpperCase()})`;
        }

        // Cập nhật Sticky Mini Verdict trên thanh điều hướng nội bộ
        const stickyGenre = document.getElementById("sticky-verdict-genre");
        const stickyConf = document.getElementById("sticky-verdict-conf");
        const stickyDot = document.getElementById("sticky-verdict-dot");
        if (stickyGenre) stickyGenre.innerText = result.prediction;
        if (stickyConf) stickyConf.innerText = `${Number(result.confidence || 0).toFixed(1)}%`;
        if (stickyDot) stickyDot.style.backgroundColor = gColor;

        // GSAP Counter Animation cho % Độ tin cậy
        if (confEl && window.gsap) {
            const targetVal = result.confidence;
            const counter = { val: 0 };
            gsap.to(counter, {
                val: targetVal,
                duration: 1.2,
                ease: "power2.out",
                onUpdate: () => {
                    confEl.innerText = `${counter.val.toFixed(1)}%`;
                },
            });
        } else if (confEl) {
            confEl.innerText = `${result.confidence}%`;
        }
    }

    function animateProbabilityBars(probs, winnerGenre) {
        if (!probs) return;

        // Sắp xếp toàn bộ các thể loại theo thứ tự xác suất GIẢM DẦN
        const sortedEntries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        if (sortedEntries.length === 0) return;

        // Nhãn được dự đoán cao nhất (Top 1)
        const [topGenre, topScore] = sortedEntries[0];
        const otherGenres = sortedEntries.slice(1);

        // 1. Render Top Winner Genre vào #prob-top-winner-wrap (Luôn hiển thị ban đầu)
        const winnerWrap = document.getElementById("prob-top-winner-wrap");
        if (winnerWrap) {
            const topSafeId = topGenre.replace(/\s+/g, "_");
            winnerWrap.innerHTML = `
        <div class="prob-row winner" id="prob-row-${topSafeId}">
          <span class="prob-genre-name">
            <i class="fa-solid fa-crown" style="color: var(--accent-amber); font-size: 11px; margin-right: 5px;"></i>${topGenre}
          </span>
          <div class="prob-track">
            <div class="prob-fill" id="fill-${topSafeId}" style="width: 0%;"></div>
          </div>
          <span class="prob-val-text" id="val-${topSafeId}">0.0%</span>
        </div>
      `;

            const fillEl = winnerWrap.querySelector(".prob-fill");
            const valEl = winnerWrap.querySelector(".prob-val-text");

            if (window.gsap && fillEl && valEl) {
                gsap.to(fillEl, {
                    width: `${topScore}%`,
                    duration: 0.9,
                    ease: "power3.out",
                    delay: 0.05,
                });
                const counter = { val: 0 };
                gsap.to(counter, {
                    val: topScore,
                    duration: 0.9,
                    ease: "power3.out",
                    delay: 0.05,
                    onUpdate: () => {
                        valEl.innerText = `${counter.val.toFixed(1)}%`;
                    },
                });
            } else if (fillEl && valEl) {
                fillEl.style.width = `${topScore}%`;
                valEl.innerText = `${topScore.toFixed(1)}%`;
            }
        }

        // 2. Render 5 thể loại còn lại theo thứ tự GIẢM DẦN vào #probabilities-other-list
        const otherList = document.getElementById("probabilities-other-list");
        if (otherList) {
            otherList.innerHTML = "";

            otherGenres.forEach(([genre, score], idx) => {
                const safeId = genre.replace(/\s+/g, "_");
                const row = document.createElement("div");
                row.className = "prob-row";
                row.id = `prob-row-${safeId}`;

                row.innerHTML = `
          <span class="prob-genre-name">${genre}</span>
          <div class="prob-track">
            <div class="prob-fill" id="fill-${safeId}" style="width: 0%;"></div>
          </div>
          <span class="prob-val-text" id="val-${safeId}">0.0%</span>
        `;
                otherList.appendChild(row);

                const fillEl = row.querySelector(".prob-fill");
                const valEl = row.querySelector(".prob-val-text");

                if (window.gsap && fillEl && valEl) {
                    gsap.to(fillEl, {
                        width: `${score}%`,
                        duration: 0.9,
                        ease: "power3.out",
                        delay: 0.08 + idx * 0.04,
                    });
                    const counter = { val: 0 };
                    gsap.to(counter, {
                        val: score,
                        duration: 0.9,
                        ease: "power3.out",
                        delay: 0.08 + idx * 0.04,
                        onUpdate: () => {
                            valEl.innerText = `${counter.val.toFixed(1)}%`;
                        },
                    });
                } else if (fillEl && valEl) {
                    fillEl.style.width = `${score}%`;
                    valEl.innerText = `${score.toFixed(1)}%`;
                }
            });
        }
    }

    function renderXaiExplanation(explanation) {
        if (!explanation) return;

        const headlineEl = document.getElementById("xai-headline");
        const bodyEl = document.getElementById("xai-body-text");
        const summaryBodyEl = document.getElementById("summary-xai-body-text");
        const previewEl = document.getElementById("overview-insight-preview");
        const citationsEl = document.getElementById("xai-citations-list");
        const summaryCitationsEl = document.getElementById("summary-citations-list");

        if (headlineEl) headlineEl.innerText = explanation.headline;
        if (explanation.narrative) {
            let formatted = explanation.narrative
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
                .replace(/\n/g, "<br>");
            if (bodyEl) bodyEl.innerHTML = formatted;
            if (summaryBodyEl) summaryBodyEl.innerHTML = formatted;

            if (previewEl) {
                const plainSummary = explanation.narrative
                    .replace(/\*\*([^*]+)\*\*/g, "$1")
                    .replace(/\n+/g, " ")
                    .trim();
                previewEl.innerText = plainSummary;
            }
        }

        if (explanation.citations) {
            const populateCitations = (targetEl) => {
                if (!targetEl) return;
                targetEl.innerHTML = "";
                explanation.citations.forEach((cit) => {
                    const li = document.createElement("li");
                    li.className = "citation-item";
                    li.innerHTML = `<i class="fa-solid fa-bookmark" style="color: var(--accent-jade); font-size: 10px;"></i> ${cit}`;
                    targetEl.appendChild(li);
                });
            };
            populateCitations(citationsEl);
            populateCitations(summaryCitationsEl);
        }

        // GSAP Fade-in
        if (window.gsap) {
            gsap.from("#xai-card", {
                opacity: 0,
                y: 10,
                duration: 0.6,
                ease: "power2.out",
            });
        }
    }

    function renderConceptPills(concepts) {
        const audioContainer = document.getElementById("audio-pills-container");
        const visualContainer = document.getElementById(
            "visual-pills-container",
        );
        const visualGroup = document.getElementById("visual-pills-group");

        if (audioContainer) audioContainer.innerHTML = "";
        if (visualContainer) visualContainer.innerHTML = "";

        if (!concepts || concepts.length === 0) {
            if (audioContainer) {
                audioContainer.innerHTML = `<span style="font-size: 12px; color: var(--text-tertiary);">Chưa có khái niệm âm thanh nào được kích hoạt.</span>`;
            }
            if (visualContainer) {
                visualContainer.innerHTML = `<span style="font-size: 12px; color: var(--text-tertiary);">Chưa có khái niệm thị giác nào được kích hoạt.</span>`;
            }
            return;
        }

        let audioCount = 0;
        let visualCount = 0;

        concepts.forEach((c) => {
            const isAudio = c.type === "audio";
            if (isAudio) audioCount++;
            else visualCount++;

            const pill = document.createElement("button");
            pill.type = "button";
            pill.className = `concept-pill ${isAudio ? "audio-pill" : "visual-pill"} active`;
            pill.id = `pill-${c.concept_id}`;
            pill.setAttribute("data-concept-id", c.concept_id);
            pill.setAttribute("data-enabled", "true");
            pill.title = `${c.concept_sentence || c.concept_id} (Bấm để bật/tắt thử nghiệm giả định)`;

            const shortAspect = c.aspect
                ? c.aspect.replace(/_/g, " ")
                : "Đặc trưng";
            const tsHint =
                c.timestamps && c.timestamps.length
                    ? `• ${c.timestamps[0]}`
                    : "";

            pill.innerHTML = `
        <span class="pill-check">✓</span>
        <span style="font-weight: 600;">${c.concept_id}</span>
        <span>${shortAspect}</span>
        <span class="pill-pct">${tsHint}</span>
      `;

            // Click: Can thiệp CBM & Fly-to 3D
            pill.addEventListener("click", (e) => {
                e.preventDefault();
                toggleConceptIntervention(c.concept_id, pill);
            });

            if (isAudio && audioContainer) {
                audioContainer.appendChild(pill);
            } else if (!isAudio && visualContainer) {
                visualContainer.appendChild(pill);
            }
        });

        if (audioContainer && audioCount === 0) {
            audioContainer.innerHTML = `<span style="font-size: 12px; color: var(--text-tertiary);">Không có khái niệm âm thanh trong phân đoạn này.</span>`;
        }

        if (visualGroup) {
            if (currentMultimodalTab === "visual") {
                visualGroup.style.display = "block";
            } else if (currentMultimodalTab === "audio") {
                visualGroup.style.display = "none";
            } else {
                visualGroup.style.display = visualCount > 0 ? "block" : "none";
            }
        }
        if (visualContainer && visualCount === 0) {
            visualContainer.innerHTML = `<span style="font-size: 12px; color: var(--text-tertiary);">Chế độ chỉ có âm thanh (không có dữ liệu hình ảnh).</span>`;
        }

        const mmAudioPill = document.getElementById("mm-pill-audio-count");
        if (mmAudioPill) {
            mmAudioPill.innerText = audioCount > 0 ? `${audioCount} khái niệm` : "7 trục";
        }

        // Reset badge trạng thái điều chỉnh
        const badge = document.getElementById("cbm-intervention-badge");
        if (badge) {
            badge.className = "tag-badge jade";
            badge.innerText = "Thử nghiệm bật/tắt";
        }

        // Đảm bảo tất cả các pills hiển thị inline-flex, để CSS kiểm soát opacity theo trạng thái enabled/disabled
        document.querySelectorAll(".concept-pill").forEach((p) => {
            p.style.display = "inline-flex";
        });
    }

    function updateInterventionBadge() {
        const badge = document.getElementById("cbm-intervention-badge");
        if (!badge || !currentAnalysis || !currentAnalysis.active_concepts)
            return;
        const disabledCount = currentAnalysis.active_concepts.filter(
            (c) => c.enabled === false,
        ).length;
        if (disabledCount > 0) {
            badge.className = "tag-badge amber";
            badge.innerText = `Đang can thiệp (${disabledCount} đã tắt)`;
        } else {
            badge.className = "tag-badge jade";
            badge.innerText = "Thử nghiệm bật/tắt";
        }
    }

    // =========================================================================
    // 7. CBM CAN THIỆP KHÁI NIỆM TRUNG GIAN (/api/intervene)
    // =========================================================================
    async function toggleConceptIntervention(conceptId, pillEl) {
        if (!currentAnalysis) return;

        const isCurrentlyEnabled =
            pillEl.getAttribute("data-enabled") === "true";
        const newState = !isCurrentlyEnabled;

        pillEl.setAttribute("data-enabled", String(newState));
        if (newState) {
            pillEl.classList.remove("disabled");
            pillEl.classList.add("active");
            pillEl.querySelector(".pill-check").innerText = "✓";
        } else {
            pillEl.classList.remove("active");
            pillEl.classList.add("disabled");
            pillEl.querySelector(".pill-check").innerText = "✗";
        }

        // 1. Bay camera 3D tới nút concept (kiểm tra an toàn tránh lỗi undefined)
        if (
            window.graph3d &&
            typeof window.graph3d.flyToConcept === "function"
        ) {
            try {
                window.graph3d.flyToConcept(conceptId);
            } catch (err) {
                console.warn("[3D Graph] flyToConcept:", err);
            }
        }

        // 2. Cập nhật trạng thái enabled trong active_concepts
        if (currentAnalysis.active_concepts) {
            const cObj = currentAnalysis.active_concepts.find(
                (c) => c.concept_id === conceptId,
            );
            if (cObj) cObj.enabled = newState;
        }

        // Cập nhật badge đếm số khái niệm đang tắt
        updateInterventionBadge();

        // 3. Gửi request can thiệp lên Backend
        try {
            const res = await fetch("/api/intervene", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    base_probs: currentAnalysis.base_probs,
                    current_probs: currentAnalysis.probs,
                    concept_id: conceptId,
                    is_enabled: newState,
                    all_concepts: currentAnalysis.active_concepts,
                    base_rules:
                        currentAnalysis.base_rules ||
                        currentAnalysis.triggered_rules,
                }),
            });

            if (!res.ok) throw new Error("Can thiệp thất bại");
            const data = await res.json();

            // Cập nhật lại state xác suất
            currentAnalysis.probs = data.probs;
            currentAnalysis.prediction = data.prediction;
            currentAnalysis.confidence = data.confidence;

            // Animate mượt mà lại 6 thanh xác suất qua GSAP mà không làm nháy trang
            updateProbabilityBarsSmooth(
                data.probs,
                data.prediction,
                data.confidence,
            );

            // Cập nhật thẻ kết quả hợp nhất ở Modalities Grid
            if (currentOverallModalities) {
                if (!currentOverallModalities.fused)
                    currentOverallModalities.fused = {};
                currentOverallModalities.fused.prediction = data.prediction;
                currentOverallModalities.fused.confidence = data.confidence;
                currentOverallModalities.fused.method =
                    "Late Fusion Bayesian + Can thiệp CBM Phản sự thực";
                updateModalitiesGrid(
                    currentOverallModalities,
                    data.prediction,
                    data.confidence,
                    false,
                );
            }

            // Cập nhật lại danh sách Luật kích hoạt
            if (data.triggered_rules) {
                currentAnalysis.triggered_rules = data.triggered_rules;
                renderTriggeredRules(data.triggered_rules);
            }

            // Cập nhật lại Bảng Ma trận Đối sánh Phân biệt & What-If
            if (data.contrastive_analysis) {
                renderContrastiveMatrix(
                    data.contrastive_analysis,
                    data.probs,
                    data.prediction,
                );
            }

            // Cập nhật lại Bản đồ bằng chứng đa phương thức (Evidence Map) sau khi can thiệp CBM
            if (data.evidence_map) {
                renderEvidenceMap(data.evidence_map);
            }

            // Cập nhật lại Tổng kết Key Findings (Khu vực 6)
            renderExecutiveSummary(currentAnalysis);

            showToast(data.explanation_note);
        } catch (e) {
            console.error(e);
            showToast("Lỗi can thiệp CBM: " + e.message);
        }
    }

    function updateProbabilityBarsSmooth(probs, winnerGenre, confidence) {
        const titleEl = document.getElementById("verdict-genre-title");
        const confEl = document.getElementById("confidence-number");
        const gColor = GENRE_COLORS[winnerGenre] || "#C84B31";

        if (titleEl) {
            titleEl.innerText = winnerGenre;
            titleEl.style.color = gColor;
        }

        // Cập nhật Sticky Mini Verdict trên thanh điều hướng nội bộ
        const stickyGenre = document.getElementById("sticky-verdict-genre");
        const stickyConf = document.getElementById("sticky-verdict-conf");
        const stickyDot = document.getElementById("sticky-verdict-dot");
        if (stickyGenre) stickyGenre.innerText = winnerGenre;
        if (stickyConf) stickyConf.innerText = `${Number(confidence || 0).toFixed(1)}%`;
        if (stickyDot) stickyDot.style.backgroundColor = gColor;

        if (confEl && window.gsap) {
            const counter = { val: parseFloat(confEl.innerText) || 50 };
            gsap.to(counter, {
                val: confidence,
                duration: 0.6,
                ease: "power2.out",
                onUpdate: () => {
                    confEl.innerText = `${counter.val.toFixed(1)}%`;
                },
            });
        } else if (confEl) {
            confEl.innerText = `${confidence.toFixed(1)}%`;
        }

        animateProbabilityBars(probs, winnerGenre);
    }

    // =========================================================================
    // 8. RENDER LUẬT TƯỢNG TRƯNG KÍCH HOẠT (RULE EXPLORER MASTER-DETAIL)
    // =========================================================================
    function renderTriggeredRules(rules) {
        const container = document.getElementById("rules-list-container");
        const countBadge = document.getElementById("rule-explorer-count-badge");
        const detailPanel = document.getElementById("rule-explorer-detail");
        if (!container) return;

        container.innerHTML = "";
        const ruleList = Array.isArray(rules) ? rules : [];

        if (countBadge) {
            countBadge.innerText = `${ruleList.length} luật kích hoạt`;
        }

        if (ruleList.length === 0) {
            container.innerHTML =
                "<p style='color: var(--text-secondary); font-size: 12px;'>Không có luật kích hoạt.</p>";
            if (detailPanel) {
                detailPanel.innerHTML = `
                  <div class="rule-detail-empty">
                    <i class="fa-solid fa-scale-balanced"></i>
                    <h4>Không có luật suy luận nào được kích hoạt</h4>
                    <p>Hãy bật lại các khái niệm trung gian (CBM Concepts) hoặc tải lên mẫu khác để khám phá các luật suy diễn.</p>
                  </div>
                `;
            }
            return;
        }

        if (activeSelectedRuleIdx < 0 || activeSelectedRuleIdx >= ruleList.length) {
            activeSelectedRuleIdx = 0;
        }

        ruleList.forEach((r, idx) => {
            const box = document.createElement("div");
            const isSelected = idx === activeSelectedRuleIdx;
            box.className = `rule-box rule-selector-item ${isSelected ? "active-rule" : ""}`;
            box.style.cursor = "pointer";
            box.title = `Nhấn để xem chi tiết Luật #${idx + 1} trong Rule Explorer`;

            const confVal = r.confidence <= 1.0 ? r.confidence * 100 : r.confidence;
            const confPct = `${Number(confVal || 0).toFixed(1)}%`;

            box.innerHTML = `
        <div class="rule-selector-top">
          <span class="rule-selector-code">[LUẬT #${idx + 1}] ➜ ${escapeContrastHtml(r.consequent)}</span>
          <div class="rule-metrics-badges">
            <span class="rule-metric-badge">Tin cậy: ${confPct}</span>
            <span class="rule-metric-badge">Mẫu: ${r.support}</span>
          </div>
        </div>
        <div class="rule-selector-snippet">IF ${escapeContrastHtml(r.rule_text)} THEN ${escapeContrastHtml(r.consequent)}</div>
      `;

            box.addEventListener("click", () => {
                activeSelectedRuleIdx = idx;
                container
                    .querySelectorAll(".rule-selector-item")
                    .forEach((el, i) => el.classList.toggle("active-rule", i === idx));
                renderRuleDetailPanel(ruleList, idx);
            });

            container.appendChild(box);
        });

        renderRuleDetailPanel(ruleList, activeSelectedRuleIdx);
    }

    function renderRuleDetailPanel(rules, activeIdx) {
        const detailPanel = document.getElementById("rule-explorer-detail");
        if (!detailPanel || !Array.isArray(rules) || !rules[activeIdx]) return;

        const r = rules[activeIdx];
        const confVal = r.confidence <= 1.0 ? r.confidence * 100 : r.confidence;
        const confPct = `${Number(confVal || 0).toFixed(1)}%`;
        const scorePct =
            r.score !== undefined
                ? `${(Number(r.score) * (r.score <= 1.0 ? 100 : 1)).toFixed(1)}%`
                : confPct;
        const gColor = GENRE_COLORS[r.consequent] || "var(--accent-vermilion)";

        // Tách các mệnh đề điều kiện từ clauses hoặc rule_text
        const clausesList =
            Array.isArray(r.clauses) && r.clauses.length > 0
                ? r.clauses
                : String(r.rule_text || "")
                      .split(/\s+AND\s+/i)
                      .filter(Boolean);

        const clausesHtml = clausesList
            .map((cl, i) => {
                if (typeof cl === "object" && cl !== null) {
                    const cId = escapeContrastHtml(cl.concept_id || cl.feature || `Mệnh đề ${i + 1}`);
                    const cDesc = escapeContrastHtml(cl.description || cl.sentence || cl.value || "");
                    return `
                      <div class="clause-chip-item">
                        <span class="clause-chip-id">${cId}</span>
                        <span class="clause-chip-text">${cDesc}</span>
                      </div>
                    `;
                }
                return `
                  <div class="clause-chip-item">
                    <span class="clause-chip-id">ĐIỀU KIỆN #${i + 1}</span>
                    <span class="clause-chip-text">${escapeContrastHtml(String(cl))}</span>
                  </div>
                `;
            })
            .join("");

        detailPanel.innerHTML = `
          <div class="rule-detail-card">
            <div class="rule-detail-header">
              <div class="rule-detail-title-group">
                <span class="rule-detail-eyebrow">LUẬT SUY LUẬN TƯỢNG TRƯNG #${activeIdx + 1} / ${rules.length}</span>
                <div class="rule-detail-consequent">
                  Kết luận suy diễn:
                  <span class="consequent-genre-tag" style="background:${gColor}15; color:${gColor}; border:1px solid ${gColor}45;">
                    <i class="fa-solid fa-crown" style="font-size:11px;"></i> ${escapeContrastHtml(r.consequent)}
                  </span>
                </div>
              </div>
              <button type="button" class="btn-locate-rule-graph" id="btn-locate-active-rule">
                <i class="fa-solid fa-diagram-project"></i> Định vị trên Đồ thị Concept–Rule
              </button>
            </div>

            <div class="rule-detail-metrics-row">
              <div class="rule-detail-metric-box">
                <span class="rd-metric-label">Độ tin cậy (Confidence)</span>
                <span class="rd-metric-val" style="color: var(--accent-vermilion);">${confPct}</span>
              </div>
              <div class="rule-detail-metric-box">
                <span class="rd-metric-label">Số mẫu huấn luyện (Support)</span>
                <span class="rd-metric-val">${escapeContrastHtml(r.support ?? "--")} mẫu</span>
              </div>
              <div class="rule-detail-metric-box">
                <span class="rd-metric-label">Cường độ kích hoạt</span>
                <span class="rd-metric-val" style="color: var(--accent-jade);">${scorePct}</span>
              </div>
            </div>

            <div>
              <div class="rule-detail-section-label">Cấu trúc mệnh đề hình thức (IF – THEN)</div>
              <div class="rule-detail-code-block">
                <span class="kw-if">IF</span> ${escapeContrastHtml(r.rule_text)} <span class="kw-then">THEN</span> <strong>${escapeContrastHtml(r.consequent)}</strong>
              </div>
            </div>

            <div>
              <div class="rule-detail-section-label">Diễn giải ngữ nghĩa chuyên gia</div>
              <div class="rule-detail-vi-box">
                ${escapeContrastHtml(r.summary_text || "Luật suy diễn kết hợp các khái niệm trung gian đa phương thức để củng cố xác suất nhận diện thể loại di sản.")}
              </div>
            </div>

            <div class="rule-clauses-accordion">
              <button type="button" class="rule-clauses-acc-header" id="btn-toggle-rule-clauses">
                <span><i class="fa-solid fa-list-check" style="color: var(--accent-indigo); margin-right: 6px;"></i> Chi tiết từng mệnh đề điều kiện (${clausesList.length} mệnh đề)</span>
                <i class="fa-solid fa-chevron-down" id="icon-toggle-rule-clauses"></i>
              </button>
              <div class="rule-clauses-acc-body" id="body-rule-clauses">
                ${clausesHtml}
              </div>
            </div>
          </div>
        `;

        const btnAcc = document.getElementById("btn-toggle-rule-clauses");
        const bodyAcc = document.getElementById("body-rule-clauses");
        if (btnAcc && bodyAcc) {
            btnAcc.addEventListener("click", () => {
                bodyAcc.classList.toggle("collapsed");
                const icon = document.getElementById("icon-toggle-rule-clauses");
                if (icon) {
                    icon.className = bodyAcc.classList.contains("collapsed")
                        ? "fa-solid fa-chevron-right"
                        : "fa-solid fa-chevron-down";
                }
            });
        }

        const btnLocate = document.getElementById("btn-locate-active-rule");
        if (btnLocate) {
            btnLocate.addEventListener("click", () => {
                const ruleNodeId = `node_rule_${activeIdx}`;
                activateWorkspaceSection("sec-concept-graph", true);
                if (
                    evidenceMapState.nodeMap &&
                    evidenceMapState.nodeMap.has(ruleNodeId)
                ) {
                    selectEvidenceNode(ruleNodeId);
                    highlightRuleEvidence(ruleNodeId);
                }
            });
        }
    }

    // =========================================================================
    // 9. TIMELINE PHÂN ĐOẠN (CHUNKS 3-5s), GỘP ĐOẠN THÔNG MINH & BIỂU ĐỒ DIỄN BIẾN
    // =========================================================================

    function renderTimelineAndSegments(chunks, mode, mergedSegments) {
        currentRawChunks = chunks || [];
        currentMergedSegments =
            mergedSegments && mergedSegments.length > 0
                ? mergedSegments
                : currentRawChunks;

        const modeBar = document.getElementById("timeline-mode-bar");
        if (modeBar) {
            modeBar.style.display =
                currentMultimodalTab === "visual" && currentRawChunks.length > 0
                    ? "inline-flex"
                    : "none";
        }

        // Thiết lập sự kiện nút chuyển đổi chế độ xem: Gộp vùng tương đồng vs Từng chunk 4s
        const btnMerged = document.getElementById("btn-mode-merged");
        const btnRaw = document.getElementById("btn-mode-raw");

        if (btnMerged) {
            btnMerged.onclick = () => {
                currentTimelineMode = "merged";
                btnMerged.classList.add("active");
                if (btnRaw) btnRaw.classList.remove("active");
                renderSegmentCardsList();
            };
        }

        if (btnRaw) {
            btnRaw.onclick = () => {
                currentTimelineMode = "raw";
                btnRaw.classList.add("active");
                if (btnMerged) btnMerged.classList.remove("active");
                renderSegmentCardsList();
            };
        }

        // Nút xem lại lời lý giải toàn cảnh
        const btnReturnOverall = document.getElementById(
            "btn-return-overall-xai",
        );
        if (btnReturnOverall) {
            btnReturnOverall.onclick = () => {
                restoreOverallExplanation();
            };
        }

        // 1. Render danh sách thẻ phân đoạn (filmstrip)
        renderSegmentCardsList();

        // 2. Render lưới phổ Mel-Spectrogram từ các chunk gốc
        const specGrid = document.getElementById("spectrogram-grid-container");
        if (specGrid) {
            specGrid.innerHTML = "";
            currentRawChunks.forEach((chk) => {
                if (chk.spectrogram_img) {
                    const specCard = document.createElement("div");
                    specCard.className = "spec-chunk-card";
                    specCard.innerHTML = `
            <img src="${chk.spectrogram_img}" alt="Spectrogram ${chk.timestamp}" />
            <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 11px;">
              <span>${chk.timestamp}</span>
              <span style="color: var(--accent-vermilion);">${chk.prediction}</span>
            </div>
          `;
                    specGrid.appendChild(specCard);
                }
            });
        }

        // 3. Render 7 Trục MPEG-7 từ phân đoạn đầu tiên
        if (currentRawChunks[0] && currentRawChunks[0].radar) {
            renderMpeg7RadarBars(currentRawChunks[0]);
        }

        // 4. Thiết lập sự kiện chuyển 4 subtab
        setupTimelineSubtabs();
    }

    function renderSegmentCardsList() {
        const strip = document.getElementById("timeline-scroll-strip");
        const countBadge = document.getElementById("segment-count-badge");
        if (!strip) return;

        strip.innerHTML = "";
        const isMerged =
            currentTimelineMode === "merged" &&
            currentMergedSegments.length > 0;
        const items = isMerged ? currentMergedSegments : currentRawChunks;

        if (countBadge) {
            if (isMerged) {
                countBadge.innerHTML = `<i class="fa-solid fa-layer-group"></i> ${items.length} vùng gộp (${currentRawChunks.length} chunk 4s)`;
            } else {
                countBadge.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> ${items.length} chunk (4s)`;
            }
        }
        const mmVisualPill = document.getElementById("mm-pill-visual-count");
        if (mmVisualPill) {
            mmVisualPill.innerText = `${items ? items.length : 0} đoạn`;
        }

        if (!items || items.length === 0) {
            strip.innerHTML =
                "<p style='color: var(--text-secondary); font-size: 12px; grid-column: 1 / -1; padding: 12px 0;'>Không có dữ liệu phân đoạn.</p>";
            return;
        }

        items.forEach((item, idx) => {
            const cardId = (isMerged ? "merged-seg-" : "raw-chunk-") + idx;
            const isActive = activeSelectedSegmentId === cardId;

            const seg = document.createElement("div");
            seg.className = `segment-item ${isActive ? "active" : ""}`;
            seg.id = cardId;

            const thumbImgSrc = item.keyframe_img || item.spectrogram_img || "";
            const mergedTag = item.is_merged
                ? `<span class="merged-count-badge"><i class="fa-solid fa-layer-group"></i> Gộp ${item.chunk_count} chunk</span>`
                : "";

            // Hiển thị kết quả theo từng phương thức trên thẻ (Requirement 2)
            let modChipsHtml = "";
            const hasAud = item.modalities && item.modalities.audio;
            const hasVis =
                item.modalities &&
                item.modalities.visual &&
                item.modalities.visual.prediction;

            if (hasAud) {
                modChipsHtml += `<span class="mod-pill mod-audio" title="Âm thanh: ${item.modalities.audio.prediction} (${item.modalities.audio.confidence}%)"><i class="fa-solid fa-volume-high"></i> ${item.modalities.audio.prediction}</span>`;
            }
            if (hasVis) {
                modChipsHtml += `<span class="mod-pill mod-visual" title="Thị giác: ${item.modalities.visual.prediction} (${item.modalities.visual.confidence}%)"><i class="fa-solid fa-eye"></i> ${item.modalities.visual.prediction}</span>`;
            }
            modChipsHtml += `<span class="mod-pill mod-fused" title="Hợp nhất: ${item.prediction} (${item.confidence}%)"><i class="fa-solid fa-scale-balanced"></i> ${item.prediction}</span>`;

            // Cues phụ
            const cuesText =
                item.visual_cues && item.visual_cues.length
                    ? item.visual_cues.join(", ")
                    : item.crisp_axes
                      ? `Cao độ: ${item.crisp_axes.Pitch || "Vừa"} • Năng lượng: ${item.crisp_axes.Energy || "Vừa"}`
                      : "";

            seg.innerHTML = `
        <div class="segment-thumb">
          ${thumbImgSrc ? `<img src="${thumbImgSrc}" alt="Segment ${idx}" />` : `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-tertiary);"><i class="fa-solid fa-music"></i></div>`}
          <span class="segment-time-badge">${item.timestamp}</span>
          ${mergedTag}
        </div>
        <div class="segment-meta">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="segment-genre" style="color: ${item.genre_color || "var(--accent-vermilion)"}; font-weight: 700;">${item.prediction}</span>
            <span class="segment-score">${item.confidence}% tin cậy</span>
          </div>
          <div class="segment-modality-tags">${modChipsHtml}</div>
          <span class="segment-tag" title="${cuesText}">${cuesText}</span>
        </div>
      `;

            // Bấm vào thẻ: Làm nổi bật viền, tua video và cập nhật diễn giải (Requirement 1)
            seg.addEventListener("click", () => {
                activeSelectedSegmentId = cardId;
                document
                    .querySelectorAll(".segment-item")
                    .forEach((s) => s.classList.remove("active"));
                seg.classList.add("active");

                seekToTimestamp(item.start_time);
                highlightChunkConcepts(item);
                renderMpeg7RadarBars(item);
                renderSegmentExplanation(item);
            });

            strip.appendChild(seg);
        });
    }

    // Cập nhật phần diễn giải theo phân đoạn tương ứng (Requirement 1 & 2)
    function renderSegmentExplanation(item) {
        const banner = document.getElementById("xai-segment-banner");
        const timeEl = document.getElementById("xai-segment-time");
        const genreEl = document.getElementById("xai-segment-genre-badge");
        const headlineEl = document.getElementById("xai-headline");
        const bodyEl = document.getElementById("xai-body-text");

        if (banner) banner.style.display = "flex";

        if (timeEl) {
            timeEl.innerText = item.is_merged
                ? `Vùng gộp: ${item.timestamp} (${item.chunk_count} chunk)`
                : `Phân đoạn: ${item.timestamp}`;
        }

        if (genreEl) {
            genreEl.innerText = `${item.prediction} • ${item.confidence}%`;
            genreEl.style.color = item.genre_color || "var(--accent-vermilion)";
        }

        if (item.explanation) {
            if (headlineEl) {
                headlineEl.innerText =
                    item.explanation.headline ||
                    `Diễn giải phân đoạn ${item.timestamp}`;
            }
            if (item.explanation.narrative) {
                let formatted = item.explanation.narrative
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
                    .replace(/\n/g, "<br>");
                if (bodyEl) bodyEl.innerHTML = formatted;
                const previewEl = document.getElementById("overview-insight-preview");
                if (previewEl) {
                    previewEl.innerText = item.explanation.narrative
                        .replace(/\*\*([^*]+)\*\*/g, "$1")
                        .replace(/\n+/g, " ")
                        .trim();
                }
            }
        }

        // Hiển thị kết quả theo từng phương thức của phân đoạn này (Requirement 2)
        updateModalitiesGrid(
            item.modalities,
            item.prediction,
            item.confidence,
            true,
        );
    }

    // Khôi phục lại lời lý giải toàn cảnh của cả video
    function restoreOverallExplanation() {
        activeSelectedSegmentId = null;
        document
            .querySelectorAll(".segment-item")
            .forEach((s) => s.classList.remove("active"));

        const banner = document.getElementById("xai-segment-banner");
        if (banner) banner.style.display = "none";

        if (currentOverallExplanation) {
            renderXaiExplanation(currentOverallExplanation);
        }

        if (currentOverallModalities && currentAnalysis) {
            updateModalitiesGrid(
                currentOverallModalities,
                currentAnalysis.prediction,
                currentAnalysis.confidence,
                false,
            );
        }

        if (currentRawChunks && currentRawChunks[0]) {
            renderMpeg7RadarBars(currentRawChunks[0]);
        }

        document.querySelectorAll(".concept-pill").forEach((p) => {
            p.classList.remove("chunk-active");
        });
    }

    // Hiển thị kết quả theo từng phương thức (Requirement 2)
    function updateModalitiesGrid(
        modalities,
        fallbackGenre,
        fallbackConfidence,
        isSegmentView,
    ) {
        const modsGrid = document.getElementById("xai-modalities-grid");
        const modAudio = document.getElementById("mod-card-audio-content");
        const visualWrap = document.getElementById("mod-card-visual-wrap");
        const modVisual = document.getElementById("mod-card-visual-content");
        const modFused = document.getElementById("mod-card-fused-content");

        if (!modsGrid) return;
        modsGrid.style.display = "grid";

        // 1. Nhánh Âm thanh
        if (modAudio) {
            if (modalities && modalities.audio && modalities.audio.prediction) {
                const aNote =
                    modalities.audio.notes ||
                    "Mô hình CNN (EXP-13) + Rừng ngẫu nhiên liên tục";
                modAudio.innerHTML = `<strong>${modalities.audio.prediction}</strong> <span style="color: var(--accent-vermilion); font-family: var(--font-mono); font-weight: 700;">${modalities.audio.confidence}%</span><br><span style="font-size: 11px; color: var(--text-secondary);">${aNote}</span>`;
            } else {
                modAudio.innerHTML = `<strong>${fallbackGenre}</strong> <span style="color: var(--accent-vermilion); font-family: var(--font-mono); font-weight: 700;">${fallbackConfidence}%</span><br><span style="font-size: 11px; color: var(--text-secondary);">Mô hình âm thanh</span>`;
            }
        }

        // 2. Nhánh Thị giác: Luôn giữ hiển thị trong ngăn điều hướng 3 lựa chọn
        if (visualWrap && modVisual) {
            visualWrap.style.display = "block";
            if (
                modalities &&
                modalities.visual &&
                modalities.visual.prediction
            ) {
                const vCues =
                    modalities.visual.cues && modalities.visual.cues.length
                        ? modalities.visual.cues.join(", ")
                        : modalities.visual.notes ||
                          "Nhận diện nhạc cụ, trang phục & đạo cụ";
                modVisual.innerHTML = `<strong>${modalities.visual.prediction}</strong> <span style="color: var(--accent-amber); font-family: var(--font-mono); font-weight: 700;">${modalities.visual.confidence}%</span><br><span style="font-size: 11px; color: var(--text-secondary);">${vCues}</span>`;
            } else {
                modVisual.innerHTML = `<span style="font-size: 11px; color: var(--text-secondary);">Dải phân đoạn thời gian &amp; Keyframes thị giác</span>`;
            }
        }

        // 3. Kết quả Hợp nhất
        if (modFused) {
            const fPred =
                modalities && modalities.fused && modalities.fused.prediction
                    ? modalities.fused.prediction
                    : fallbackGenre;
            const fConf =
                modalities && modalities.fused && modalities.fused.confidence
                    ? modalities.fused.confidence
                    : fallbackConfidence;
            const fMethod =
                modalities && modalities.fused && modalities.fused.method
                    ? modalities.fused.method
                    : "Late Fusion Bayesian + Suy luận Mệnh đề Luật Mờ";
            modFused.innerHTML = `<strong>${fPred}</strong> <span style="color: var(--accent-jade); font-family: var(--font-mono); font-weight: 700;">${fConf}%</span><br><span style="font-size: 11px; color: var(--text-secondary);">${fMethod}</span>`;
        }
    }

    // Tua media tới thời điểm giây start_time
    function seekToTimestamp(seconds) {
        const sec = Math.max(0, Number(seconds) || 0);

        if (activeMediaType === "video" && videoPlayer && videoPlayer.src) {
            if (currentActiveTab !== "media") switchTab("media");
            if (videoContainer) {
                const rect = videoContainer.getBoundingClientRect();
                if (rect.bottom < 50 || rect.top > window.innerHeight - 50) {
                    videoContainer.scrollIntoView({
                        behavior: "auto",
                        block: "nearest",
                    });
                }
            }
            if ("fastSeek" in videoPlayer) {
                try {
                    videoPlayer.fastSeek(sec);
                } catch (e) {
                    videoPlayer.currentTime = sec;
                }
            } else {
                videoPlayer.currentTime = sec;
            }
            videoPlayer.playbackRate = 1.0;
            const playPromise = videoPlayer.play();
            if (playPromise !== undefined) {
                playPromise.catch((e) =>
                    console.warn("Video play interrupted:", e),
                );
            }
        } else if (
            activeMediaType === "audio" &&
            audioPlayer &&
            audioPlayer.src
        ) {
            if (currentActiveTab !== "media" && currentActiveTab !== "mic")
                switchTab("media");
            if (audioContainer) {
                const rect = audioContainer.getBoundingClientRect();
                if (rect.bottom < 50 || rect.top > window.innerHeight - 50) {
                    audioContainer.scrollIntoView({
                        behavior: "auto",
                        block: "nearest",
                    });
                }
            }
            if ("fastSeek" in audioPlayer) {
                try {
                    audioPlayer.fastSeek(sec);
                } catch (e) {
                    audioPlayer.currentTime = sec;
                }
            } else {
                audioPlayer.currentTime = sec;
            }
            audioPlayer.playbackRate = 1.0;
            const playPromise = audioPlayer.play();
            if (playPromise !== undefined) {
                playPromise.catch((e) =>
                    console.warn("Audio play interrupted:", e),
                );
            }
        }
    }

    function highlightChunkConcepts(chunk) {
        const activeIds = [
            ...(chunk.active_audio_concepts || []),
            ...(chunk.active_visual_concepts || []),
        ];
        document.querySelectorAll(".concept-pill").forEach((p) => {
            const cid = p.getAttribute("data-concept-id");
            if (activeIds.includes(cid)) {
                p.classList.add("chunk-active");
            } else {
                p.classList.remove("chunk-active");
            }
        });
    }

    function renderMpeg7RadarBars(chunk) {
        const radarList = document.getElementById("radar-bars-container");
        if (!radarList) return;

        radarList.innerHTML = "";
        const axesOrder = [
            "Energy",
            "Harmony",
            "Pitch",
            "Spectral",
            "Tempo",
            "Timbre",
            "Vibrato",
        ];

        axesOrder.forEach((ax) => {
            const val =
                chunk.radar && chunk.radar[ax] !== undefined
                    ? chunk.radar[ax]
                    : 0.5;
            const crispVal =
                chunk.crisp_axes && chunk.crisp_axes[ax]
                    ? chunk.crisp_axes[ax]
                    : "Vừa";

            const row = document.createElement("div");
            row.className = "axis-bar-row";
            row.innerHTML = `
        <span class="axis-name">${ax}</span>
        <div class="axis-progress-track">
          <div class="axis-progress-fill" style="width: ${val * 100}%;"></div>
        </div>
        <span class="axis-crisp-badge" title="${crispVal}">${crispVal}</span>
        <span class="axis-val-num">${(val * 100).toFixed(0)}%</span>
      `;
            radarList.appendChild(row);
        });
    }

    // Chọn phân đoạn tương ứng với mốc thời gian và làm nổi bật
    function selectSegmentAtTime(targetSec) {
        const isMerged =
            currentTimelineMode === "merged" &&
            currentMergedSegments.length > 0;
        const items = isMerged ? currentMergedSegments : currentRawChunks;
        if (!items || items.length === 0) return;

        let matchedIdx = items.findIndex(
            (item) =>
                targetSec >= item.start_time && targetSec <= item.end_time,
        );
        if (matchedIdx === -1) {
            let minDiff = Infinity;
            items.forEach((item, idx) => {
                const mid = (item.start_time + item.end_time) / 2;
                const diff = Math.abs(mid - targetSec);
                if (diff < minDiff) {
                    minDiff = diff;
                    matchedIdx = idx;
                }
            });
        }

        if (matchedIdx !== -1) {
            const cardId =
                (isMerged ? "merged-seg-" : "raw-chunk-") + matchedIdx;
            activeSelectedSegmentId = cardId;
            document
                .querySelectorAll(".segment-item")
                .forEach((s) => s.classList.remove("active"));
            const card = document.getElementById(cardId);
            if (card) {
                card.classList.add("active");
                card.scrollIntoView({
                    behavior: "smooth",
                    block: "nearest",
                    inline: "nearest",
                });
            }
            const item = items[matchedIdx];
            if (item) {
                highlightChunkConcepts(item);
                renderMpeg7RadarBars(item);
                renderSegmentExplanation(item);
            }
        }
    }

    // Dải Ribbon Màu Thể loại Liên tục theo Thời gian
    function renderGenreTimelineRibbon(chunks, totalDuration) {
        const ribbonWrap = document.getElementById(
            "genre-timeline-ribbon-wrap",
        );
        const ribbon = document.getElementById("genre-timeline-ribbon");
        const ribbonPlayhead = document.getElementById("ribbon-playhead");
        if (!ribbonWrap || !ribbon || !chunks || chunks.length === 0) return;

        ribbonWrap.style.display = "block";
        ribbon.innerHTML = "";

        const activePlayer =
            activeMediaType === "video" && videoPlayer && videoPlayer.src
                ? videoPlayer
                : audioPlayer;
        const totalDur =
            activePlayer && activePlayer.duration > 0
                ? activePlayer.duration
                : totalDuration || chunks[chunks.length - 1].end_time || 1;

        chunks.forEach((chk, idx) => {
            const segDur = chk.duration || chk.end_time - chk.start_time || 4;
            const widthPct = Math.max(0.5, (segDur / totalDur) * 100);
            const color = GENRE_COLORS[chk.prediction] || "#C84B31";

            const rSeg = document.createElement("div");
            rSeg.className = "ribbon-segment";
            rSeg.style.flex = `${segDur} 0 0%`;
            rSeg.style.width = `${widthPct}%`;
            rSeg.style.background = color;
            rSeg.title = `${chk.timestamp}: ${chk.prediction} (${chk.confidence}%)`;

            ribbon.appendChild(rSeg);
        });

        // Bấm chuột vào bất kỳ vị trí nào trên dải Ribbon: Tua và đặt playhead chính xác tại vị trí ấn
        ribbonWrap.onclick = (e) => {
            const rect = ribbonWrap.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const ratio = Math.max(0, Math.min(1, clickX / rect.width));
            const curPlayer =
                activeMediaType === "video" && videoPlayer && videoPlayer.src
                    ? videoPlayer
                    : audioPlayer;
            const realDur =
                curPlayer && curPlayer.duration > 0
                    ? curPlayer.duration
                    : totalDur;
            const targetSec = ratio * realDur;

            if (ribbonPlayhead) {
                ribbonPlayhead.style.left = `${ratio * 100}%`;
            }

            seekToTimestamp(targetSec);
            selectSegmentAtTime(targetSec);
        };
    }

    // =========================================================================
    // 10. BIỂU ĐỒ DIỄN BIẾN KẾT QUẢ THEO THỜI GIAN (TIMELINE SVG STREAM CHART)
    // =========================================================================

    function renderTimelineChart(timelineSeries, totalDuration) {
        const chartView = document.getElementById("timeline-chart-container");
        const legendContainer = document.getElementById(
            "chart-legend-container",
        );
        const svg = document.getElementById("timeline-svg-chart");
        if (!chartView || !svg) return;

        const series =
            timelineSeries && timelineSeries.length > 0
                ? timelineSeries
                : currentRawChunks;
        if (!series || series.length === 0) return;

        const activePlayer =
            activeMediaType === "video" && videoPlayer && videoPlayer.src
                ? videoPlayer
                : audioPlayer;
        const totalDur =
            activePlayer && activePlayer.duration > 0
                ? activePlayer.duration
                : totalDuration ||
                  series[series.length - 1].end_time ||
                  series.length * 4;

        // 1. Tìm các thể loại có xác suất đáng kể trong chuỗi thời gian để tạo Legend
        const activeGenres = new Set();
        series.forEach((pt) => {
            if (pt.prediction) activeGenres.add(pt.prediction);
            if (pt.probs) {
                Object.entries(pt.probs).forEach(([g, rawP]) => {
                    const pVal = rawP <= 1.0 ? rawP * 100 : rawP;
                    if (pVal >= 8.0) activeGenres.add(g);
                });
            }
        });

        if (legendContainer) {
            legendContainer.innerHTML = "";
            Array.from(activeGenres).forEach((g) => {
                const item = document.createElement("span");
                item.className = "chart-legend-item";
                const col = GENRE_COLORS[g] || "#C84B31";
                item.style.color = col;
                item.style.borderColor = `${col}40`;
                item.style.backgroundColor = `${col}14`;
                item.innerHTML = `<span class="chart-legend-dot" style="background: ${col};"></span> ${g}`;
                legendContainer.appendChild(item);
            });
        }

        // 2. Vẽ biểu đồ SVG
        drawTimelineSvgChart(svg, series, Array.from(activeGenres), totalDur);
    }

    function drawTimelineSvgChart(svg, series, activeGenres, totalDuration) {
        const W = 600;
        const H = 200;
        const pad = { top: 22, right: 30, bottom: 35, left: 45 };
        const plotW = W - pad.left - pad.right;
        const plotH = H - pad.top - pad.bottom;

        const dur = Math.max(1, totalDuration);

        let svgInner = "";

        // 1. Trục Y & Lưới phụ ngang (0%, 25%, 50%, 75%, 100%)
        const yLevels = [0, 25, 50, 75, 100];
        yLevels.forEach((pct) => {
            const y = pad.top + plotH - (pct / 100) * plotH;
            const isBase = pct === 0;
            svgInner += `
        <line x1="${pad.left}" y1="${y}" x2="${W - pad.right}" y2="${y}" 
              stroke="${isBase ? "rgba(0,0,0,0.22)" : "rgba(0,0,0,0.07)"}" 
              stroke-width="${isBase ? "1.5" : "1"}" 
              stroke-dasharray="${isBase ? "" : "3,3"}" />
        <text x="${pad.left - 8}" y="${y + 4}" fill="#777" font-size="10" font-family="monospace" text-anchor="end">${pct}%</text>
      `;
        });

        // 2. Trục X & Các mốc thời gian (Ticks dạng số nguyên, đơn vị (s) ở cuối trục)
        const step = dur <= 25 ? 5 : dur <= 60 ? 10 : 20;
        for (let t = 0; t <= dur; t += step) {
            const x = pad.left + (t / dur) * plotW;
            svgInner += `
        <line x1="${x}" y1="${pad.top + plotH}" x2="${x}" y2="${pad.top + plotH + 5}" stroke="rgba(0,0,0,0.25)" stroke-width="1" />
        <text x="${x}" y="${pad.top + plotH + 16}" fill="#777" font-size="10" font-family="monospace" text-anchor="middle">${t}</text>
      `;
        }
        // Ghi nhãn đơn vị (s) duy nhất ở góc cuối phải của trục X
        svgInner += `
      <text x="${pad.left + plotW + 12}" y="${pad.top + plotH + 16}" fill="#666" font-size="10.5" font-family="monospace" font-weight="600" text-anchor="start">(s)</text>
    `;

        // 3. Đường cong xác suất cho từng thể loại
        activeGenres.forEach((genre) => {
            const col = GENRE_COLORS[genre] || "#888";

            // Chuẩn hóa xác suất sang thang 0..100
            const rawPts = series.map((pt) => {
                const t =
                    pt.time !== undefined
                        ? pt.time
                        : (pt.start_time + pt.end_time) / 2;
                let pVal = 0;
                if (pt.probs && pt.probs[genre] !== undefined) {
                    pVal = pt.probs[genre];
                } else if (pt.prediction === genre) {
                    pVal = pt.confidence || 60;
                }
                if (pVal <= 1.0) pVal = pVal * 100;
                return {
                    time: t,
                    prob: Math.max(0, Math.min(100, pVal)),
                    item: pt,
                };
            });

            if (rawPts.length === 0) return;

            // Thêm điểm neo ở t = 0 và t = dur để đường cong bao phủ toàn bộ thời lượng
            const pts = [];
            pts.push({
                x: pad.left,
                y: pad.top + plotH - (rawPts[0].prob / 100) * plotH,
                prob: rawPts[0].prob,
                time: 0,
            });

            rawPts.forEach((p) => {
                pts.push({
                    x:
                        pad.left +
                        (Math.max(0, Math.min(dur, p.time)) / dur) * plotW,
                    y: pad.top + plotH - (p.prob / 100) * plotH,
                    prob: p.prob,
                    time: p.time,
                });
            });

            const lastRaw = rawPts[rawPts.length - 1];
            pts.push({
                x: pad.left + plotW,
                y: pad.top + plotH - (lastRaw.prob / 100) * plotH,
                prob: lastRaw.prob,
                time: dur,
            });

            // Tạo đường cong Bézier trơn qua các điểm
            let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
            for (let i = 1; i < pts.length; i++) {
                const prev = pts[i - 1];
                const curr = pts[i];
                const cpX1 = prev.x + (curr.x - prev.x) * 0.5;
                const cpY1 = prev.y;
                const cpX2 = prev.x + (curr.x - prev.x) * 0.5;
                const cpY2 = curr.y;
                d += ` C ${cpX1.toFixed(1)} ${cpY1.toFixed(1)}, ${cpX2.toFixed(1)} ${cpY2.toFixed(1)}, ${curr.x.toFixed(1)} ${curr.y.toFixed(1)}`;
            }

            svgInner += `
        <path d="${d}" fill="none" stroke="${col}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" opacity="0.88" />
      `;
        });

        // 4. Các điểm dữ liệu mốc thời gian (Points) của thể loại Winner
        series.forEach((pt, idx) => {
            const t =
                pt.time !== undefined
                    ? pt.time
                    : (pt.start_time + pt.end_time) / 2;
            const x = pad.left + (Math.max(0, Math.min(dur, t)) / dur) * plotW;
            const topGenre = pt.prediction;
            let topProb =
                pt.probs && pt.probs[topGenre] !== undefined
                    ? pt.probs[topGenre]
                    : pt.confidence || 60;
            if (topProb <= 1.0) topProb = topProb * 100;
            const y = pad.top + plotH - (topProb / 100) * plotH;
            const col = GENRE_COLORS[topGenre] || "#C84B31";

            svgInner += `
        <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5.5" 
                fill="${col}" stroke="#ffffff" stroke-width="2" 
                class="chart-point" data-chunk-idx="${idx}" data-time="${t}" data-genre="${topGenre}" data-prob="${topProb.toFixed(1)}"
                style="cursor: pointer; transition: transform 0.15s ease;" />
      `;
        });

        svg.innerHTML = svgInner;

        // 5. Gắn tương tác Hover & Click
        setupChartInteractivity(svg, series, pad, plotW, dur);
    }

    function setupChartInteractivity(svg, series, pad, plotW, totalDuration) {
        const tooltip = document.getElementById("chart-hover-tooltip");
        const scrubLine = document.getElementById("chart-scrub-line");
        const wrapper = document.getElementById("chart-svg-wrapper");
        if (!wrapper) return;

        // Hover trên từng điểm hình tròn
        const points = svg.querySelectorAll(".chart-point");
        points.forEach((pt) => {
            const idx = parseInt(pt.getAttribute("data-chunk-idx"), 10);
            const chk = series[idx];
            if (!chk) return;

            pt.addEventListener("mouseenter", (e) => {
                pt.setAttribute("r", "8");
                if (tooltip) {
                    const topG = chk.prediction;
                    let conf = chk.confidence;
                    if (conf <= 1.0) conf = conf * 100;
                    const cues =
                        chk.visual_cues && chk.visual_cues.length
                            ? chk.visual_cues.join(", ")
                            : "";

                    // Liệt kê xác suất các lớp
                    let probsListHtml = "";
                    if (chk.probs) {
                        const sorted = Object.entries(chk.probs)
                            .map(([g, p]) => [g, p <= 1.0 ? p * 100 : p])
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 3);
                        probsListHtml = sorted
                            .map(
                                ([g, p]) => `
              <div style="display: flex; justify-content: space-between; gap: 8px; font-size: 10px; margin-top: 2px;">
                <span style="color: ${GENRE_COLORS[g] || "#666"}">${g}:</span>
                <span style="font-weight: 700;">${p.toFixed(1)}%</span>
              </div>
            `,
                            )
                            .join("");
                    }

                    tooltip.innerHTML = `
            <div style="font-weight: 700; color: ${GENRE_COLORS[topG] || "#C84B31"}; margin-bottom: 3px; font-size: 11.5px;">
              ${topG} (${conf.toFixed(1)}%)
            </div>
            <div style="color: var(--text-secondary); font-size: 10px; margin-bottom: 4px;">
              Mốc: ${chk.timestamp}
            </div>
            ${probsListHtml}
            ${cues ? `<div style="color: var(--accent-amber); font-size: 10px; margin-top: 4px; border-top: 1px dashed rgba(0,0,0,0.1); padding-top: 2px;">Vật thể: ${cues}</div>` : ""}
          `;
                    tooltip.style.display = "block";
                    const rect = wrapper.getBoundingClientRect();
                    const xPos = e.clientX - rect.left;
                    tooltip.style.left = `${Math.min(rect.width - 170, Math.max(10, xPos - 80))}px`;
                    tooltip.style.top = "10px";
                }
            });

            pt.addEventListener("mouseleave", () => {
                pt.setAttribute("r", "5.5");
                if (tooltip) tooltip.style.display = "none";
            });

            pt.addEventListener("click", (e) => {
                e.stopPropagation();
                const nodeTime =
                    chk.time !== undefined
                        ? chk.time
                        : (chk.start_time + chk.end_time) / 2;
                seekToTimestamp(nodeTime);
                selectSegmentAtTime(nodeTime);

                // Đồng bộ dải ribbon playhead và chart scrub line tới đúng mốc thời gian của node
                const ribbonPlayhead =
                    document.getElementById("ribbon-playhead");
                const activePlayer =
                    activeMediaType === "video" &&
                    videoPlayer &&
                    videoPlayer.src
                        ? videoPlayer
                        : audioPlayer;
                const dur =
                    activePlayer && activePlayer.duration > 0
                        ? activePlayer.duration
                        : totalDuration;
                if (ribbonPlayhead && dur > 0) {
                    ribbonPlayhead.style.left = `${(nodeTime / dur) * 100}%`;
                }
                if (scrubLine && dur > 0) {
                    const chartPct =
                        ((pad.left + (nodeTime / dur) * plotW) / 600) * 100;
                    scrubLine.style.left = `${chartPct}%`;
                    scrubLine.style.display = "block";
                }
            });
        });

        // Tắt animation line màu đỏ di chuyển theo chuột: line chỉ cập nhật khi video đang phát hoặc người dùng bấm tua
        // (Đã loại bỏ hoàn toàn sự kiện mousemove/mouseleave di chuyển scrubLine)

        // Bấm bất kỳ đâu trên SVG biểu đồ để tua tới đúng thời gian
        svg.addEventListener("click", (e) => {
            const rect = svg.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const ratio = Math.max(0, Math.min(1, clickX / rect.width));
            const viewX = ratio * 600;

            let targetSec = 0;
            if (viewX <= pad.left) {
                targetSec = 0;
            } else if (viewX >= 600 - pad.right) {
                targetSec = totalDuration;
            } else {
                const timeFraction = (viewX - pad.left) / plotW;
                targetSec = Math.max(
                    0,
                    Math.min(totalDuration, timeFraction * totalDuration),
                );
            }

            seekToTimestamp(targetSec);
            selectSegmentAtTime(targetSec);

            const ribbonPlayhead = document.getElementById("ribbon-playhead");
            if (ribbonPlayhead) {
                const activePlayer =
                    activeMediaType === "video" &&
                    videoPlayer &&
                    videoPlayer.src
                        ? videoPlayer
                        : audioPlayer;
                const dur =
                    activePlayer && activePlayer.duration > 0
                        ? activePlayer.duration
                        : totalDuration;
                ribbonPlayhead.style.left = `${(targetSec / dur) * 100}%`;
            }
        });
    }

    // Đồng bộ vị trí phát Video/Audio với thanh Ribbon và đường Scrubber trên biểu đồ
    function setupPlaybackTimeSync(player, totalDuration) {
        if (!player) return;

        const ribbonPlayhead = document.getElementById("ribbon-playhead");
        const chartScrubLine = document.getElementById("chart-scrub-line");

        if (playbackSyncHandler) {
            player.removeEventListener("timeupdate", playbackSyncHandler);
        }

        playbackSyncHandler = () => {
            const dur =
                player.duration && player.duration > 0
                    ? player.duration
                    : totalDuration || 1;
            if (!dur || dur <= 0) return;

            const cur = player.currentTime;
            const pct = Math.min(100, Math.max(0, (cur / dur) * 100));

            if (ribbonPlayhead) {
                ribbonPlayhead.style.left = `${pct}%`;
            }

            if (chartScrubLine) {
                // Tọa độ biểu đồ: pad.left = 45, plotW = 525, tổng W = 600
                const chartPct = ((45 + (pct / 100) * 525) / 600) * 100;
                chartScrubLine.style.left = `${chartPct}%`;
                chartScrubLine.style.display = "block";
            }
        };

        player.addEventListener("timeupdate", playbackSyncHandler);
    }

    const MM_TAB_METAS = {
        visual: {
            icon: "fa-solid fa-film",
            color: "var(--accent-amber)",
            label: "Dải Keyframes",
        },
        audio: {
            icon: "fa-solid fa-wave-square",
            color: "var(--accent-jade)",
            label: "Ảnh Phổ Mel & 7 Trục MPEG-7",
        },
        fused: {
            icon: "fa-solid fa-chart-line",
            color: "var(--accent-vermilion)",
            label: "Biểu đồ Diễn biến & CBM",
        },
    };

    // Hàm chuyển đổi giữa 3 Tab Bằng chứng Đa phương thức: "visual" | "audio" | "fused"
    function switchMultimodalTab(tabName) {
        currentMultimodalTab = tabName || "visual";

        // Mặc định khi bấm tab dải keyframe là hiển thị chế độ Từng chunk
        if (currentMultimodalTab === "visual") {
            currentTimelineMode = "raw";
            renderSegmentCardsList();
        }

        // 1. Cập nhật Trigger Icon & Label của Menu Dropdown
        const meta = MM_TAB_METAS[currentMultimodalTab] || MM_TAB_METAS["visual"];
        const triggerIcon = document.getElementById("subtab-trigger-icon");
        const triggerText = document.getElementById("subtab-trigger-text");
        if (triggerIcon && meta) {
            triggerIcon.innerHTML = `<i class="${meta.icon}" style="color: ${meta.color};"></i>`;
        }
        if (triggerText && meta) {
            triggerText.innerText = meta.label;
        }

        // Cập nhật trạng thái active trên danh sách các mục trong dropdown
        document.querySelectorAll(".custom-dropdown-item").forEach((item) => {
            const isMatch = item.getAttribute("data-mm-tab") === currentMultimodalTab;
            item.classList.toggle("active", isMatch);
        });

        // Hỗ trợ cập nhật các nút tab cũ nếu còn xuất hiện
        document.querySelectorAll(".mm-tab-btn").forEach((btn) => {
            const isMatch = btn.getAttribute("data-mm-tab") === currentMultimodalTab;
            btn.classList.toggle("active", isMatch);
            btn.setAttribute("aria-selected", isMatch ? "true" : "false");
        });

        // 2. Hiển thị đúng pane (#mm-pane-audio, #mm-pane-visual, #mm-pane-fused)
        const paneAudio = document.getElementById("mm-pane-audio");
        const paneVisual = document.getElementById("mm-pane-visual");
        const paneFused = document.getElementById("mm-pane-fused");

        if (paneAudio) {
            paneAudio.style.display = currentMultimodalTab === "audio" ? "block" : "none";
            paneAudio.classList.toggle("active", currentMultimodalTab === "audio");
        }
        if (paneVisual) {
            paneVisual.style.display = currentMultimodalTab === "visual" ? "block" : "none";
            paneVisual.classList.toggle("active", currentMultimodalTab === "visual");
        }
        if (paneFused) {
            paneFused.style.display = currentMultimodalTab === "fused" ? "block" : "none";
            paneFused.classList.toggle("active", currentMultimodalTab === "fused");
        }

        // 3. Ẩn thanh Gộp vùng / Từng chunk theo yêu cầu tinh gọn giao diện
        const modeBar = document.getElementById("timeline-mode-bar");
        if (modeBar) {
            modeBar.style.display = "none";
        }

        // 4. Lọc hiển thị nhóm Khái niệm CBM tương ứng với Tab đang mở
        const audioPillsGroup = document.getElementById("audio-pills-group");
        const visualPillsGroup = document.getElementById("visual-pills-group");
        if (audioPillsGroup) {
            audioPillsGroup.style.display =
                currentMultimodalTab === "visual" ? "none" : "block";
        }
        if (visualPillsGroup) {
            if (currentMultimodalTab === "audio") {
                visualPillsGroup.style.display = "none";
            } else {
                visualPillsGroup.style.display = "block";
            }
        }

        // 5. Nếu mở tab Hợp nhất & Diễn biến thời gian, vẽ lại biểu đồ SVG để đảm bảo kích thước chuẩn
        const chartView = document.getElementById("timeline-chart-container");
        if (currentMultimodalTab === "fused") {
            if (chartView) chartView.style.display = "flex";
            const series =
                currentTimelineSeries && currentTimelineSeries.length > 0
                    ? currentTimelineSeries
                    : currentRawChunks;
            if (series && series.length > 0) {
                renderTimelineChart(
                    series,
                    currentAnalysis ? currentAnalysis.duration_seconds : 0,
                );
            }
        }
    }

    // Tương thích ngược với tên hàm switchTimelineSubtab
    function switchTimelineSubtab(target) {
        currentTimelineSubtab = target;
        if (target === "filmstrip") {
            switchMultimodalTab("visual");
        } else if (target === "chart") {
            switchMultimodalTab("fused");
        } else {
            switchMultimodalTab("audio");
        }
    }

    // Khởi tạo sự kiện cho Menu Dropdown Bằng chứng Đa phương thức
    function setupTimelineSubtabs() {
        const trigger = document.getElementById("custom-subtab-trigger");
        const menu = document.getElementById("custom-subtab-menu");

        if (trigger && menu) {
            trigger.onclick = (e) => {
                e.stopPropagation();
                const isOpen = menu.classList.toggle("show");
                trigger.classList.toggle("open", isOpen);
                trigger.setAttribute("aria-expanded", isOpen ? "true" : "false");
            };

            document.addEventListener("click", (e) => {
                if (!trigger.contains(e.target) && !menu.contains(e.target)) {
                    menu.classList.remove("show");
                    trigger.classList.remove("open");
                    trigger.setAttribute("aria-expanded", "false");
                }
            });

            document.addEventListener("keydown", (e) => {
                if (e.key === "Escape" && menu.classList.contains("show")) {
                    menu.classList.remove("show");
                    trigger.classList.remove("open");
                    trigger.setAttribute("aria-expanded", "false");
                }
            });
        }

        // Gắn sự kiện chuyển tab cho từng item trong menu dropdown
        document.querySelectorAll(".custom-dropdown-item").forEach((item) => {
            item.onclick = (e) => {
                e.stopPropagation();
                const tabName = item.getAttribute("data-mm-tab");
                if (tabName) {
                    switchMultimodalTab(tabName);
                }
                if (menu && trigger) {
                    menu.classList.remove("show");
                    trigger.classList.remove("open");
                    trigger.setAttribute("aria-expanded", "false");
                }
            };
        });

        document.querySelectorAll(".mm-tab-btn").forEach((btn) => {
            btn.onclick = () => {
                const tabName = btn.getAttribute("data-mm-tab");
                switchMultimodalTab(tabName);
            };
        });
    }

    // =========================================================================
    // 10. TIỆN ÍCH GIAO DIỆN: LOADING OVERLAY & TOAST NOTIFICATION
    // =========================================================================
    function showLoading(title, subtitle) {
        if (audioWave && typeof audioWave.setProcessingState === "function") {
            audioWave.setProcessingState(true);
        }
        const sphereBackdrop = document.getElementById("ambient-sphere-backdrop");
        if (sphereBackdrop) {
            sphereBackdrop.classList.add("loading-centered");
        }
        if (loadingOverlay) {
            const h3 = loadingOverlay.querySelector("h3");
            const p = loadingOverlay.querySelector("p");
            if (h3) h3.innerText = title;
            if (p) p.innerText = subtitle;
            loadingOverlay.style.display = "flex";
        }
    }

    function hideLoading() {
        if (audioWave && typeof audioWave.setProcessingState === "function") {
            audioWave.setProcessingState(false);
        }
        const sphereBackdrop = document.getElementById("ambient-sphere-backdrop");
        if (sphereBackdrop) {
            sphereBackdrop.classList.remove("loading-centered");
        }
        if (loadingOverlay) {
            loadingOverlay.style.display = "none";
        }
    }

    function showToast(message) {
        if (!toastBox) return;
        toastBox.innerText = message;
        toastBox.style.display = "flex";
        setTimeout(() => {
            toastBox.style.display = "none";
        }, 4500);
    }

    // =========================================================================
    // 11. GIẢI THÍCH ĐỐI CHỨNG (CONTRASTIVE EXPLANATION) & THỬ NGHIỆM WHAT-IF
    // =========================================================================
    let whatIfState = {
        top1Genre: "",
        top2Genre: "",
        top1BaseProb: 50,
        top2BaseProb: 20,
    };

    function escapeContrastHtml(str) {
        if (str === null || str === undefined) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function parseTimestampToSeconds(tsStr) {
        if (!tsStr || typeof tsStr !== "string") return 0;
        const firstPart = tsStr.split("-")[0].trim();
        const parts = firstPart.split(":").map((n) => parseFloat(n));
        if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
            return parts[0] * 60 + parts[1];
        }
        if (parts.length === 3 && !isNaN(parts[0]) && !isNaN(parts[1]) && !isNaN(parts[2])) {
            return parts[0] * 3600 + parts[1] * 60 + parts[2];
        }
        return 0;
    }

    function renderTimestampChipsHtml(timestamps) {
        if (!Array.isArray(timestamps) || timestamps.length === 0) return "";
        const chips = timestamps
            .slice(0, 4)
            .map((ts) => {
                const sec = parseTimestampToSeconds(ts);
                return `<button type="button" class="contrast-ts-chip" data-seek-sec="${sec}" title="Nhấn để tua trình phát tới phân đoạn ${escapeContrastHtml(ts)}"><i class="fa-regular fa-clock"></i> ${escapeContrastHtml(ts)}</button>`;
            })
            .join("");
        return `<div class="contrast-ts-list">${chips}</div>`;
    }

    function getStatusBadgeConfig(status, labelOverride) {
        switch (status) {
            case "detected":
                return {
                    cls: "ev-badge-detected",
                    icon: "fa-circle-check",
                    text: labelOverride || "Có bằng chứng từ đầu vào",
                };
            case "shared_signal":
                return {
                    cls: "ev-badge-shared",
                    icon: "fa-arrows-left-right",
                    text: labelOverride || "Tín hiệu đầu vào có giao thoa một phần",
                };
            case "unverified_modality":
                return {
                    cls: "ev-badge-unverified",
                    icon: "fa-eye-slash",
                    text: labelOverride || "Chưa có dữ liệu hình ảnh (Chỉ âm thanh)",
                };
            case "unverified_context":
                return {
                    cls: "ev-badge-unverified",
                    icon: "fa-circle-question",
                    text: labelOverride || "Không thể kiểm chứng tự động từ tín hiệu",
                };
            case "not_detected":
            default:
                return {
                    cls: "ev-badge-not-detected",
                    icon: "fa-circle-minus",
                    text: labelOverride || "Không phát hiện bằng chứng từ đầu vào",
                };
        }
    }

    function preserveChunkEvidenceInContrastive(newContrastive) {
        if (!newContrastive || !newContrastive.genre_profiles) return;
        const hasAnyTimestamps = Object.values(newContrastive.genre_profiles).some((gp) => {
            const critMap = gp?.criteria || gp || {};
            return Object.values(critMap).some(
                (c) => Array.isArray(c?.timestamps) && c.timestamps.length > 0,
            );
        });
        if (hasAnyTimestamps) {
            initialContrastiveProfiles = JSON.parse(
                JSON.stringify(newContrastive.genre_profiles),
            );
            return;
        }
        // Nếu kết quả trả về từ /api/intervene không đính kèm chunks, kế thừa timestamps & số đo âm học từ lần phân tích gốc
        if (!initialContrastiveProfiles) return;
        Object.keys(newContrastive.genre_profiles).forEach((g) => {
            const initProf =
                initialContrastiveProfiles[g]?.criteria ||
                initialContrastiveProfiles[g];
            const curProf =
                newContrastive.genre_profiles[g]?.criteria ||
                newContrastive.genre_profiles[g];
            if (!initProf || !curProf) return;
            Object.keys(curProf).forEach((dimKey) => {
                const initCell = initProf[dimKey];
                const curCell = curProf[dimKey];
                if (!initCell || !curCell) return;
                if (
                    (!curCell.timestamps || curCell.timestamps.length === 0) &&
                    Array.isArray(initCell.timestamps) &&
                    initCell.timestamps.length > 0 &&
                    (curCell.evidence_status === "detected" ||
                        curCell.evidence_status === "shared_signal")
                ) {
                    curCell.timestamps = initCell.timestamps.slice();
                }
                if (
                    curCell.evidence_status === "not_detected" &&
                    initCell.evidence_status === "shared_signal"
                ) {
                    curCell.evidence_status = "shared_signal";
                    curCell.evidence_label = initCell.evidence_label;
                    curCell.input_evidence = initCell.input_evidence;
                    curCell.support_metric = initCell.support_metric;
                    curCell.timestamps = (initCell.timestamps || []).slice();
                }
            });
        });

        // Đồng bộ lại vào comparisons
        if (newContrastive.comparisons) {
            const top1 = newContrastive.top1_genre;
            Object.keys(newContrastive.comparisons).forEach((cg) => {
                const comp = newContrastive.comparisons[cg];
                if (!comp || !Array.isArray(comp.dimensions)) return;
                comp.dimensions.forEach((dim) => {
                    const k = dim.id || dim.key;
                    const top1Criteria =
                        newContrastive.genre_profiles[top1]?.criteria ||
                        newContrastive.genre_profiles[top1];
                    const cgCriteria =
                        newContrastive.genre_profiles[cg]?.criteria ||
                        newContrastive.genre_profiles[cg];
                    if (top1Criteria?.[k]) {
                        dim.top1_cell = top1Criteria[k];
                    }
                    if (cgCriteria?.[k]) {
                        dim.contrast_cell = cgCriteria[k];
                    }
                });
            });
        }
    }

    function renderContrastiveMatrix(contrastive, probs, currentPrediction) {
        if (!contrastive) return;

        preserveChunkEvidenceInContrastive(contrastive);
        currentContrastiveData = contrastive;

        const top1 = contrastive.top1_genre || currentPrediction || "Chèo";
        const defaultTop2 = contrastive.top2_genre || "Quan họ";

        // Chọn thể loại đối chứng:
        // Mặc định là thể loại có điểm đầu ra cao thứ 2 (top2_genre).
        // Nếu người dùng đã chọn một thể loại đối chứng khác trước đó (và nó khác top1 mới), giữ nguyên hoặc cho phép chuyển đổi.
        const candidates = Array.isArray(contrastive.candidates)
            ? contrastive.candidates
            : [];
        const validCandidateGenres = candidates.map((c) => c.genre);

        if (
            !selectedContrastGenre ||
            selectedContrastGenre === top1 ||
            (validCandidateGenres.length > 0 && !validCandidateGenres.includes(selectedContrastGenre))
        ) {
            selectedContrastGenre = defaultTop2;
        }

        // Điền dropdown chọn thể loại đối chứng (#contrast-genre-select)
        const selectEl = document.getElementById("contrast-genre-select");
        if (selectEl) {
            selectEl.disabled = false;
            selectEl.innerHTML = "";
            if (candidates.length > 0) {
                candidates.forEach((cand) => {
                    const opt = document.createElement("option");
                    opt.value = cand.genre;
                    const rankNote = cand.is_default_runnerup
                        ? "Mặc định - Cao thứ 2"
                        : `Hạng ${cand.rank}`;
                    opt.textContent = `${cand.genre} (${cand.prob}% • ${rankNote})`;
                    if (cand.genre === selectedContrastGenre) {
                        opt.selected = true;
                    }
                    selectEl.appendChild(opt);
                });
            } else {
                const opt = document.createElement("option");
                opt.value = defaultTop2;
                opt.textContent = `${defaultTop2} (${contrastive.top2_prob || 0}% • Cao thứ 2)`;
                opt.selected = true;
                selectEl.appendChild(opt);
            }
        }

        renderContrastivePairView(selectedContrastGenre);
    }

    function renderContrastivePairView(contrastGenre) {
        if (!currentContrastiveData) return;

        const contrastive = currentContrastiveData;
        const top1 = contrastive.top1_genre || "Chèo";
        const top1Prob = contrastive.top1_prob ?? 0;
        const top1Color =
            contrastive.top1_color || GENRE_COLORS[top1] || "#C84B31";

        const targetContrast =
            contrastGenre && contrastGenre !== top1
                ? contrastGenre
                : contrastive.top2_genre || "Quan họ";
        selectedContrastGenre = targetContrast;

        // Lấy dữ liệu đối sánh từ comparisons[targetContrast]
        const pairData =
            (contrastive.comparisons && contrastive.comparisons[targetContrast]) ||
            contrastive;
        const contrastProb =
            pairData.contrast_prob ?? contrastive.top2_prob ?? 0;
        const contrastColor =
            pairData.contrast_color ||
            GENRE_COLORS[targetContrast] ||
            "#1E6B5B";
        const margin =
            pairData.margin_pct ??
            Number((top1Prob - contrastProb).toFixed(1));

        // Tìm thứ hạng của thể loại đối chứng đang chọn
        const candObj = (contrastive.candidates || []).find(
            (c) => c.genre === targetContrast,
        );
        const rankLabel = candObj
            ? candObj.is_default_runnerup
                ? "MẶC ĐỊNH: ĐIỂM CAO THỨ 2"
                : `ĐỐI CHIẾU TÙY CHỌN: HẠNG ${candObj.rank}`
            : "THỂ LOẠI ĐỐI CHỨNG";

        whatIfState = {
            top1Genre: top1,
            top2Genre: targetContrast,
            top1BaseProb: top1Prob,
            top2BaseProb: contrastProb,
        };

        // 1. Cập nhật Header Metadata & Màu nhận diện riêng của từng thể loại
        const winnerTag = document.getElementById("meta-winner-tag");
        const runnerTag = document.getElementById("meta-runnerup-tag");
        const selectDot = document.getElementById("contrast-select-dot");

        if (winnerTag) {
            winnerTag.innerHTML = `<span class="meta-genre-dot" style="background:${top1Color};"></span> Dự đoán: <strong>${escapeContrastHtml(top1)}</strong> (${top1Prob}%)`;
            winnerTag.style.borderColor = `${top1Color}44`;
        }
        if (runnerTag) {
            runnerTag.innerText = `Đối chứng: ${targetContrast} (${contrastProb}%)`;
        }
        if (selectDot) {
            selectDot.style.backgroundColor = contrastColor;
        }

        // 2. Cập nhật Tiêu đề 2 Cột với màu nhận diện riêng
        const th1Title = document.getElementById("th-top1-title");
        const th1Dot = document.getElementById("th-top1-dot");
        const th1Name = document.getElementById("th-top1-name");
        const th1Prob = document.getElementById("th-top1-prob");

        const th2Title = document.getElementById("th-top2-title");
        const th2Dot = document.getElementById("th-top2-dot");
        const th2Eyebrow = document.getElementById("th-top2-eyebrow");
        const th2Name = document.getElementById("th-top2-name");
        const th2Prob = document.getElementById("th-top2-prob");

        if (th1Title) {
            th1Title.style.borderTop = `3px solid ${top1Color}`;
            th1Title.style.borderLeftColor = `${top1Color}55`;
        }
        if (th1Dot) th1Dot.style.backgroundColor = top1Color;
        if (th1Name) {
            th1Name.innerText = top1;
            th1Name.style.color = top1Color;
        }
        if (th1Prob) {
            th1Prob.innerText = `${top1Prob}%`;
            th1Prob.style.backgroundColor = top1Color;
        }

        if (th2Title) {
            th2Title.style.borderTop = `3px solid ${contrastColor}`;
            th2Title.style.borderRightColor = `${contrastColor}55`;
        }
        if (th2Dot) th2Dot.style.backgroundColor = contrastColor;
        if (th2Eyebrow) th2Eyebrow.innerText = `THỂ LOẠI ĐỐI CHỨNG (${rankLabel})`;
        if (th2Name) {
            th2Name.innerText = targetContrast;
            th2Name.style.color = contrastColor;
        }
        if (th2Prob) {
            th2Prob.innerText = `${contrastProb}%`;
            th2Prob.style.backgroundColor = contrastColor;
        }

        // 3. Hiển thị Ba Nhóm Thông Tin Đối Chứng Tổng Quan (Requirement 3)
        renderThreeContrastiveGroups(
            pairData.groups,
            top1,
            targetContrast,
            top1Color,
            contrastColor,
        );

        // Hiển thị thanh lọc nhanh
        const filterBar = document.getElementById("contrastive-filter-bar");
        if (filterBar) filterBar.style.display = "flex";

        // 4. Hiển thị Bảng Đối chiếu 2 Cột Trực quan theo Từng Tiêu chí (Requirement 4, 5, 7)
        renderTwoColumnContrastiveTable(
            pairData.dimensions || contrastive.dimensions || [],
            top1,
            targetContrast,
            top1Color,
            contrastColor,
        );

        // 5. Cập nhật Hộp Thử Nghiệm Giả Định (What-If Sandbox) cho cặp (top1, targetContrast)
        renderWhatIfSandboxForPair(
            pairData.what_if || contrastive.what_if,
            top1,
            targetContrast,
            margin,
        );
    }

    function renderThreeContrastiveGroups(
        groups,
        top1Genre,
        contrastGenre,
        top1Color,
        contrastColor,
    ) {
        const container = document.getElementById("contrastive-three-groups");
        if (!container) return;

        if (!groups) {
            container.style.display = "none";
            return;
        }
        container.style.display = "grid";

        const indPred = document.getElementById("group-ind-predicted");
        const indCon = document.getElementById("group-ind-contrast");
        if (indPred) indPred.style.backgroundColor = top1Color;
        if (indCon) indCon.style.backgroundColor = contrastColor;

        const titleSup = document.getElementById("group-title-supporting");
        const titleSha = document.getElementById("group-title-shared");
        const titleDis = document.getElementById("group-title-discriminative");
        if (titleSup)
            titleSup.innerHTML = `Đặc trưng ủng hộ <strong style="color:${top1Color}">${escapeContrastHtml(top1Genre)}</strong>`;
        if (titleSha)
            titleSha.innerHTML = `Đặc trưng cũng xuất hiện / phù hợp với <strong style="color:${contrastColor}">${escapeContrastHtml(contrastGenre)}</strong>`;
        if (titleDis)
            titleDis.innerHTML = `Đặc trưng phân biệt giữa <strong>${escapeContrastHtml(top1Genre)}</strong> &amp; <strong>${escapeContrastHtml(contrastGenre)}</strong>`;

        const supList = groups.supporting_predicted || [];
        const shaList = groups.shared_or_contrast || [];
        const disList = groups.discriminative || [];

        const countSup = document.getElementById("group-count-supporting");
        const countSha = document.getElementById("group-count-shared");
        const countDis = document.getElementById("group-count-discriminative");
        if (countSup) countSup.innerText = `${supList.length} tiêu chí`;
        if (countSha) countSha.innerText = `${shaList.length} tiêu chí`;
        if (countDis) countDis.innerText = `${disList.length} tiêu chí`;

        // Render Nhóm 1: Đặc trưng ủng hộ thể loại được dự đoán
        const ulSup = document.getElementById("group-list-supporting");
        if (ulSup) {
            if (supList.length === 0) {
                ulSup.innerHTML = `<li class="group-empty-item">Chưa ghi nhận tiêu chí nào có bằng chứng trực tiếp vượt ngưỡng từ đầu vào; kết quả dự đoán hiện dựa trên phân phối xác suất tổng thể của mạng nơ-ron.</li>`;
            } else {
                ulSup.innerHTML = supList
                    .map(
                        (item) => `
                    <li class="group-item-card" style="border-left: 3px solid ${top1Color};">
                        <div class="group-item-head">
                            <span class="group-item-name"><i class="fa-solid ${item.icon || "fa-check"}"></i> ${escapeContrastHtml(item.criterion_name)}</span>
                            ${item.support_metric ? `<span class="group-item-metric">${escapeContrastHtml(item.support_metric)}</span>` : ""}
                        </div>
                        <div class="group-item-evidence">${escapeContrastHtml(item.input_evidence)}</div>
                        ${renderTimestampChipsHtml(item.timestamps)}
                        ${
                            Array.isArray(item.sources) && item.sources.length > 0
                                ? `<div class="group-item-source"><i class="fa-solid fa-book-bookmark"></i> Nguồn: ${escapeContrastHtml(item.sources[0])}</div>`
                                : ""
                        }
                    </li>
                `,
                    )
                    .join("");
            }
        }

        // Render Nhóm 2: Đặc trưng cũng xuất hiện hoặc có thể phù hợp với thể loại đối chứng
        const ulSha = document.getElementById("group-list-shared");
        if (ulSha) {
            if (shaList.length === 0) {
                ulSha.innerHTML = `<li class="group-empty-item">Không phát hiện đặc trưng nào từ đầu vào trùng khớp hoặc giao thoa đáng kể với <strong>${escapeContrastHtml(contrastGenre)}</strong>. Các tiêu chí của ${escapeContrastHtml(contrastGenre)} đều không xuất hiện trong mẫu này.</li>`;
            } else {
                ulSha.innerHTML = shaList
                    .map(
                        (item) => `
                    <li class="group-item-card" style="border-left: 3px solid ${contrastColor};">
                        <div class="group-item-head">
                            <span class="group-item-name"><i class="fa-solid ${item.icon || "fa-arrows-left-right"}"></i> ${escapeContrastHtml(item.criterion_name)}</span>
                            ${item.support_metric ? `<span class="group-item-metric">${escapeContrastHtml(item.support_metric)}</span>` : ""}
                        </div>
                        <div class="group-item-evidence">${escapeContrastHtml(item.input_evidence)}</div>
                        ${item.overlap_note || item.shared_note ? `<div class="group-item-note">${escapeContrastHtml(item.overlap_note || item.shared_note)}</div>` : ""}
                        ${renderTimestampChipsHtml(item.timestamps)}
                        ${
                            Array.isArray(item.sources) && item.sources.length > 0
                                ? `<div class="group-item-source"><i class="fa-solid fa-book-bookmark"></i> Nguồn: ${escapeContrastHtml(item.sources[0])}</div>`
                                : ""
                        }
                    </li>
                `,
                    )
                    .join("");
            }
        }

        // Render Nhóm 3: Những đặc trưng có giá trị phân biệt giữa hai thể loại
        const ulDis = document.getElementById("group-list-discriminative");
        if (ulDis) {
            if (disList.length === 0) {
                ulDis.innerHTML = `<li class="group-empty-item">Hai thể loại có sự giao thoa cao trên các tiêu chí hiện tại; sự phân biệt dựa trên tổng hợp phân phối xác suất đầu ra.</li>`;
            } else {
                ulDis.innerHTML = disList
                    .map(
                        (item) => `
                    <li class="group-item-card disc-item-card">
                        <div class="group-item-head">
                            <span class="group-item-name"><i class="fa-solid ${item.icon || "fa-scale-balanced"}"></i> ${escapeContrastHtml(item.criterion_name)}</span>
                            <span class="disc-mini-tag">Tiêu chí phân biệt</span>
                        </div>
                        <div class="group-item-reason">${escapeContrastHtml(item.why_discriminative || item.discriminative_reason || "")}</div>
                        ${
                            item.support_metric
                                ? `<div class="group-item-metric-pair"><span>${escapeContrastHtml(item.support_metric)}</span></div>`
                                : item.top1_metric || item.contrast_metric
                                  ? `<div class="group-item-metric-pair">
                                      ${item.top1_metric ? `<span><strong>${escapeContrastHtml(top1Genre)}:</strong> ${escapeContrastHtml(item.top1_metric)}</span>` : ""}
                                      ${item.contrast_metric ? `<span><strong>${escapeContrastHtml(contrastGenre)}:</strong> ${escapeContrastHtml(item.contrast_metric)}</span>` : ""}
                                    </div>`
                                  : ""
                        }
                        ${renderTimestampChipsHtml(item.timestamps)}
                        ${
                            Array.isArray(item.sources) && item.sources.length > 0
                                ? `<div class="group-item-source"><i class="fa-solid fa-book-bookmark"></i> Nguồn: ${escapeContrastHtml(item.sources[0])}</div>`
                                : ""
                        }
                    </li>
                `,
                    )
                    .join("");
            }
        }

        // Gắn sự kiện click cho các chip timestamp trong 3 nhóm
        container.querySelectorAll(".contrast-ts-chip").forEach((btn) => {
            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                const sec = parseFloat(btn.getAttribute("data-seek-sec")) || 0;
                seekToTimestamp(sec);
            });
        });
    }

    function renderCriterionCellHtml(cell, genreColor, isPredictedCol) {
        if (!cell) {
            return `<div class="contrast-cell-empty">Không có dữ liệu tiêu chí</div>`;
        }

        const statusCfg = getStatusBadgeConfig(
            cell.evidence_status,
            cell.evidence_label || cell.evidence_status_label,
        );
        const isMissing =
            cell.evidence_status === "not_detected" ||
            cell.evidence_status === "unverified_modality" ||
            cell.evidence_status === "unverified_context";

        // 1. Mô tả đặc trưng theo kho tri thức (Requirement 4.1)
        const kbDesc = escapeContrastHtml(
            cell.kb_description || "Chưa có mô tả trong kho tri thức.",
        );

        // 2. Bằng chứng thực tế phát hiện từ đầu vào (Requirement 4.2 & Requirement 5)
        const inputEv = escapeContrastHtml(
            cell.input_evidence ||
                "Không phát hiện bằng chứng của tiêu chí này trong mẫu đầu vào.",
        );

        // 3. Mức độ hỗ trợ hoặc điểm số thực tế nếu backend cung cấp (Requirement 4.3 & Note 4)
        const metricHtml = cell.support_metric
            ? `<div class="cell-metric-row">
                 <span class="cell-metric-label"><i class="fa-solid fa-gauge-high"></i> Mức độ hỗ trợ / Số đo thực tế:</span>
                 <span class="cell-metric-val">${escapeContrastHtml(cell.support_metric)}</span>
               </div>`
            : `<div class="cell-metric-row metric-unavailable">
                 <span class="cell-metric-label"><i class="fa-solid fa-circle-info"></i> Mức độ đóng góp riêng:</span>
                 <span class="cell-metric-val-muted">Chỉ hiển thị mô tả &amp; trạng thái kiểm chứng (Backend không tách trọng số đơn lẻ cho tiêu chí này)</span>
               </div>`;

        // 4. Danh sách khái niệm nguyên tử (kb_concepts) & Nguồn tri thức (sources) trong ngăn thu gọn (Requirement 4.4 & UI Collapsible)
        const concepts = Array.isArray(cell.kb_concepts)
            ? cell.kb_concepts
            : [];
        const sources = Array.isArray(cell.sources) ? cell.sources : [];

        const conceptsListHtml =
            concepts.length > 0
                ? `<div class="drawer-section">
                     <div class="drawer-sec-title">Khái niệm chuẩn trong Kho tri thức (${concepts.length}):</div>
                     <ul class="drawer-concept-list">
                       ${concepts
                           .map((c) => {
                               const cSt = c.status || c.evidence_status;
                               const cDetected =
                                   cSt === "detected" || cSt === "shared_signal";
                               const cUnverified =
                                   cSt === "unverified_modality" ||
                                   cSt === "unverified_context";
                               const cBadgeText =
                                   c.status_label ||
                                   (cDetected
                                       ? "Đã phát hiện"
                                       : cUnverified
                                         ? "Chưa kiểm chứng"
                                         : "Không xuất hiện ở đầu vào");
                               const cBadge = cDetected
                                   ? `<span class="mini-ev-tag detected"><i class="fa-solid fa-check"></i> ${escapeContrastHtml(cBadgeText)}</span>`
                                   : cUnverified
                                     ? `<span class="mini-ev-tag unverified"><i class="fa-regular fa-eye-slash"></i> ${escapeContrastHtml(cBadgeText)}</span>`
                                     : `<span class="mini-ev-tag missing"><i class="fa-solid fa-xmark"></i> ${escapeContrastHtml(cBadgeText)}</span>`;
                               return `<li class="drawer-concept-item ${cDetected ? "is-detected" : "is-missing"}">
                                 <div class="drawer-concept-top">
                                   <span class="drawer-concept-id">${escapeContrastHtml(c.concept_id)}</span>
                                   <span class="drawer-concept-level">Mức bằng chứng: ${escapeContrastHtml(c.evidence_level || "--")}</span>
                                   ${cBadge}
                                 </div>
                                 <div class="drawer-concept-sentence">${escapeContrastHtml(c.sentence)}</div>
                                 ${c.evidence_note ? `<div class="drawer-concept-note" style="font-size:10.5px; color:var(--text-tertiary); margin-top:2px;">${escapeContrastHtml(c.evidence_note)}</div>` : ""}
                               </li>`;
                           })
                           .join("")}
                     </ul>
                   </div>`
                : "";

        const sourcesListHtml =
            sources.length > 0
                ? `<div class="drawer-section">
                     <div class="drawer-sec-title"><i class="fa-solid fa-book-open"></i> Nguồn tri thức học thuật liên quan:</div>
                     <ul class="drawer-source-list">
                       ${sources.map((s) => `<li>${escapeContrastHtml(s)}</li>`).join("")}
                     </ul>
                   </div>`
                : "";

        const expandedClass = areAllContrastDetailsExpanded ? "open" : "";

        return `
            <div class="contrast-cell-card ${isMissing ? "cell-is-missing" : "cell-is-active"}" style="--cell-genre-color: ${genreColor};">
                <!-- Trạng thái bằng chứng rõ ràng (Requirement 5) -->
                <div class="cell-status-header">
                    <span class="cell-ev-status-badge ${statusCfg.cls}">
                        <i class="fa-solid ${statusCfg.icon}"></i>
                        <span>${escapeContrastHtml(statusCfg.text)}</span>
                    </span>
                </div>

                <!-- Mô tả đặc trưng theo kho tri thức -->
                <div class="cell-block kb-block">
                    <div class="cell-block-label">Mô tả theo Kho tri thức:</div>
                    <div class="cell-kb-desc">${kbDesc}</div>
                </div>

                <!-- Bằng chứng thực tế phát hiện từ đầu vào -->
                <div class="cell-block evidence-block ${isMissing ? "evidence-missing-box" : "evidence-detected-box"}">
                    <div class="cell-block-label">
                        ${isMissing ? '<i class="fa-regular fa-circle-xmark"></i> Trạng thái kiểm chứng từ đầu vào:' : '<i class="fa-solid fa-magnifying-glass-chart"></i> Bằng chứng thực tế từ đầu vào:'}
                    </div>
                    <div class="cell-input-ev-text">${inputEv}</div>
                    ${renderTimestampChipsHtml(cell.timestamps)}
                </div>

                <!-- Mức độ hỗ trợ / điểm số thực tế -->
                ${metricHtml}

                <!-- Ngăn thu gọn / mở rộng chi tiết khái niệm & nguồn tri thức -->
                <div class="contrast-detail-drawer-wrap">
                    <button type="button" class="btn-cell-drawer-toggle ${expandedClass}">
                        <span><i class="fa-solid fa-book-atlas"></i> Nguồn tri thức (${sources.length}) &amp; Khái niệm chuẩn (${concepts.length})</span>
                        <i class="fa-solid fa-chevron-down drawer-chevron"></i>
                    </button>
                    <div class="contrast-detail-drawer ${expandedClass}">
                        ${conceptsListHtml}
                        ${sourcesListHtml}
                    </div>
                </div>
            </div>
        `;
    }

    function renderTwoColumnContrastiveTable(
        dimensions,
        top1Genre,
        contrastGenre,
        top1Color,
        contrastColor,
    ) {
        const tbody = document.getElementById("contrastive-table-body");
        const aspectList = document.getElementById("aspect-selector-list");
        const aspectBadge = document.getElementById("aspect-count-badge");
        if (!tbody) return;

        tbody.innerHTML = "";

        const filteredDims = (dimensions || []).filter((dim) => {
            const relType = dim.relation_type || dim.comparison_group;
            if (currentContrastFilter === "all") return true;
            if (currentContrastFilter === "discriminative")
                return Boolean(dim.is_discriminative);
            if (currentContrastFilter === "supporting")
                return (
                    relType === "supporting_predicted" ||
                    dim.top1_cell?.evidence_status === "detected"
                );
            if (currentContrastFilter === "shared")
                return (
                    relType === "shared" ||
                    relType === "contrast_leaning" ||
                    dim.contrast_cell?.evidence_status === "detected" ||
                    dim.contrast_cell?.evidence_status === "shared_signal"
                );
            return true;
        });

        if (aspectBadge) {
            aspectBadge.innerText = `${filteredDims.length}`;
        }

        if (filteredDims.length === 0) {
            if (aspectList) {
                aspectList.innerHTML = `<div class="aspect-empty-hint">Không có khía cạnh nào thỏa mãn bộ lọc</div>`;
            }
            tbody.innerHTML = `
                <tr>
                    <td colspan="3" style="text-align: center; padding: 28px 18px; color: var(--text-tertiary);">
                        Không có tiêu chí nào thuộc bộ lọc hiện tại. Hãy chọn "Tất cả tiêu chí" để xem đầy đủ.
                    </td>
                </tr>
            `;
            return;
        }

        const getDimKey = (dim, idx) =>
            String(dim.id || dim.key || dim.name || `dim_${idx}`);

        const validKeys = filteredDims.map((d, i) => getDimKey(d, i));
        if (
            !selectedAspectKey ||
            (selectedAspectKey !== "__ALL__" &&
                !validKeys.includes(selectedAspectKey))
        ) {
            selectedAspectKey = validKeys[0];
        }

        // Render danh sách khía cạnh bên trái (Aspect Selector)
        if (aspectList) {
            aspectList.innerHTML = "";

            filteredDims.forEach((dim, idx) => {
                const dKey = getDimKey(dim, idx);
                const isDisc = Boolean(dim.is_discriminative);
                const relType = dim.relation_type || dim.comparison_group;
                const isSelected = selectedAspectKey === dKey;

                let dotClass = "neutral";
                let subLabel = "Tiêu chí đối sánh";
                if (isDisc) {
                    dotClass = "disc";
                    subLabel = "★ Đặc trưng phân biệt";
                } else if (
                    relType === "shared" ||
                    relType === "contrast_leaning" ||
                    dim.contrast_cell?.evidence_status === "detected" ||
                    dim.contrast_cell?.evidence_status === "shared_signal"
                ) {
                    dotClass = "shared";
                    subLabel = "Giao thoa / Tương đồng";
                } else if (
                    relType === "supporting_predicted" ||
                    dim.top1_cell?.evidence_status === "detected"
                ) {
                    dotClass = "support";
                    subLabel = `Ủng hộ ${top1Genre}`;
                }

                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = `aspect-selector-btn ${isSelected ? "active" : ""}`;
                btn.innerHTML = `
                    <div class="aspect-btn-icon"><i class="fa-solid ${dim.icon || "fa-check"}"></i></div>
                    <div class="aspect-btn-info">
                        <span class="aspect-btn-name">${escapeContrastHtml(dim.name)}</span>
                        <span class="aspect-btn-role">${escapeContrastHtml(subLabel)}</span>
                    </div>
                    <span class="aspect-role-dot ${dotClass}"></span>
                `;
                btn.addEventListener("click", () => {
                    selectedAspectKey = dKey;
                    renderTwoColumnContrastiveTable(
                        dimensions,
                        top1Genre,
                        contrastGenre,
                        top1Color,
                        contrastColor,
                    );
                });
                aspectList.appendChild(btn);
            });

            // Tùy chọn xem đồng thời tất cả khía cạnh trong bộ lọc
            const allBtn = document.createElement("button");
            allBtn.type = "button";
            allBtn.className = `aspect-selector-btn ${selectedAspectKey === "__ALL__" ? "active" : ""}`;
            allBtn.innerHTML = `
                <div class="aspect-btn-icon"><i class="fa-solid fa-table-list"></i></div>
                <div class="aspect-btn-info">
                    <span class="aspect-btn-name">Xem tất cả (${filteredDims.length} khía cạnh)</span>
                    <span class="aspect-btn-role">Hiển thị toàn bộ bảng đối chiếu</span>
                </div>
            `;
            allBtn.addEventListener("click", () => {
                selectedAspectKey = "__ALL__";
                renderTwoColumnContrastiveTable(
                    dimensions,
                    top1Genre,
                    contrastGenre,
                    top1Color,
                    contrastColor,
                );
            });
            aspectList.appendChild(allBtn);
        }

        const dimsToRender =
            selectedAspectKey === "__ALL__"
                ? filteredDims.map((dim, idx) => ({ dim, idx }))
                : filteredDims
                      .map((dim, idx) => ({ dim, idx }))
                      .filter(
                          ({ dim, idx }) =>
                              getDimKey(dim, idx) === selectedAspectKey,
                      );

        dimsToRender.forEach(({ dim, idx }) => {
            const tr = document.createElement("tr");
            const isDisc = Boolean(dim.is_discriminative);
            const relType = dim.relation_type || dim.comparison_group;
            if (isDisc) {
                tr.classList.add("row-discriminative");
            }

            // Nhãn vai trò của tiêu chí
            let roleBadgeHtml = "";
            if (isDisc) {
                roleBadgeHtml = `<span class="criterion-role-badge badge-disc">★ ${escapeContrastHtml(dim.relation_label || "ĐẶC TRƯNG PHÂN BIỆT")}</span>`;
            } else if (relType === "shared") {
                roleBadgeHtml = `<span class="criterion-role-badge badge-shared"><i class="fa-solid fa-link"></i> ${escapeContrastHtml(dim.relation_label || "Đặc trưng giao thoa")}</span>`;
            } else if (relType === "contrast_leaning") {
                roleBadgeHtml = `<span class="criterion-role-badge badge-contrast-lean">${escapeContrastHtml(dim.relation_label || `Tương thích một phần với ${contrastGenre}`)}</span>`;
            } else if (
                relType === "supporting_predicted" ||
                dim.top1_cell?.evidence_status === "detected"
            ) {
                roleBadgeHtml = `<span class="criterion-role-badge badge-support">Ủng hộ ${escapeContrastHtml(top1Genre)}</span>`;
            } else {
                roleBadgeHtml = `<span class="criterion-role-badge badge-neutral">${escapeContrastHtml(dim.relation_label || "Không phát hiện bằng chứng")}</span>`;
            }

            const top1Cell = dim.top1_cell || {
                genre: top1Genre,
                kb_description: dim.top1_desc || "",
                evidence_status: "detected",
                evidence_status_label: "Có bằng chứng từ đầu vào",
                input_evidence: dim.evidence || "",
                support_metric: null,
                timestamps: [],
                sources: [],
                kb_concepts: [],
            };

            const contrastCell = dim.contrast_cell || {
                genre: contrastGenre,
                kb_description: dim.top2_desc || "",
                evidence_status: "not_detected",
                evidence_status_label: "Không phát hiện bằng chứng từ đầu vào",
                input_evidence:
                    "Không phát hiện bằng chứng của tiêu chí này trong mẫu đầu vào.",
                support_metric: null,
                timestamps: [],
                sources: [],
                kb_concepts: [],
            };

            const summaryText = isDisc
                ? dim.discriminative_reason || dim.relation_summary || ""
                : dim.relation_summary || "";

            tr.innerHTML = `
                <td class="dim-header-cell">
                    <div class="dim-header-top">
                        <span class="dim-index-tag">Tiêu chí ${idx + 1}/${filteredDims.length}</span>
                        ${roleBadgeHtml}
                    </div>
                    <div class="contrastive-dim-title">
                        <i class="fa-solid ${dim.icon || "fa-check"}"></i>
                        <span>${escapeContrastHtml(dim.name)}</span>
                    </div>
                    ${
                        summaryText
                            ? `<div class="dim-relation-summary ${isDisc ? "is-disc-summary" : ""}">
                                 ${escapeContrastHtml(summaryText)}
                               </div>`
                            : ""
                    }
                </td>
                <td class="cell-winner" style="--col-accent: ${top1Color};">
                    ${renderCriterionCellHtml(top1Cell, top1Color, true)}
                </td>
                <td class="cell-runner" style="--col-accent: ${contrastColor};">
                    ${renderCriterionCellHtml(contrastCell, contrastColor, false)}
                </td>
            `;

            tbody.appendChild(tr);
        });

        // Gắn sự kiện đóng/mở từng ngăn chi tiết trong bảng
        tbody.querySelectorAll(".btn-cell-drawer-toggle").forEach((btn) => {
            btn.addEventListener("click", () => {
                btn.classList.toggle("open");
                const drawer = btn.nextElementSibling;
                if (drawer) {
                    drawer.classList.toggle("open");
                }
            });
        });

        // Gắn sự kiện tua video/audio khi bấm vào chip timestamp
        tbody.querySelectorAll(".contrast-ts-chip").forEach((btn) => {
            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                const sec = parseFloat(btn.getAttribute("data-seek-sec")) || 0;
                seekToTimestamp(sec);
            });
        });
    }

    function renderWhatIfSandboxForPair(whatIf, top1, contrastGenre, margin) {
        const wName = document.getElementById("whatif-winner-name");
        const rName = document.getElementById("whatif-runner-name");
        const condGrid = document.getElementById("whatif-conditions-grid");
        const slider = document.getElementById("whatif-intervene-slider");
        const sliderVal = document.getElementById("whatif-slider-val");
        const simResult = document.getElementById("whatif-sim-result");
        const statusBadge = document.getElementById("whatif-status-badge");

        if (wName) wName.innerText = top1;
        if (rName) rName.innerText = contrastGenre;

        if (condGrid && whatIf && Array.isArray(whatIf.flip_conditions)) {
            condGrid.innerHTML = "";
            whatIf.flip_conditions.forEach((cond, i) => {
                const cCard = document.createElement("div");
                cCard.className = "whatif-cond-card";
                cCard.innerHTML = `
                    <span class="cond-step-num">${i + 1}</span>
                    <span class="cond-text">${escapeContrastHtml(cond)}</span>
                `;
                condGrid.appendChild(cCard);
            });
        }

        if (statusBadge) {
            statusBadge.innerText = "Kịch bản giả định";
            statusBadge.style.backgroundColor = "rgba(30, 107, 91, 0.12)";
            statusBadge.style.color = "var(--accent-jade)";
        }

        if (slider) {
            slider.value = 50;
            if (sliderVal) {
                sliderVal.innerText = `50% (Cân bằng thực tế)`;
                sliderVal.style.color = "var(--text-secondary)";
            }
            if (simResult) {
                simResult.innerHTML = `
                    <i class="fa-solid fa-circle-info" style="color: var(--accent-indigo);"></i>
                    <span>Thực tế: <strong>${escapeContrastHtml(top1)}</strong> dẫn trước <strong>${escapeContrastHtml(contrastGenre)}</strong> với khoảng cách <strong>+${margin}%</strong>. Kéo thanh trượt sang phải để mô phỏng kịch bản chuyển dịch sang ${escapeContrastHtml(contrastGenre)}.</span>
                `;
                simResult.style.backgroundColor = "#FAF9F6";
                simResult.style.borderColor = "var(--border-hairline)";
            }
        }
    }

    // Sự kiện đổi thể loại đối chứng mà không cần tải lại video (Requirement 1, 6)
    const contrastGenreSelect = document.getElementById(
        "contrast-genre-select",
    );
    if (contrastGenreSelect) {
        contrastGenreSelect.addEventListener("change", (e) => {
            const newGenre = e.target.value;
            if (newGenre && currentContrastiveData) {
                renderContrastivePairView(newGenre);
            }
        });
    }

    // Sự kiện lọc tiêu chí trong bảng đối chiếu 2 cột
    document.querySelectorAll(".contrast-filter-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document
                .querySelectorAll(".contrast-filter-btn")
                .forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            currentContrastFilter = btn.getAttribute("data-filter") || "all";
            if (currentContrastiveData) {
                renderContrastivePairView(selectedContrastGenre);
            }
        });
    });

    // Sự kiện nút Mở rộng / Thu gọn toàn bộ chi tiết khái niệm & nguồn tri thức
    const btnToggleAllContrast = document.getElementById(
        "btn-toggle-all-contrast-details",
    );
    if (btnToggleAllContrast) {
        btnToggleAllContrast.addEventListener("click", () => {
            areAllContrastDetailsExpanded = !areAllContrastDetailsExpanded;
            const textEl = document.getElementById("text-toggle-all-contrast");
            const iconEl = document.getElementById("icon-toggle-all-contrast");
            if (textEl) {
                textEl.innerText = areAllContrastDetailsExpanded
                    ? "Thu gọn chi tiết"
                    : "Mở rộng chi tiết";
            }
            if (iconEl) {
                iconEl.className = areAllContrastDetailsExpanded
                    ? "fa-solid fa-angles-up"
                    : "fa-solid fa-angles-down";
            }
            document
                .querySelectorAll(
                    "#contrastive-table-body .btn-cell-drawer-toggle, #contrastive-table-body .contrast-detail-drawer",
                )
                .forEach((el) => {
                    if (areAllContrastDetailsExpanded) {
                        el.classList.add("open");
                    } else {
                        el.classList.remove("open");
                    }
                });
        });
    }

    // Khởi tạo Event Listener cho thanh trượt What-If Interactive Sandbox
    const whatIfSlider = document.getElementById("whatif-intervene-slider");
    if (whatIfSlider) {
        whatIfSlider.addEventListener("input", (e) => {
            const val = parseInt(e.target.value);
            const sliderVal = document.getElementById("whatif-slider-val");
            const simResult = document.getElementById("whatif-sim-result");
            const statusBadge = document.getElementById("whatif-status-badge");

            if (!whatIfState.top1Genre) return;

            const top1 = whatIfState.top1Genre;
            const top2 = whatIfState.top2Genre;

            // Mô phỏng trọng số can thiệp
            // 50% là gốc. >50% chuyển dịch về top2, <50% củng cố top1
            const delta = (val - 50) * 1.25;
            let simP1 = Math.max(
                2.0,
                Math.min(96.0, whatIfState.top1BaseProb - delta),
            );
            let simP2 = Math.max(
                2.0,
                Math.min(96.0, whatIfState.top2BaseProb + delta),
            );

            const sumSim = simP1 + simP2;
            const basePool =
                whatIfState.top1BaseProb + whatIfState.top2BaseProb;
            const p1Norm = ((simP1 / sumSim) * basePool).toFixed(1);
            const p2Norm = ((simP2 / sumSim) * basePool).toFixed(1);

            if (sliderVal) {
                if (val === 50) {
                    sliderVal.innerText = "50% (Thực tế)";
                    sliderVal.style.color = "var(--text-secondary)";
                } else if (val > 50) {
                    sliderVal.innerText = `${val}% (Nghiêng về ${top2})`;
                    sliderVal.style.color = "var(--accent-jade)";
                } else {
                    sliderVal.innerText = `${val}% (Củng cố ${top1})`;
                    sliderVal.style.color = "var(--accent-vermilion)";
                }
            }

            if (parseFloat(p2Norm) > parseFloat(p1Norm)) {
                if (statusBadge) {
                    statusBadge.innerText = "ĐÃ ĐẢO CHIỀU THÀNH CÔNG";
                    statusBadge.style.backgroundColor =
                        "var(--accent-vermilion-subtle)";
                    statusBadge.style.color = "var(--accent-vermilion)";
                }
                if (simResult) {
                    simResult.innerHTML = `
            <i class="fa-solid fa-triangle-exclamation" style="color: var(--accent-vermilion); font-size: 14px;"></i>
            <span><strong>KỊCH BẢN ĐẢO CHIỀU:</strong> Khi tăng cường độ đặc trưng phân cực của ${top2} lên ${val}%, <strong>${top2} (${p2Norm}%)</strong> đã chính thức vượt qua <strong>${top1} (${p1Norm}%)</strong> và trở thành dự đoán hàng đầu!</span>
          `;
                    simResult.style.backgroundColor =
                        "var(--accent-vermilion-subtle)";
                    simResult.style.borderColor = "rgba(200, 75, 49, 0.3)";
                }
            } else {
                if (statusBadge) {
                    statusBadge.innerText = "Kịch bản giả định";
                    statusBadge.style.backgroundColor =
                        "rgba(30, 107, 91, 0.12)";
                    statusBadge.style.color = "var(--accent-jade)";
                }
                if (simResult) {
                    simResult.innerHTML = `
            <i class="fa-solid fa-circle-info" style="color: var(--accent-jade);"></i>
            <span>Mô phỏng tại ${val}%: <strong>${top1}</strong> = ${p1Norm}% | <strong>${top2}</strong> = ${p2Norm}% (Khoảng cách còn ${Math.abs(p1Norm - p2Norm).toFixed(1)}%).</span>
          `;
                    simResult.style.backgroundColor = "#FAF9F6";
                    simResult.style.borderColor = "var(--border-hairline)";
                }
            }
        });
    }

    // =========================================================================
    // 12. KHỞI TẠO MODAL THÔNG TIN MÔ HÌNH & PHƯƠNG PHÁP NGHIÊN CỨU
    // =========================================================================
    const modalMethodology = document.getElementById("methodology-modal");
    const btnOpenMethodology = document.getElementById("btn-open-methodology");
    const btnCloseMethodology = document.getElementById(
        "btn-close-methodology",
    );
    const btnCloseMethodologyFooter = document.getElementById(
        "btn-close-methodology-footer",
    );

    function openMethodologyModal() {
        if (modalMethodology) modalMethodology.style.display = "flex";
    }

    function closeMethodologyModal() {
        if (modalMethodology) modalMethodology.style.display = "none";
    }

    if (btnOpenMethodology) {
        btnOpenMethodology.addEventListener("click", openMethodologyModal);
    }
    if (btnCloseMethodology) {
        btnCloseMethodology.addEventListener("click", closeMethodologyModal);
    }
    if (btnCloseMethodologyFooter) {
        btnCloseMethodologyFooter.addEventListener(
            "click",
            closeMethodologyModal,
        );
    }
    if (modalMethodology) {
        modalMethodology.addEventListener("click", (e) => {
            if (e.target === modalMethodology) {
                closeMethodologyModal();
            }
        });
    }
    document.addEventListener("keydown", (e) => {
        if (
            e.key === "Escape" &&
            modalMethodology &&
            modalMethodology.style.display === "flex"
        ) {
            closeMethodologyModal();
        }
    });

    // =========================================================================
    // 13. MULTIMODAL EVIDENCE MAP CONTROLLER (Bản đồ bằng chứng đa phương thức)
    // =========================================================================
    let currentEvidenceMapData = null;
    const evidenceMapState = {
        zoom: 1.0,
        panX: 0,
        panY: 0,
        isDragging: false,
        dragStartX: 0,
        dragStartY: 0,
        activeBranch: {
            audio: true,
            visual: true,
            reasoning: true,
        },
        selectedNodeId: null,
        highlightedRuleId: null,
        nodeMap: new Map(),
    };

    const mapSvg = document.getElementById("evidence-map-svg");
    const mapViewport = document.getElementById("evidence-map-viewport");
    const mapHeadersLayer = document.getElementById("map-headers-layer");
    const mapEdgesLayer = document.getElementById("map-edges-layer");
    const mapNodesLayer = document.getElementById("map-nodes-layer");
    const mapOverlayBanner = document.getElementById("map-overlay-banner");
    const mapOverlayBannerText = document.getElementById(
        "map-overlay-banner-text",
    );
    const btnClearRuleHighlight = document.getElementById(
        "btn-clear-rule-highlight",
    );
    const btnZoomIn = document.getElementById("map-zoom-in");
    const btnZoomOut = document.getElementById("map-zoom-out");
    const btnZoomReset = document.getElementById("map-zoom-reset");
    const btnToggleAudio = document.getElementById("map-toggle-audio");
    const btnToggleVisual = document.getElementById("map-toggle-visual");
    const btnToggleReasoning = document.getElementById("map-toggle-reasoning");

    // Inspector Panel Elements
    const mapInspector = document.getElementById("evidence-map-inspector");
    const inspectorNodeType = document.getElementById("inspector-node-type");
    const inspectorNodeName = document.getElementById("inspector-node-name");
    const inspectorCloseBtn = document.getElementById("inspector-close-btn");
    const inspectorEmptyState = document.getElementById(
        "inspector-empty-state",
    );
    const inspectorContent = document.getElementById("inspector-content");
    const inspectorModalityBadge = document.getElementById(
        "inspector-modality-badge",
    );
    const inspectorValBadge = document.getElementById("inspector-val-badge");
    const inspectorRealValueLabel = document.getElementById(
        "inspector-real-value-label",
    );
    const inspectorRealValue = document.getElementById("inspector-real-value");
    const inspectorTimeField = document.getElementById("inspector-time-field");
    const inspectorTimestamps = document.getElementById("inspector-timestamps");
    const inspectorDesc = document.getElementById("inspector-description");
    const inspectorSource = document.getElementById("inspector-source");
    const inspectorRelationsField = document.getElementById(
        "inspector-relations-field",
    );
    const inspectorRelationsLabel = document.getElementById(
        "inspector-relations-label",
    );
    const inspectorRelationsList = document.getElementById(
        "inspector-relations-list",
    );
    const inspectorRuleActions = document.getElementById(
        "inspector-rule-actions",
    );
    const btnInspectorHighlightRule = document.getElementById(
        "btn-inspector-highlight-rule",
    );

    function escapeXml(unsafe) {
        if (unsafe === undefined || unsafe === null) return "";
        return String(unsafe)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&apos;");
    }

    function truncateStr(str, maxLen) {
        if (!str) return "";
        str = String(str);
        return str.length > maxLen ? str.slice(0, maxLen - 1) + "…" : str;
    }

    function applyEvidenceMapTransform() {
        if (!mapViewport) return;
        mapViewport.setAttribute(
            "transform",
            `translate(${evidenceMapState.panX}, ${evidenceMapState.panY}) scale(${evidenceMapState.zoom})`,
        );
    }

    function clearEvidenceMap() {
        currentEvidenceMapData = null;
        evidenceMapState.selectedNodeId = null;
        evidenceMapState.highlightedRuleId = null;
        evidenceMapState.nodeMap.clear();

        if (mapHeadersLayer) mapHeadersLayer.innerHTML = "";
        if (mapEdgesLayer) mapEdgesLayer.innerHTML = "";
        if (mapNodesLayer) mapNodesLayer.innerHTML = "";

        if (mapOverlayBanner) mapOverlayBanner.style.display = "none";

        if (inspectorEmptyState) inspectorEmptyState.style.display = "block";
        if (inspectorContent) inspectorContent.style.display = "none";
        if (inspectorNodeName)
            inspectorNodeName.innerText = "Chưa chọn thành phần";
        if (inspectorNodeType)
            inspectorNodeType.innerText = "Chi tiết Bằng chứng";
    }

    function renderEvidenceMap(data) {
        if (!data || !data.nodes || data.nodes.length === 0) {
            clearEvidenceMap();
            return;
        }

        currentEvidenceMapData = data;
        evidenceMapState.selectedNodeId = null;
        evidenceMapState.highlightedRuleId = null;
        evidenceMapState.nodeMap.clear();

        if (mapOverlayBanner) mapOverlayBanner.style.display = "none";

        // Cập nhật trạng thái nút lọc nhánh thị giác nếu tệp chỉ có âm thanh thuần túy
        if (btnToggleVisual) {
            if (!data.has_visual) {
                btnToggleVisual.classList.remove("active");
                btnToggleVisual.style.opacity = "0.45";
                btnToggleVisual.style.pointerEvents = "none";
                btnToggleVisual.title =
                    "Tệp âm thanh đơn thuần không chứa thông tin thị giác";
            } else {
                btnToggleVisual.style.opacity = "1";
                btnToggleVisual.style.pointerEvents = "auto";
                btnToggleVisual.title = "Bật/Tắt hiển thị nhánh Thị giác";
                if (evidenceMapState.activeBranch.visual) {
                    btnToggleVisual.classList.add("active");
                }
            }
        }

        // 1. Phân chia nodes theo 5 tầng (Layers 0 -> 4)
        const layers = [[], [], [], [], []];
        data.nodes.forEach((node) => {
            const l = Math.max(0, Math.min(4, parseInt(node.layer || 0)));
            layers[l].push(node);
            evidenceMapState.nodeMap.set(node.id, node);
        });

        // Cấu hình kích thước và tọa độ từng cột
        const colConfig = [
            { x: 45, w: 180, h: 56, title: "1. ĐẦU VÀO PHƯƠNG THỨC" },
            { x: 265, w: 200, h: 46, title: "2. ĐẶC TRƯNG TRÍCH XUẤT" },
            { x: 505, w: 205, h: 52, title: "3. KHÁI NIỆM TRUNG GIAN (CBM)" },
            { x: 750, w: 220, h: 58, title: "4. LUẬT SUY LUẬN MỜ (RULES)" },
            { x: 1010, w: 215, h: 64, title: "5. PHÁN QUYẾT DI SẢN (VERDICT)" },
        ];

        const topMargin = 65;

        // Tính toán chiều cao tối đa của các cột để căn giữa dọc (Vertical Centering)
        const colHeights = layers.map((nodeList, idx) => {
            const nodeH = colConfig[idx].h;
            const gap = nodeList.length > 7 ? 8 : 12;
            return nodeList.length > 0
                ? nodeList.length * nodeH + (nodeList.length - 1) * gap
                : 0;
        });
        const maxColHeight = Math.max(...colHeights, 420);

        // Tính toán tọa độ (x, y, w, h) cho từng node
        layers.forEach((nodeList, idx) => {
            const cfg = colConfig[idx];
            const colH = colHeights[idx];
            const gap = nodeList.length > 7 ? 8 : 12;
            const startY = topMargin + Math.max(0, (maxColHeight - colH) / 2);

            nodeList.forEach((node, nodeIdx) => {
                node.x = cfg.x;
                node.y = Math.round(startY + nodeIdx * (cfg.h + gap));
                node.w = cfg.w;
                node.h = cfg.h;
            });
        });

        // 2. Render Header Cột (Tầng 1 -> 5)
        if (mapHeadersLayer) {
            let headersHtml = "";
            colConfig.forEach((cfg, idx) => {
                const count = layers[idx].length;
                const countText = count > 0 ? `(${count})` : "(0)";
                headersHtml += `
          <g class="map-header-group">
            <rect class="map-header-rect" x="${cfg.x}" y="20" width="${cfg.w}" height="28" rx="6" />
            <text class="map-header-text" x="${cfg.x + cfg.w / 2}" y="38">${cfg.title} ${countText}</text>
          </g>
        `;
            });
            mapHeadersLayer.innerHTML = headersHtml;
        }

        // 3. Render Đường nối (Edges - Cubic Bézier Curves)
        if (mapEdgesLayer) {
            let edgesHtml = "";
            (data.links || []).forEach((link) => {
                const src = evidenceMapState.nodeMap.get(link.source);
                const tgt = evidenceMapState.nodeMap.get(link.target);
                if (!src || !tgt) return;

                const x1 = src.x + src.w;
                const y1 = src.y + src.h / 2;
                const x2 = tgt.x;
                const y2 = tgt.y + tgt.h / 2;

                const dx = Math.max(30, (x2 - x1) * 0.45);
                const d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
                const modality = link.modality || "fused";
                const isVerdictEdge = tgt.category === "verdict";
                const edgeClass = isVerdictEdge
                    ? "edge-verdict"
                    : `edge-${modality}`;
                const markerUrl = isVerdictEdge
                    ? "url(#map-arrow-verdict)"
                    : `url(#map-arrow-${modality})`;

                edgesHtml += `
          <path class="evidence-edge ${edgeClass}" 
                data-source="${escapeXml(link.source)}" 
                data-target="${escapeXml(link.target)}" 
                data-modality="${escapeXml(modality)}" 
                d="${d}" 
                marker-end="${markerUrl}">
            <title>${escapeXml(link.label || "")} (Trọng số: ${link.weight || 1.0})</title>
          </path>
        `;
            });
            mapEdgesLayer.innerHTML = edgesHtml;
        }

        // 4. Render Nodes (Cards dạng SVG phong cách Bauhaus / Modern Flat)
        if (mapNodesLayer) {
            let nodesHtml = "";
            data.nodes.forEach((node) => {
                const mod = node.modality || "fused";
                const isVerdict = node.category === "verdict";
                const nodeClass = isVerdict ? "node-verdict" : `node-${mod}`;

                let pillColor = "#65645E";
                if (mod === "audio") pillColor = "#1E6B5B";
                else if (mod === "visual") pillColor = "#C97A1B";
                else if (isVerdict) pillColor = node.color || "#C84B31";

                // Tự động chọn icon biểu trưng
                let iconSymbol = "•";
                if (node.category === "input")
                    iconSymbol = mod === "audio" ? "🎧" : "👁️";
                else if (node.category === "feature") {
                    iconSymbol =
                        node.id === "node_feat_audio_cnn"
                            ? "🧠"
                            : mod === "audio"
                              ? "〰"
                              : "🖼";
                } else if (node.category === "concept") iconSymbol = "✦";
                else if (node.category === "rule") iconSymbol = "⚖";
                else if (isVerdict) iconSymbol = "👑";

                const badgeText = node.badge ? escapeXml(node.badge) : "";
                const badgeW = Math.max(50, badgeText.length * 6.5 + 12);
                const badgeX = node.x + node.w - badgeW - 8;

                nodesHtml += `
          <g class="evidence-node ${nodeClass}" 
             data-id="${escapeXml(node.id)}" 
             data-layer="${node.layer}" 
             data-modality="${escapeXml(mod)}" 
             data-category="${escapeXml(node.category)}">
            <!-- Thẻ nền chính -->
            <rect class="node-card" x="${node.x}" y="${node.y}" width="${node.w}" height="${node.h}" />
            
            <!-- Dải màu chỉ thị phương thức bên trái -->
            <rect class="node-pill" x="${node.x + 3}" y="${node.y + 6}" width="3.5" height="${node.h - 12}" rx="2" fill="${pillColor}" />
            
            <!-- Biểu tượng ký hiệu -->
            <text x="${node.x + 14}" y="${node.y + 20}" font-size="11" fill="${pillColor}">${iconSymbol}</text>
            
            <!-- Tiêu đề Node -->
            <text class="node-title-text" x="${node.x + 28}" y="${node.y + 19}">
              ${escapeXml(truncateStr(node.name, 19))}
            </text>
            
            <!-- Huy hiệu Badge (Định lượng/Trạng thái) -->
            ${
                badgeText
                    ? `
              <rect class="node-badge-rect" x="${badgeX}" y="${node.y + 6}" width="${badgeW}" height="16" fill="rgba(240, 238, 233, 0.9)" stroke="var(--border-hairline)" />
              <text class="node-badge-text" x="${badgeX + badgeW / 2}" y="${node.y + 18}" fill="${pillColor}">
                ${badgeText}
              </text>
            `
                    : ""
            }
            
            <!-- Dòng giá trị trích xuất thực tế / Nhãn định tính -->
            <text class="node-sub-text" x="${node.x + 14}" y="${node.y + (node.h > 56 ? 43 : 38)}">
              ${escapeXml(truncateStr(node.value || "", 26))}
            </text>
          </g>
        `;
            });
            mapNodesLayer.innerHTML = nodesHtml;

            // Gắn sự kiện click vào từng Node
            const nodeElements =
                mapNodesLayer.querySelectorAll(".evidence-node");
            nodeElements.forEach((el) => {
                el.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const nodeId = el.getAttribute("data-id");
                    selectEvidenceNode(nodeId);
                });
            });
        }

        // 5. Căn chỉnh vị trí đồ thị (Fit to View)
        fitEvidenceMapView(false);

        // 6. Áp dụng bộ lọc ẩn/hiện nhánh đang hoạt động
        applyBranchFiltersToDOM();
    }

    function fitEvidenceMapView(animated) {
        if (!mapSvg || !currentEvidenceMapData) return;
        const svgRect = mapSvg.getBoundingClientRect();
        const svgW = svgRect.width || 850;
        const svgH = svgRect.height || 540;

        const graphW = 1270; // 1015 + 215 + 40
        const graphH = 500;

        const scaleX = (svgW - 40) / graphW;
        const scaleY = (svgH - 40) / graphH;
        const targetZoom = Math.min(
            1.05,
            Math.max(0.45, Math.min(scaleX, scaleY)),
        );

        const targetPanX = Math.round((svgW - graphW * targetZoom) / 2);
        const targetPanY = Math.round((svgH - graphH * targetZoom) / 2);

        if (animated && window.gsap) {
            gsap.to(evidenceMapState, {
                zoom: targetZoom,
                panX: targetPanX,
                panY: targetPanY,
                duration: 0.5,
                ease: "power2.out",
                onUpdate: applyEvidenceMapTransform,
            });
        } else {
            evidenceMapState.zoom = targetZoom;
            evidenceMapState.panX = targetPanX;
            evidenceMapState.panY = targetPanY;
            applyEvidenceMapTransform();
        }
    }

    function parseTimeStringToSeconds(timeStr) {
        if (!timeStr) return 0;
        // Lấy phần mốc bắt đầu nếu có khoảng dải "00:10 - 00:21"
        const startPart = timeStr.split("-")[0].trim();
        if (startPart.includes(":")) {
            const parts = startPart
                .split(":")
                .map((p) => parseFloat(p.trim()) || 0);
            if (parts.length === 2) {
                return parts[0] * 60 + parts[1];
            } else if (parts.length === 3) {
                return parts[0] * 3600 + parts[1] * 60 + parts[2];
            }
        }
        const num = parseFloat(startPart.replace(/[^0-9.]/g, ""));
        return isNaN(num) ? 0 : num;
    }

    function seekAndPlayMediaAtTime(seconds) {
        const hasVideo =
            videoPlayer &&
            videoPlayer.src &&
            videoContainer &&
            videoContainer.style.display !== "none";
        const activePl = hasVideo ? videoPlayer : audioPlayer;
        const activeCont = hasVideo ? videoContainer : audioContainer;

        if (activePl && activePl.src) {
            if (activePl.duration && seconds > activePl.duration) {
                seconds = Math.max(0, activePl.duration - 0.5);
            }
            activePl.currentTime = Math.max(0, seconds);
            const playPromise = activePl.play();
            if (playPromise !== undefined) {
                playPromise.catch((err) => {
                    console.warn("Auto play prevented:", err);
                });
            }

            // Tự động cuộn trang mượt mà lên khung phát video/audio để người dùng xem ngay
            if (activeCont) {
                activeCont.scrollIntoView({
                    behavior: "smooth",
                    block: "center",
                });
                activeCont.style.transition = "box-shadow 0.35s ease";
                activeCont.style.boxShadow =
                    "0 0 0 3px var(--accent-vermilion)";
                setTimeout(() => {
                    activeCont.style.boxShadow = "";
                }, 1500);
            }

            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            const formatted = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
            showToast(`Đang phát tại thời điểm mốc: ${formatted}`);
        } else {
            showToast("Chưa có phương tiện video/audio nào đang tải.");
        }
    }

    function selectEvidenceNode(nodeId) {
        const node = evidenceMapState.nodeMap.get(nodeId);
        if (!node) return;

        // Xử lý logic làm nổi bật / tắt nổi bật theo yêu cầu:
        if (node.category === "rule") {
            // 1. Nếu node được click vào là luật: làm nổi bật chuỗi của luật đó
            highlightRuleEvidence(node.id);
        } else if (node.category === "concept") {
            // 2. Nếu node được click vào là khái niệm trung gian:
            if (evidenceMapState.highlightedRuleId) {
                const currentRule = evidenceMapState.nodeMap.get(
                    evidenceMapState.highlightedRuleId,
                );
                const inputConcepts = currentRule
                    ? currentRule.input_concept_ids || []
                    : [];
                // Nếu khái niệm này KHÔNG thuộc chuỗi suy luận đang làm nổi bật -> tắt đi để xem toàn cảnh
                if (!inputConcepts.includes(node.id)) {
                    clearRuleHighlight();
                }
            }
        } else {
            // 3. Nếu click vào node khác ngoài chuỗi (như input hay feature) và đang có luật highlight -> tắt đi để xem toàn cảnh
            if (evidenceMapState.highlightedRuleId) {
                clearRuleHighlight();
            }
        }

        evidenceMapState.selectedNodeId = nodeId;

        // Cập nhật trạng thái viền nổi bật (Active Selected) trên SVG
        if (mapNodesLayer) {
            const allNodes = mapNodesLayer.querySelectorAll(".evidence-node");
            allNodes.forEach((el) => {
                if (el.getAttribute("data-id") === nodeId) {
                    el.classList.add("active-selected");
                } else {
                    el.classList.remove("active-selected");
                }
            });
        }

        // Hiển thị nội dung trong Inspector Panel
        if (inspectorEmptyState) inspectorEmptyState.style.display = "none";
        if (inspectorContent) inspectorContent.style.display = "block";

        // 1. Phân loại node
        const catLabels = {
            input: "DỮ LIỆU ĐẦU VÀO",
            feature: "ĐẶC TRƯNG TRÍCH XUẤT",
            concept: "KHÁI NIỆM TRUNG GIAN (CBM)",
            rule: "LUẬT SUY LUẬN MỜ",
            verdict: "PHÁN QUYẾT DI SẢN CUỐI CÙNG",
        };
        if (inspectorNodeType)
            inspectorNodeType.innerText =
                catLabels[node.category] || "THÀNH PHẦN MINH CHỨNG";
        if (inspectorNodeName)
            inspectorNodeName.innerText = node.name || nodeId;

        // 2. Modality & Value Badge
        if (inspectorModalityBadge) {
            inspectorModalityBadge.className = `inspector-badge ${node.modality || "fused"}`;
            if (node.modality === "audio") {
                inspectorModalityBadge.innerHTML =
                    '<i class="fa-solid fa-headphones"></i> Nhánh Âm thanh';
            } else if (node.modality === "visual") {
                inspectorModalityBadge.innerHTML =
                    '<i class="fa-solid fa-video"></i> Nhánh Thị giác';
            } else {
                inspectorModalityBadge.innerHTML =
                    '<i class="fa-solid fa-scale-balanced"></i> Hợp nhất Suy luận';
            }
        }
        if (inspectorValBadge) {
            inspectorValBadge.innerText =
                node.badge ||
                (node.raw_score !== undefined ? `Điểm: ${node.raw_score}` : "");
        }

        // 3. Giá trị thực tế / Mệnh đề luật mờ (Hỗ trợ Thu gọn / Mở rộng toàn bộ)
        if (inspectorRealValueLabel) {
            inspectorRealValueLabel.innerText =
                node.category === "rule"
                    ? "Mệnh đề logic mờ:"
                    : "Giá trị thực tế:";
        }

        if (inspectorRealValue) {
            if (node.category === "rule") {
                inspectorRealValue.classList.remove("highlight");
                const rawClauses =
                    node.clauses && node.clauses.length
                        ? node.clauses
                        : node.rule_text
                          ? node.rule_text.split(" AND ")
                          : [];
                const consequent = node.consequent || "Di sản";

                let clausesHtml = "";
                rawClauses.forEach((cl, i) => {
                    const prefix = i === 0 ? "IF" : "AND";
                    clausesHtml += `<div class="rule-clause-line"><span class="rule-clause-kw">${prefix}</span> <span class="rule-clause-cond">${escapeXml(cl)}</span></div>`;
                });
                clausesHtml += `<div class="rule-clause-line rule-then"><span class="rule-clause-kw">THEN</span> <span class="rule-clause-target">${escapeXml(consequent)}</span></div>`;

                const needsToggle = rawClauses.length > 2;
                inspectorRealValue.innerHTML = `
          <div class="rule-syntax-box ${needsToggle ? "collapsed" : ""}" id="rule-syntax-box">
            <div class="rule-syntax-inner">
              ${clausesHtml}
            </div>
          </div>
          ${
              needsToggle
                  ? `
          <button type="button" class="btn-toggle-rule-expand" id="btn-toggle-rule-expand">
            <i class="fa-solid fa-chevron-down"></i> Xem toàn bộ luật (${rawClauses.length} mệnh đề)
          </button>
          `
                  : ""
          }
        `;

                if (needsToggle) {
                    const toggleBtn = inspectorRealValue.querySelector(
                        "#btn-toggle-rule-expand",
                    );
                    const syntaxBox =
                        inspectorRealValue.querySelector("#rule-syntax-box");
                    if (toggleBtn && syntaxBox) {
                        toggleBtn.addEventListener("click", () => {
                            const isCollapsed =
                                syntaxBox.classList.contains("collapsed");
                            if (isCollapsed) {
                                syntaxBox.classList.remove("collapsed");
                                syntaxBox.classList.add("expanded");
                                toggleBtn.innerHTML =
                                    '<i class="fa-solid fa-chevron-up"></i> Thu gọn';
                            } else {
                                syntaxBox.classList.remove("expanded");
                                syntaxBox.classList.add("collapsed");
                                toggleBtn.innerHTML = `<i class="fa-solid fa-chevron-down"></i> Xem toàn bộ luật (${rawClauses.length} mệnh đề)`;
                            }
                        });
                    }
                }
            } else {
                inspectorRealValue.classList.add("highlight");
                inspectorRealValue.innerText =
                    node.value ||
                    (node.raw_score !== undefined
                        ? String(node.raw_score)
                        : "Đạt ngưỡng nhận diện");
            }
        }

        // 4. Mốc thời gian liên quan (Cho phép nhấp để nhảy video & cuộn trang)
        if (inspectorTimeField && inspectorTimestamps) {
            if (node.timestamps && node.timestamps.length > 0) {
                inspectorTimeField.style.display = "block";
                inspectorTimestamps.innerHTML = node.timestamps
                    .map(
                        (t) => `
            <button type="button" class="inspector-chip inspector-time-chip" data-time-str="${escapeXml(t)}" title="Bấm để phát tại mốc ${escapeXml(t)} và cuộn lên xem video">
              <i class="fa-solid fa-play" style="font-size: 8.5px; color: var(--accent-vermilion); margin-right: 4px;"></i>${escapeXml(t)}
            </button>
          `,
                    )
                    .join("");

                // Gắn sự kiện click vào các chip thời gian để nhảy video và cuộn trang
                const timeChips = inspectorTimestamps.querySelectorAll(
                    ".inspector-time-chip",
                );
                timeChips.forEach((chip) => {
                    chip.addEventListener("click", () => {
                        const timeStr = chip.getAttribute("data-time-str");
                        const targetSec = parseTimeStringToSeconds(timeStr);
                        seekAndPlayMediaAtTime(targetSec);
                    });
                });
            } else {
                inspectorTimeField.style.display = "block";
                inspectorTimestamps.innerHTML =
                    '<span class="inspector-chip"><i class="fa-solid fa-wave-square"></i> Toàn bài biểu diễn</span>';
            }
        }

        // 5. Diễn giải & Ý nghĩa
        if (inspectorDesc) {
            inspectorDesc.innerText =
                node.description ||
                "Thành phần tham gia vào mạng lưới suy luận mờ đa phương thức.";
        }

        // 6. Nguồn kiểm chứng
        if (inspectorSource) {
            inspectorSource.innerText =
                node.source_evidence || "UNESCO / Viện Âm nhạc Việt Nam";
        }

        // 7. Liên kết suy luận quan hệ (Relations Chips)
        if (
            inspectorRelationsField &&
            inspectorRelationsList &&
            inspectorRelationsLabel
        ) {
            if (node.category === "rule") {
                inspectorRelationsField.style.display = "block";
                inspectorRelationsLabel.innerText =
                    "Các khái niệm tiền đề (IF):";
                const cIds = node.input_concept_ids || [];
                if (cIds.length > 0) {
                    inspectorRelationsList.innerHTML = cIds
                        .map((cid) => {
                            const cNode = evidenceMapState.nodeMap.get(cid);
                            const cName = cNode ? cNode.name : cid;
                            return `<button type="button" class="inspector-chip" data-node-link="${escapeXml(cid)}"><i class="fa-solid fa-sparkles"></i> ${escapeXml(cName)}</button>`;
                        })
                        .join("");
                } else {
                    inspectorRelationsList.innerHTML =
                        "<span style='font-size:11px; color:var(--text-tertiary); font-style:italic;'>Không có khái niệm phụ thuộc trực tiếp.</span>";
                }
            } else if (node.category === "concept") {
                inspectorRelationsField.style.display = "block";
                inspectorRelationsLabel.innerText =
                    "Các luật sử dụng khái niệm này:";
                const rIds = node.used_in_rules || [];
                if (rIds.length > 0) {
                    inspectorRelationsList.innerHTML = rIds
                        .map((rid) => {
                            const rNode = evidenceMapState.nodeMap.get(rid);
                            const rName = rNode ? rNode.name : rid;
                            return `<button type="button" class="inspector-chip" data-node-link="${escapeXml(rid)}"><i class="fa-solid fa-scale-balanced"></i> ${escapeXml(rName)}</button>`;
                        })
                        .join("");
                } else {
                    inspectorRelationsList.innerHTML =
                        "<span style='font-size:11px; color:var(--text-tertiary); font-style:italic;'>Chưa có luật kích hoạt trực tiếp từ khái niệm này.</span>";
                }
            } else {
                inspectorRelationsField.style.display = "none";
            }

            // Gắn sự kiện click chip liên kết để nhảy sang node tương ứng
            const chips =
                inspectorRelationsList.querySelectorAll("[data-node-link]");
            chips.forEach((chip) => {
                chip.addEventListener("click", () => {
                    const targetId = chip.getAttribute("data-node-link");
                    selectEvidenceNode(targetId);
                    if (targetId.startsWith("node_rule_")) {
                        highlightRuleEvidence(targetId);
                    }
                });
            });
        }

        // Nếu người dùng chọn trực tiếp một Luật suy luận, tự động làm nổi bật chuỗi bằng chứng của luật đó (Requirement 5)
        if (node.category === "rule") {
            highlightRuleEvidence(node.id);
        }
    }

    function highlightRuleEvidence(ruleId) {
        if (!currentEvidenceMapData) return;
        const ruleNode = evidenceMapState.nodeMap.get(ruleId);
        if (!ruleNode) return;

        evidenceMapState.highlightedRuleId = ruleId;
        const inputConcepts = new Set(ruleNode.input_concept_ids || []);
        const highlightedNodeIds = new Set([
            ruleId,
            ...inputConcepts,
            "node_verdict",
        ]);

        // Cập nhật class trên Nodes
        if (mapNodesLayer) {
            const allNodes = mapNodesLayer.querySelectorAll(".evidence-node");
            allNodes.forEach((el) => {
                const nid = el.getAttribute("data-id");
                if (highlightedNodeIds.has(nid)) {
                    el.classList.add("active-highlight");
                    el.classList.remove("dimmed");
                } else {
                    el.classList.remove("active-highlight");
                    el.classList.add("dimmed");
                }
            });
        }

        // Cập nhật class trên Edges
        if (mapEdgesLayer) {
            const allEdges = mapEdgesLayer.querySelectorAll(".evidence-edge");
            allEdges.forEach((el) => {
                const src = el.getAttribute("data-source");
                const tgt = el.getAttribute("data-target");

                const isInputToRule = inputConcepts.has(src) && tgt === ruleId;
                const isRuleToVerdict =
                    src === ruleId && tgt === "node_verdict";

                if (isInputToRule || isRuleToVerdict) {
                    el.classList.add("active-highlight-edge");
                    el.classList.remove("dimmed");
                    el.setAttribute("marker-end", "url(#map-arrow-highlight)");
                } else {
                    el.classList.remove("active-highlight-edge");
                    el.classList.add("dimmed");
                    const mod = el.getAttribute("data-modality") || "fused";
                    el.setAttribute(
                        "marker-end",
                        tgt === "node_verdict"
                            ? "url(#map-arrow-verdict)"
                            : `url(#map-arrow-${mod})`,
                    );
                }
            });
        }

        // Không hiển thị banner trạng thái "Đang làm nổi bật chuỗi suy luận" theo yêu cầu người dùng
        if (mapOverlayBanner) {
            mapOverlayBanner.style.display = "none";
        }
    }

    function clearRuleHighlight() {
        evidenceMapState.highlightedRuleId = null;

        if (mapNodesLayer) {
            const allNodes = mapNodesLayer.querySelectorAll(".evidence-node");
            allNodes.forEach((el) => {
                el.classList.remove("active-highlight");
                el.classList.remove("dimmed");
            });
        }

        if (mapEdgesLayer) {
            const allEdges = mapEdgesLayer.querySelectorAll(".evidence-edge");
            allEdges.forEach((el) => {
                el.classList.remove("active-highlight-edge");
                el.classList.remove("dimmed");
                const mod = el.getAttribute("data-modality") || "fused";
                const tgt = el.getAttribute("data-target");
                el.setAttribute(
                    "marker-end",
                    tgt === "node_verdict"
                        ? "url(#map-arrow-verdict)"
                        : `url(#map-arrow-${mod})`,
                );
            });
        }

        if (mapOverlayBanner) {
            mapOverlayBanner.style.display = "none";
        }
    }

    function toggleBranchFilter(branchName) {
        if (!branchName) return;
        const currentState = evidenceMapState.activeBranch[branchName];
        evidenceMapState.activeBranch[branchName] = !currentState;

        const btnMap = {
            audio: btnToggleAudio,
            visual: btnToggleVisual,
            reasoning: btnToggleReasoning,
        };
        const targetBtn = btnMap[branchName];
        if (targetBtn) {
            targetBtn.classList.toggle(
                "active",
                evidenceMapState.activeBranch[branchName],
            );
        }

        applyBranchFiltersToDOM();
    }

    function applyBranchFiltersToDOM() {
        const hiddenNodeIds = new Set();

        if (mapNodesLayer) {
            const allNodes = mapNodesLayer.querySelectorAll(".evidence-node");
            allNodes.forEach((el) => {
                const mod = el.getAttribute("data-modality");
                const nid = el.getAttribute("data-id");
                let isVisible = true;

                if (mod === "audio" && !evidenceMapState.activeBranch.audio)
                    isVisible = false;
                else if (
                    mod === "visual" &&
                    !evidenceMapState.activeBranch.visual
                )
                    isVisible = false;
                else if (
                    mod === "fused" &&
                    !evidenceMapState.activeBranch.reasoning
                )
                    isVisible = false;

                el.classList.toggle("branch-hidden", !isVisible);
                if (!isVisible) hiddenNodeIds.add(nid);
            });
        }

        if (mapEdgesLayer) {
            const allEdges = mapEdgesLayer.querySelectorAll(".evidence-edge");
            allEdges.forEach((el) => {
                const src = el.getAttribute("data-source");
                const tgt = el.getAttribute("data-target");
                const isHidden =
                    hiddenNodeIds.has(src) || hiddenNodeIds.has(tgt);
                el.classList.toggle("branch-hidden", isHidden);
            });
        }
    }

    function initEvidenceMap() {
        // 1. Phóng to / Thu nhỏ / Đặt lại vị trí
        if (btnZoomIn) {
            btnZoomIn.addEventListener("click", () => {
                evidenceMapState.zoom = Math.min(
                    2.5,
                    evidenceMapState.zoom * 1.25,
                );
                applyEvidenceMapTransform();
            });
        }
        if (btnZoomOut) {
            btnZoomOut.addEventListener("click", () => {
                evidenceMapState.zoom = Math.max(
                    0.35,
                    evidenceMapState.zoom * 0.8,
                );
                applyEvidenceMapTransform();
            });
        }
        if (btnZoomReset) {
            btnZoomReset.addEventListener("click", () => {
                fitEvidenceMapView(true);
            });
        }

        // 2. Bộ lọc ẩn/hiện nhánh (Requirement 6)
        if (btnToggleAudio) {
            btnToggleAudio.addEventListener("click", () =>
                toggleBranchFilter("audio"),
            );
        }
        if (btnToggleVisual) {
            btnToggleVisual.addEventListener("click", () =>
                toggleBranchFilter("visual"),
            );
        }
        if (btnToggleReasoning) {
            btnToggleReasoning.addEventListener("click", () =>
                toggleBranchFilter("reasoning"),
            );
        }

        // 3. Kéo chuột để di chuyển (Pan) & Cuộn chuột để Phóng to (Zoom)
        if (mapSvg) {
            mapSvg.addEventListener("mousedown", (e) => {
                if (
                    e.target.closest(".evidence-node") ||
                    e.target.closest("button")
                )
                    return;
                evidenceMapState.isDragging = true;
                evidenceMapState.dragStartX = e.clientX - evidenceMapState.panX;
                evidenceMapState.dragStartY = e.clientY - evidenceMapState.panY;
                mapSvg.style.cursor = "grabbing";
            });

            window.addEventListener("mousemove", (e) => {
                if (!evidenceMapState.isDragging) return;
                evidenceMapState.panX = Math.round(
                    e.clientX - evidenceMapState.dragStartX,
                );
                evidenceMapState.panY = Math.round(
                    e.clientY - evidenceMapState.dragStartY,
                );
                applyEvidenceMapTransform();
            });

            window.addEventListener("mouseup", () => {
                if (evidenceMapState.isDragging) {
                    evidenceMapState.isDragging = false;
                    if (mapSvg) mapSvg.style.cursor = "grab";
                }
            });

            mapSvg.addEventListener(
                "wheel",
                (e) => {
                    e.preventDefault();
                    const factor = e.deltaY < 0 ? 1.12 : 0.89;
                    const prevZoom = evidenceMapState.zoom;
                    const nextZoom = Math.min(
                        2.5,
                        Math.max(0.35, prevZoom * factor),
                    );

                    const rect = mapSvg.getBoundingClientRect();
                    const mouseX = e.clientX - rect.left;
                    const mouseY = e.clientY - rect.top;

                    evidenceMapState.panX = Math.round(
                        mouseX -
                            (mouseX - evidenceMapState.panX) *
                                (nextZoom / prevZoom),
                    );
                    evidenceMapState.panY = Math.round(
                        mouseY -
                            (mouseY - evidenceMapState.panY) *
                                (nextZoom / prevZoom),
                    );
                    evidenceMapState.zoom = nextZoom;
                    applyEvidenceMapTransform();
                },
                { passive: false },
            );

            // Nhấn vào khoảng trống SVG để bỏ chọn làm nổi bật
            mapSvg.addEventListener("click", (e) => {
                if (
                    !e.target.closest(".evidence-node") &&
                    !e.target.closest(".map-branch-btn")
                ) {
                    if (evidenceMapState.highlightedRuleId) {
                        clearRuleHighlight();
                    }
                }
            });
        }

        // 4. Banner nút hủy nổi bật
        if (btnClearRuleHighlight) {
            btnClearRuleHighlight.addEventListener("click", () => {
                clearRuleHighlight();
            });
        }

        // 5. Nút đóng Inspector Panel
        if (inspectorCloseBtn) {
            inspectorCloseBtn.addEventListener("click", () => {
                if (inspectorEmptyState)
                    inspectorEmptyState.style.display = "block";
                if (inspectorContent) inspectorContent.style.display = "none";
                if (mapNodesLayer) {
                    const allNodes =
                        mapNodesLayer.querySelectorAll(".evidence-node");
                    allNodes.forEach((el) =>
                        el.classList.remove("active-selected"),
                    );
                }
                evidenceMapState.selectedNodeId = null;
            });
        }

        // 6. Tự động căn chỉnh khi thay đổi kích thước cửa sổ trình duyệt
        window.addEventListener("resize", () => {
            if (currentEvidenceMapData && !evidenceMapState.isDragging) {
                fitEvidenceMapView(false);
            }
        });
    }

    // Khởi tạo các sự kiện lắng nghe cho Evidence Map
    initEvidenceMap();

    // =========================================================================
    // 14. INTERACTIVE MULTIMODAL WORKSPACE CONTROLLER (DUAL-MODE RIGHT COLUMN + STICKY NAV)
    // =========================================================================

    function switchRightColumnMode(mode, optionalMmTab) {
        currentRightColumnMode = mode === "evidence" ? "evidence" : "verdict";

        const btnVerdict = document.getElementById("btn-right-mode-verdict");
        const btnEvidence = document.getElementById("btn-right-mode-evidence");
        const paneVerdict = document.getElementById("right-mode-verdict-pane");
        const paneEvidence = document.getElementById("right-mode-evidence-pane");

        const isVerdict = currentRightColumnMode === "verdict";

        if (btnVerdict) {
            btnVerdict.classList.toggle("active", isVerdict);
            btnVerdict.setAttribute("aria-selected", isVerdict ? "true" : "false");
        }
        if (btnEvidence) {
            btnEvidence.classList.toggle("active", !isVerdict);
            btnEvidence.setAttribute("aria-selected", !isVerdict ? "true" : "false");
        }

        // Loại trừ tuyệt đối giữa Chế độ A (Nhận diện) và Chế độ B (Bằng chứng)
        if (paneVerdict) {
            paneVerdict.style.display = isVerdict ? "flex" : "none";
            paneVerdict.classList.toggle("active", isVerdict);
        }
        if (paneEvidence) {
            paneEvidence.style.display = isVerdict ? "none" : "block";
            paneEvidence.classList.toggle("active", !isVerdict);
        }

        if (!isVerdict && optionalMmTab) {
            switchMultimodalTab(optionalMmTab);
        } else if (!isVerdict && currentMultimodalTab === "fused") {
            setTimeout(() => {
                renderTimelineChart(
                    currentTimelineSeries && currentTimelineSeries.length > 0
                        ? currentTimelineSeries
                        : currentRawChunks,
                    currentAnalysis ? currentAnalysis.duration_seconds : 0,
                );
            }, 40);
        }

        // Đồng bộ trạng thái nút 1 (Tổng quan) và nút 2 (Bằng chứng) trên Sticky Nav nếu đang ở vùng đầu trang
        const overviewSec = document.getElementById("sec-overview");
        if (overviewSec && overviewSec.classList.contains("active-ws-section")) {
            const activeNavTarget = isVerdict ? "sec-overview" : "sec-multimodal";
            document.querySelectorAll(".ws-nav-btn").forEach((btn) => {
                const t = btn.getAttribute("data-target");
                if (t === "sec-overview" || t === "sec-multimodal") {
                    btn.classList.toggle("active", t === activeNavTarget);
                }
            });
        }
    }

    // =========================================================================
    // SLIDE-OVER DRAWER FOR DEEP-DIVE ANALYSIS (GIAO DIỆN 1 MÀN HÌNH DASHBOARD)
    // =========================================================================
    const DRAWER_PANES_META = {
        "sec-concept-graph": {
            badge: "KHU VỰC 3",
            title: "Phân tích chuyên sâu: Đồ thị Concept–Rule (Evidence Map)",
        },
        "sec-rule-explorer": {
            badge: "KHU VỰC 4",
            title: "Khám phá luật suy luận (Rule Explorer)",
        },
        "sec-explanation-explorer": {
            badge: "KHU VỰC 5",
            title: "Trình khám phá đối chứng (Explanation Explorer)",
        },
        "sec-summary": {
            badge: "KHU VỰC 6",
            title: "Tổng kết & Kết luận phân tích (Executive Summary)",
        },
    };

    function openSlideDrawer(sectionId) {
        const targetId = sectionId || "sec-concept-graph";
        const drawer = document.getElementById("slide-drawer");
        const backdrop = document.getElementById("drawer-backdrop");
        if (!drawer) return;

        drawer.classList.add("open");
        drawer.setAttribute("aria-hidden", "false");
        if (backdrop) backdrop.classList.add("open");

        switchDrawerPane(targetId);
    }

    function closeSlideDrawer() {
        const drawer = document.getElementById("slide-drawer");
        const backdrop = document.getElementById("drawer-backdrop");
        if (drawer) {
            drawer.classList.remove("open");
            drawer.setAttribute("aria-hidden", "true");
        }
        if (backdrop) {
            backdrop.classList.remove("open");
        }
        document.querySelectorAll(".dock-edge-tab").forEach((tab) => {
            tab.classList.remove("active");
        });
    }

    function switchDrawerPane(targetSectionId) {
        const meta =
            DRAWER_PANES_META[targetSectionId] ||
            DRAWER_PANES_META["sec-concept-graph"];
        const badgeEl = document.getElementById("drawer-step-badge");
        const titleEl = document.getElementById("drawer-header-title");
        if (badgeEl && meta) badgeEl.innerText = meta.badge;
        if (titleEl && meta) titleEl.innerText = meta.title;

        // Kích hoạt pane tương ứng và ẩn các pane còn lại
        document.querySelectorAll(".drawer-pane").forEach((pane) => {
            const isMatch = pane.getAttribute("data-pane") === targetSectionId;
            pane.classList.toggle("active", isMatch);
            pane.style.display = isMatch ? "block" : "none";
        });

        // Cập nhật nút nhanh trên Header của Drawer
        document.querySelectorAll(".drawer-quick-btn").forEach((btn) => {
            btn.classList.toggle(
                "active",
                btn.getAttribute("data-drawer-target") === targetSectionId,
            );
        });

        // Cập nhật thẻ mũi tên active ở cạnh phải màn hình
        document.querySelectorAll(".dock-edge-tab").forEach((tab) => {
            tab.classList.toggle(
                "active",
                tab.getAttribute("data-open-drawer") === targetSectionId,
            );
        });

        // Đảm bảo section được active trong DOM
        document.querySelectorAll(".workspace-section").forEach((sec) => {
            sec.classList.toggle("active-ws-section", sec.id === targetSectionId);
        });

        // Nếu mở Đồ thị Concept-Rule, tự động canh chỉnh viewBox SVG
        if (targetSectionId === "sec-concept-graph") {
            setTimeout(() => {
                if (typeof fitEvidenceMapView === "function") {
                    fitEvidenceMapView(false);
                }
            }, 180);
        }
    }

    function activateWorkspaceSection(targetSectionId, shouldScroll = true) {
        if (!targetSectionId) return;

        // Nếu mục tiêu thuộc một trong các công cụ phân tích chuyên sâu (Khu vực 3, 4, 5, 6): mở Slide Drawer
        if (DRAWER_PANES_META[targetSectionId]) {
            openSlideDrawer(targetSectionId);
            return;
        }

        // Nếu là Khu vực Tổng quan (sec-overview) hoặc Bằng chứng (sec-multimodal), đóng Drawer nếu đang mở
        closeSlideDrawer();

        // 1. Cập nhật nút active trên Sticky Navigation (nếu có)
        document.querySelectorAll(".ws-nav-btn").forEach((btn) => {
            const isTarget =
                btn.getAttribute("data-target") === targetSectionId;
            btn.classList.toggle("active", isTarget);
        });

        // 2. Cập nhật trạng thái active-ws-section trên các section
        document.querySelectorAll(".workspace-section").forEach((sec) => {
            sec.classList.toggle(
                "active-ws-section",
                sec.id === targetSectionId,
            );
        });

        const targetEl = document.getElementById(targetSectionId);
        if (!targetEl) return;

        // Nếu khu vực đang thu gọn (collapsible-section collapsed), tự động mở rộng để hiển thị nội dung
        if (
            targetEl.classList.contains("collapsible-section") &&
            targetEl.classList.contains("collapsed")
        ) {
            targetEl.classList.remove("collapsed");
            const heading = targetEl.querySelector(".accordion-heading-bar");
            if (heading) {
                heading.setAttribute("aria-expanded", "true");
                const toggleTxt = heading.querySelector(".accordion-toggle-text");
                if (toggleTxt) toggleTxt.innerText = "Thu gọn";
            }
        }

        if (
            targetSectionId === "sec-multimodal" &&
            currentMultimodalTab === "fused"
        ) {
            setTimeout(() => {
                renderTimelineChart(
                    currentTimelineSeries && currentTimelineSeries.length > 0
                        ? currentTimelineSeries
                        : currentRawChunks,
                    currentAnalysis ? currentAnalysis.duration_seconds : 0,
                );
            }, 60);
        }

        // Cuộn mượt tới khu vực nếu ở chế độ Scroll liên tục
        if (shouldScroll) {
            targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }

    function setWorkspaceViewMode(mode) {
        currentWorkspaceMode = mode === "focus" ? "focus" : "scroll";
        const workspaceEl = document.getElementById("analysis-workspace");
        const btnScroll = document.getElementById("btn-ws-mode-scroll");
        const btnFocus = document.getElementById("btn-ws-mode-focus");

        if (workspaceEl) {
            workspaceEl.classList.toggle(
                "focus-mode",
                currentWorkspaceMode === "focus",
            );
        }
        if (btnScroll) {
            btnScroll.classList.toggle(
                "active",
                currentWorkspaceMode === "scroll",
            );
        }
        if (btnFocus) {
            btnFocus.classList.toggle(
                "active",
                currentWorkspaceMode === "focus",
            );
        }

        // Đảm bảo có ít nhất 1 section active khi chuyển sang focus mode
        const activeBtn = document.querySelector(".ws-nav-btn.active");
        const targetId = activeBtn
            ? activeBtn.getAttribute("data-target")
            : "sec-overview";
        activateWorkspaceSection(targetId, false);
    }

    function switchExpExplorerTab(expTab) {
        currentExpExplorerTab = expTab || "aspect";

        document.querySelectorAll(".exp-tab-btn").forEach((btn) => {
            const isMatch =
                btn.getAttribute("data-exp-tab") === currentExpExplorerTab;
            btn.classList.toggle("active", isMatch);
        });

        const paneAspect = document.getElementById("exp-pane-aspect");
        const paneGroups = document.getElementById("exp-pane-groups");
        const paneWhatif = document.getElementById("exp-pane-whatif");

        if (paneAspect) {
            paneAspect.style.display =
                currentExpExplorerTab === "aspect" ? "block" : "none";
            paneAspect.classList.toggle(
                "active",
                currentExpExplorerTab === "aspect",
            );
        }
        if (paneGroups) {
            paneGroups.style.display =
                currentExpExplorerTab === "groups" ? "block" : "none";
            paneGroups.classList.toggle(
                "active",
                currentExpExplorerTab === "groups",
            );
        }
        if (paneWhatif) {
            paneWhatif.style.display =
                currentExpExplorerTab === "whatif" ? "block" : "none";
            paneWhatif.classList.toggle(
                "active",
                currentExpExplorerTab === "whatif",
            );
        }
    }

    function renderExecutiveSummary(result) {
        if (!result) return;

        const tkVerdictVal = document.getElementById("takeaway-verdict-val");
        const tkVerdictSub = document.getElementById("takeaway-verdict-sub");
        const tkModVal = document.getElementById("takeaway-modality-val");
        const tkModSub = document.getElementById("takeaway-modality-sub");
        const tkRuleVal = document.getElementById("takeaway-rule-val");
        const tkRuleSub = document.getElementById("takeaway-rule-sub");
        const tkDiscVal = document.getElementById("takeaway-disc-val");
        const tkDiscSub = document.getElementById("takeaway-disc-sub");

        const gColor =
            result.genre_color || GENRE_COLORS[result.prediction] || "#C84B31";

        // 1. Kết luận nhận diện chính
        if (tkVerdictVal) {
            tkVerdictVal.innerText = result.prediction || "Chưa xác định";
            tkVerdictVal.style.color = gColor;
        }
        if (tkVerdictSub) {
            tkVerdictSub.innerText = `Độ tin cậy: ${Number(result.confidence || 0).toFixed(1)}% • Thời lượng: ${result.duration_seconds || "--"}s`;
        }

        // 2. Đóng góp Đa phương thức
        const mods = result.modalities || {};
        const audPred = mods.audio?.prediction || result.prediction;
        const audConf = mods.audio?.confidence ?? result.confidence;
        const visPred = mods.visual?.prediction;
        const visConf = mods.visual?.confidence;
        if (tkModVal) {
            if (visPred) {
                tkModVal.innerText =
                    audPred === visPred
                        ? `Đồng thuận: ${audPred}`
                        : `Âm thanh: ${audPred} • Thị giác: ${visPred}`;
            } else {
                tkModVal.innerText = `Âm thanh: ${audPred} (${audConf}%)`;
            }
        }
        if (tkModSub) {
            const activeConcepts = Array.isArray(result.active_concepts)
                ? result.active_concepts
                : [];
            const aCnt = activeConcepts.filter(
                (c) => c.type === "audio" && c.enabled !== false,
            ).length;
            const vCnt = activeConcepts.filter(
                (c) => c.type !== "audio" && c.enabled !== false,
            ).length;
            tkModSub.innerText = visPred
                ? `${aCnt} khái niệm âm thanh • ${vCnt} khái niệm thị giác (Thị giác ${visConf}%)`
                : `${aCnt} khái niệm âm thanh kích hoạt (Chế độ Audio)`;
        }

        // 3. Luật suy luận quyết định
        const rules = Array.isArray(result.triggered_rules)
            ? result.triggered_rules
            : [];
        if (tkRuleVal) {
            if (rules.length > 0) {
                const topRule = rules[0];
                const topRConf =
                    topRule.confidence <= 1.0
                        ? (topRule.confidence * 100).toFixed(1)
                        : Number(topRule.confidence).toFixed(1);
                tkRuleVal.innerText = `${rules.length} luật (Top #1: ${topRConf}%)`;
            } else {
                tkRuleVal.innerText = "0 luật kích hoạt";
            }
        }
        if (tkRuleSub) {
            if (rules.length > 0 && rules[0].rule_text) {
                tkRuleSub.innerText = `IF ${rules[0].rule_text} ➜ ${rules[0].consequent}`;
            } else {
                tkRuleSub.innerText = "Suy luận dựa trên mạng nơ-ron cơ sở";
            }
        }

        // 4. Điểm phân biệt đối chiếu
        const ca = result.contrastive_analysis;
        if (ca) {
            const runnerGenre = selectedContrastGenre || ca.top2_genre || "Hạng 2";
            const pairData =
                (ca.comparisons && ca.comparisons[runnerGenre]) || ca;
            const margin =
                pairData.margin_pct ??
                Number(((ca.top1_prob || 0) - (ca.top2_prob || 0)).toFixed(1));
            const discList = pairData.groups?.discriminative || [];

            if (tkDiscVal) {
                tkDiscVal.innerText = `Vượt ${runnerGenre} (+${margin}%)`;
            }
            if (tkDiscSub) {
                if (discList.length > 0) {
                    const names = discList
                        .slice(0, 2)
                        .map((d) => d.criterion_name)
                        .join(", ");
                    tkDiscSub.innerText = `Phân biệt bởi: ${names}`;
                } else {
                    tkDiscSub.innerText = `Đối chiếu trên ${(pairData.dimensions || []).length} khía cạnh học thuật`;
                }
            }
        }
    }

    function initWorkspaceNavigation() {
        // 1. Sự kiện click thanh điều hướng nội bộ (Sticky Nav)
        document.querySelectorAll(".ws-nav-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const targetId = btn.getAttribute("data-target");
                activateWorkspaceSection(targetId, true);
            });
        });

        // 2. Sự kiện chuyển chế độ Hiển thị (Cuộn liên tục vs Tập trung 1 khu vực)
        const btnModeScroll = document.getElementById("btn-ws-mode-scroll");
        const btnModeFocus = document.getElementById("btn-ws-mode-focus");
        if (btnModeScroll) {
            btnModeScroll.addEventListener("click", () =>
                setWorkspaceViewMode("scroll"),
            );
        }
        if (btnModeFocus) {
            btnModeFocus.addEventListener("click", () =>
                setWorkspaceViewMode("focus"),
            );
        }

        // 2b. Xử lý Accordion thu gọn / mở rộng cho các Khu vực 3, 4, 5, 6 (mặc định thu gọn)
        document.querySelectorAll(".accordion-heading-bar").forEach((heading) => {
            const toggleSection = () => {
                const sec = heading.closest(".collapsible-section");
                if (!sec) return;
                const isNowCollapsed = sec.classList.toggle("collapsed");
                heading.setAttribute(
                    "aria-expanded",
                    isNowCollapsed ? "false" : "true",
                );
                const toggleTxt = heading.querySelector(".accordion-toggle-text");
                if (toggleTxt) {
                    toggleTxt.innerText = isNowCollapsed ? "Mở rộng" : "Thu gọn";
                }
                if (
                    !isNowCollapsed &&
                    sec.id === "sec-concept-graph" &&
                    typeof fitEvidenceMapView === "function"
                ) {
                    setTimeout(() => fitEvidenceMapView(false), 80);
                }
            };

            heading.addEventListener("click", () => {
                toggleSection();
            });

            heading.addEventListener("keydown", (e) => {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    toggleSection();
                }
            });
        });

        // 3. Sự kiện chuyển đổi 2 chế độ Cột phải: [ Nhận diện ] | [ Bằng chứng đa phương thức ]
        const btnRightVerdict = document.getElementById(
            "btn-right-mode-verdict",
        );
        const btnRightEvidence = document.getElementById(
            "btn-right-mode-evidence",
        );
        if (btnRightVerdict) {
            btnRightVerdict.addEventListener("click", () => {
                switchRightColumnMode("verdict");
            });
        }
        if (btnRightEvidence) {
            btnRightEvidence.addEventListener("click", () => {
                switchRightColumnMode("evidence");
            });
        }

        // 4. Sự kiện các thẻ lối tắt chuyển thẳng sang Chế độ Bằng chứng theo từng Tab ([data-switch-evidence-tab])
        document
            .querySelectorAll("[data-switch-evidence-tab]")
            .forEach((el) => {
                const handleSwitchEvidence = (e) => {
                    e.preventDefault();
                    const targetTab = el.getAttribute(
                        "data-switch-evidence-tab",
                    );
                    switchRightColumnMode("evidence", targetTab || "audio");
                };
                el.addEventListener("click", handleSwitchEvidence);
                el.addEventListener("keydown", (e) => {
                    if (e.key === "Enter" || e.key === " ") {
                        handleSwitchEvidence(e);
                    }
                });
            });

        // 5. Sự kiện các nút nhảy nhanh giữa các khu vực ([data-jump-section] & [data-jump-mmtab])
        document.querySelectorAll("[data-jump-section]").forEach((el) => {
            const handleJump = (e) => {
                e.preventDefault();
                const targetSec = el.getAttribute("data-jump-section");
                const targetMmTab = el.getAttribute("data-jump-mmtab");
                if (targetMmTab) {
                    switchMultimodalTab(targetMmTab);
                }
                if (targetSec) {
                    activateWorkspaceSection(targetSec, true);
                }
            };
            el.addEventListener("click", handleJump);
            el.addEventListener("keydown", (e) => {
                if (e.key === "Enter" || e.key === " ") {
                    handleJump(e);
                }
            });
        });

        // 6. Sự kiện chuyển 3 Tab trong Khu vực 5 (Explanation Explorer)
        document.querySelectorAll(".exp-tab-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const expTab = btn.getAttribute("data-exp-tab");
                switchExpExplorerTab(expTab);
            });
        });

        // 7. Sự kiện Accordion mở/đóng Tài liệu viện dẫn học thuật (Khu vực 6)
        const btnCitations = document.getElementById("btn-toggle-citations");
        const bodyCitations = document.getElementById(
            "citations-accordion-body",
        );
        const iconCitations = document.getElementById("citations-acc-icon");
        if (btnCitations && bodyCitations) {
            btnCitations.addEventListener("click", () => {
                const isCollapsed =
                    bodyCitations.classList.toggle("collapsed");
                btnCitations.setAttribute(
                    "aria-expanded",
                    isCollapsed ? "false" : "true",
                );
                if (iconCitations) {
                    iconCitations.className = isCollapsed
                        ? "fa-solid fa-chevron-down"
                        : "fa-solid fa-chevron-up";
                }
            });
        }

        // 8. Sự kiện mở Slide Drawer từ các thẻ mũi tên dọc cạnh phải màn hình (.dock-edge-tab)
        document.querySelectorAll(".dock-edge-tab").forEach((tab) => {
            tab.addEventListener("click", () => {
                const targetSec = tab.getAttribute("data-open-drawer");
                const drawer = document.getElementById("slide-drawer");
                const isOpen = drawer && drawer.classList.contains("open");
                const isCurrentActive = tab.classList.contains("active");
                if (isOpen && isCurrentActive) {
                    closeSlideDrawer();
                } else {
                    openSlideDrawer(targetSec);
                }
            });
        });

        // 9. Sự kiện chuyển Tab nhanh ngay trong Header của Slide Drawer (.drawer-quick-btn)
        document.querySelectorAll(".drawer-quick-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const targetSec = btn.getAttribute("data-drawer-target");
                switchDrawerPane(targetSec);
            });
        });

        // 10. Sự kiện đóng Slide Drawer (Nút Đóng, Backdrop mờ, và phím Escape)
        const btnCloseDrawer = document.getElementById("drawer-close-btn");
        if (btnCloseDrawer) {
            btnCloseDrawer.addEventListener("click", closeSlideDrawer);
        }
        const drawerBackdrop = document.getElementById("drawer-backdrop");
        if (drawerBackdrop) {
            drawerBackdrop.addEventListener("click", closeSlideDrawer);
        }
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                const drawer = document.getElementById("slide-drawer");
                if (drawer && drawer.classList.contains("open")) {
                    closeSlideDrawer();
                }
            }
        });

        // Khởi tạo trạng thái mặc định cho Cột phải, Tab Đa phương thức & Explanation Explorer
        setupTimelineSubtabs();
        switchRightColumnMode("verdict");
        switchMultimodalTab("visual");
        switchExpExplorerTab("aspect");

        // Expose helper functions globally for programmatic access
        window.openSlideDrawer = openSlideDrawer;
        window.closeSlideDrawer = closeSlideDrawer;
        window.switchDrawerPane = switchDrawerPane;
    }

    initWorkspaceNavigation();
});

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
  const graph3d = new SemanticGraph3D("graph-canvas-container", "graph-tooltip");
  const recorder = new LiveAudioRecorder({
    timerId: "mic-timer",
    ringId: "mic-timer-ring",
    statusId: "mic-status-text"
  });

  // Lưu trữ tham chiếu toàn cục
  window.audioWave = audioWave;
  window.graph3d = graph3d;

  // Bảng màu 6 Di sản Văn hóa Âm nhạc Việt Nam chuẩn hóa đồng bộ với Backend
  const GENRE_COLORS = {
    "Ca trù": "#C84B31",     // Đỏ gạch nung (Vermilion)
    "Chèo": "#2E5A88",       // Xanh chàm đình làng (Indigo)
    "Chầu văn": "#8C3B88",    // Tím khăn chầu áo ngự (Ceremonial Purple)
    "Cải lương": "#C97A1B",   // Vàng đồng / Hổ phách (Amber Ochre)
    "Hát xẩm": "#7A6248",     // Nâu đất chiếu xẩm (Earth Brown)
    "Quan họ": "#1E6B5B"      // Xanh ngọc quan họ (Jade Emerald)
  };

  // State quản lý kết quả hiện tại & Nguồn đầu vào duy nhất (Single Active Input)
  let currentAnalysis = null;
  let currentActiveTab = "media"; // "media" | "mic" | "url"
  let activeMediaType = null;     // "video" | "audio" | null (Chỉ cho phép 1 input tồn tại tại 1 thời điểm)

  // State quản lý phân đoạn thời gian, gộp đoạn thông minh & XAI
  let currentTimelineMode = "merged"; // "merged" | "raw"
  let currentRawChunks = [];
  let currentMergedSegments = [];
  let currentTimelineSeries = [];
  let currentOverallExplanation = null;
  let currentOverallModalities = null;
  let activeSelectedSegmentId = null;
  let playbackSyncHandler = null;
  let currentTimelineSubtab = "filmstrip";

  // Tham chiếu các phần tử DOM chính
  const mediaInput = document.getElementById("media-file-input") || document.getElementById("video-file-input");
  const mediaDropzone = document.getElementById("media-dropzone") || document.getElementById("video-dropzone");
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
      mediaDropzone.style.display = (keepType === null) ? "block" : "none";
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
    currentTimelineMode = "merged";
    currentRawChunks = [];
    currentMergedSegments = [];
    currentTimelineSeries = [];
    currentOverallExplanation = null;
    currentOverallModalities = null;
    activeSelectedSegmentId = null;

    const titleEl = document.getElementById("verdict-genre-title");
    const confEl = document.getElementById("confidence-number");
    const durEl = document.getElementById("verdict-duration");

    if (titleEl) {
      titleEl.innerText = "Chưa có dữ liệu";
      titleEl.style.color = "var(--text-tertiary)";
    }
    if (confEl) confEl.innerText = "--.-%";
    if (durEl) durEl.innerText = "Thời lượng phân tích: --";

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
    if (headlineEl) headlineEl.innerText = "Chưa có kết quả phân tích";
    if (bodyEl) bodyEl.innerText = "Hãy tải lên video/audio hoặc thu âm trực tiếp để hệ thống thực hiện suy luận mờ 2 tầng và sinh lời lý giải khoa học.";

    const xaiBanner = document.getElementById("xai-segment-banner");
    if (xaiBanner) xaiBanner.style.display = "none";
    const xaiModsGrid = document.getElementById("xai-modalities-grid");
    if (xaiModsGrid) xaiModsGrid.style.display = "none";

    renderConceptPills([]);

    const rulesContainer = document.getElementById("rules-list-container");
    if (rulesContainer) {
      rulesContainer.innerHTML = "<span style='font-size: 12px; color: var(--text-tertiary);'>Chưa có luật kích hoạt.</span>";
    }

    const stripContainer = document.getElementById("timeline-scroll-strip");
    if (stripContainer) {
      stripContainer.innerHTML = "<p style='color: var(--text-secondary); font-size: 12px; grid-column: 1 / -1; padding: 12px 0;'>Chưa có dữ liệu phân đoạn. Hãy chọn một mẫu phía trên hoặc tải lên file để phân tích.</p>";
    }
    const modeBar = document.getElementById("timeline-mode-bar");
    if (modeBar) modeBar.style.display = "none";
    const ribbonWrap = document.getElementById("genre-timeline-ribbon-wrap");
    if (ribbonWrap) ribbonWrap.style.display = "none";
    const chartView = document.getElementById("timeline-chart-container");
    if (chartView) chartView.style.display = "none";
    const specContainer = document.getElementById("spectrogram-grid-container");
    if (specContainer) specContainer.innerHTML = "";
    const radarContainer = document.getElementById("radar-bars-container");
    if (radarContainer) radarContainer.innerHTML = "";

    const tableBody = document.getElementById("contrastive-table-body");
    if (tableBody) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="4" style="text-align: center; padding: 36px 20px; color: var(--text-tertiary);">
            <i class="fa-solid fa-arrows-split-up-and-left" style="font-size: 26px; margin-bottom: 8px; display: block; opacity: 0.5;"></i>
            Vui lòng chọn một mẫu âm thanh/video hoặc tải lên tệp để kích hoạt Ma trận Đối sánh Phân biệt &amp; Hộp cát Thử nghiệm What-If.
          </td>
        </tr>
      `;
    }
    const metaWinner = document.getElementById("meta-winner-tag");
    const metaRunner = document.getElementById("meta-runnerup-tag");
    if (metaWinner) metaWinner.innerText = "Top 1: Đang chờ";
    if (metaRunner) metaRunner.innerText = "Top 2: Đang chờ";
    const thTop1Prob = document.getElementById("th-top1-prob");
    const thTop2Prob = document.getElementById("th-top2-prob");
    if (thTop1Prob) thTop1Prob.innerText = "--%";
    if (thTop2Prob) thTop2Prob.innerText = "--%";
  }

  // Nút thu nhỏ / mở rộng danh sách xác suất ô dự đoán
  let isVerdictExpanded = false;
  const btnToggleVerdict = document.getElementById("btn-toggle-verdict");
  const probsCollapseWrapper = document.getElementById("probs-collapsible-wrapper");
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
        btnToggleVerdict.title = "Mở rộng toàn bộ xác suất (5 thể loại khác)";
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
      chip.id = `sample-chip-${s.genre.replace(/\s+/g, '_')}`;
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
    showLoading(`Đang phân tích mẫu ${sample.genre}...`, "Trích xuất chuỗi phân đoạn 3-5s & 7 trục âm học...");

    try {
      const res = await fetch("/api/analyze_sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_path: sample.file_path,
          mode: sample.mode
        })
      });

      if (!res.ok) throw new Error("Phân tích mẫu thất bại");
      const data = await res.json();
      if (!data.file_name) data.file_name = sample.title || sample.file_path;
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
    const isVideo = file.type.startsWith("video/") || /\.(mp4|avi|mov|mkv|webm|flv|m4v)$/i.test(file.name);
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
    const isVideo = (mode === "video") || file.type.startsWith("video/") || /\.(mp4|avi|mov|mkv|webm|flv|m4v)$/i.test(file.name);
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
      if (videoFileName) videoFileName.innerHTML = `<i class="fa-solid fa-film"></i> ${file.name}`;
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
      const displayName = (mode === "mic") ? "Bản ghi âm trực tiếp" : file.name;
      if (audioFileName) audioFileName.innerHTML = `<i class="fa-solid fa-file-audio"></i> ${displayName}`;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("mode", targetMode);

    showLoading("Đang tải tệp & xử lý phân đoạn...", `Tệp: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!res.ok) throw new Error("Máy chủ trả về mã lỗi " + res.status);
      const data = await res.json();
      if (!data.file_name) data.file_name = file.name;
      if (localBlobUrl) data._localBlobUrl = localBlobUrl;
      handleAnalysisResult(data);
      showToast(`Đã hoàn tất phân loại: ${data.prediction} (${data.confidence}%)`);
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
    showLoading("Đang nạp video từ YouTube / URL...", "Sử dụng yt-dlp tải luồng & trích xuất phân đoạn 3-5s...");

    try {
      const res = await fetch("/api/analyze_url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url, mode: "video" })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || ("Máy chủ trả về mã lỗi " + res.status));
      }

      const data = await res.json();
      handleAnalysisResult(data);
      showToast(`Đã phân tích thành công: ${data.file_name || data.prediction}`);
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

      const started = await recorder.startRecording((stream) => {
        // Kết nối luồng mic trực tiếp sang Three.js Wave Ribbon
        audioWave.connectMicrophoneStream(stream);
      });

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
        showToast("Bản ghi ngắn hơn 5s. Khuyến nghị ghi 6-15s để đạt độ tin cậy chuẩn!");
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
    currentOverallExplanation = result.explanation;
    currentOverallModalities = result.modalities || null;
    currentRawChunks = result.chunks || [];
    currentMergedSegments = result.merged_segments || [];
    currentTimelineSeries = result.timeline_series || [];

    // 1. Kích hoạt media player và gắn Web Audio API
    setupMediaPlayer(result);

    // 2. Điền thông tin Verdict chính
    renderVerdict(result);

    // 3. Animate 6 thanh xác suất bằng GSAP
    animateProbabilityBars(result.probs, result.prediction);

    // 4. Render lời lý giải tự nhiên XAI
    renderXaiExplanation(result.explanation);

    // 5. Render hiển thị theo từng phương thức tổng thể (Requirement 2)
    if (result.modalities) {
      updateModalitiesGrid(result.modalities, result.prediction, result.confidence, false);
    }

    // 6. Render CBM Concept Pills tương tác
    renderConceptPills(result.active_concepts);

    // 7. Render Triggered Symbolic Rules
    renderTriggeredRules(result.triggered_rules);

    // 8. Render Segment Timeline & Filmstrip (Gộp thông minh, Ribbon & Biểu đồ thời gian)
    renderTimelineAndSegments(result.chunks, result.mode, result.merged_segments);
    renderGenreTimelineRibbon(result.chunks, result.duration_seconds);
    renderTimelineChart(result.timeline_series || result.chunks, result.duration_seconds);

    // 9. Render Ma trận Đối sánh Phân biệt Di sản & Thử nghiệm What-If (Phương án 3)
    if (result.contrastive_analysis) {
      renderContrastiveMatrix(result.contrastive_analysis, result.probs, result.prediction);
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

      const displayName = result.file_name || result.prediction || "Video biểu diễn";
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
          else if (Math.abs(ratio - 4 / 3) < 0.15) label = "4:3 Classic";
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

      const displayName = result.file_name || (result.mode === "mic" ? "Bản ghi âm trực tiếp" : result.prediction) || "Tệp âm thanh";
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

    if (titleEl) {
      titleEl.innerText = result.prediction;
      titleEl.style.color = result.genre_color || "#C84B31";
    }

    if (durEl) {
      durEl.innerText = `${result.duration_seconds}s (${result.mode.toUpperCase()})`;
    }

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
        }
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
          delay: 0.05
        });
        const counter = { val: 0 };
        gsap.to(counter, {
          val: topScore,
          duration: 0.9,
          ease: "power3.out",
          delay: 0.05,
          onUpdate: () => {
            valEl.innerText = `${counter.val.toFixed(1)}%`;
          }
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
            delay: 0.08 + idx * 0.04
          });
          const counter = { val: 0 };
          gsap.to(counter, {
            val: score,
            duration: 0.9,
            ease: "power3.out",
            delay: 0.08 + idx * 0.04,
            onUpdate: () => {
              valEl.innerText = `${counter.val.toFixed(1)}%`;
            }
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
    const citationsEl = document.getElementById("xai-citations-list");

    if (headlineEl) headlineEl.innerText = explanation.headline;
    if (bodyEl && explanation.narrative) {
      let formatted = explanation.narrative
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");
      bodyEl.innerHTML = formatted;
    }

    if (citationsEl && explanation.citations) {
      citationsEl.innerHTML = "";
      explanation.citations.forEach((cit) => {
        const li = document.createElement("li");
        li.className = "citation-item";
        li.innerHTML = `<i class="fa-solid fa-bookmark" style="color: var(--accent-jade); font-size: 10px;"></i> ${cit}`;
        citationsEl.appendChild(li);
      });
    }

    // GSAP Fade-in
    if (window.gsap) {
      gsap.from("#xai-card", {
        opacity: 0,
        y: 10,
        duration: 0.6,
        ease: "power2.out"
      });
    }
  }

  function renderConceptPills(concepts) {
    const audioContainer = document.getElementById("audio-pills-container");
    const visualContainer = document.getElementById("visual-pills-container");
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
      const isAudio = (c.type === "audio");
      if (isAudio) audioCount++;
      else visualCount++;

      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = `concept-pill ${isAudio ? "audio-pill" : "visual-pill"} active`;
      pill.id = `pill-${c.concept_id}`;
      pill.setAttribute("data-concept-id", c.concept_id);
      pill.setAttribute("data-enabled", "true");
      pill.title = `${c.concept_sentence || c.concept_id} (Bấm để bật/tắt thử nghiệm phản sự thực)`;

      const shortAspect = c.aspect ? c.aspect.replace(/_/g, " ") : "Đặc trưng";
      const tsHint = c.timestamps && c.timestamps.length ? `• ${c.timestamps[0]}` : "";

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
      visualGroup.style.display = visualCount > 0 ? "block" : "none";
    }
    if (visualContainer && visualCount === 0) {
      visualContainer.innerHTML = `<span style="font-size: 12px; color: var(--text-tertiary);">Chế độ chỉ có âm thanh (không có dữ liệu hình ảnh).</span>`;
    }

    // Đảm bảo tất cả các pills luôn hiển thị rõ ràng 100% opacity, không bị stuck ẩn
    document.querySelectorAll(".concept-pill").forEach(p => {
      p.style.opacity = "1";
      p.style.display = "inline-flex";
    });
  }

  // =========================================================================
  // 7. CBM CAN THIỆP KHÁI NIỆM TRUNG GIAN (/api/intervene)
  // =========================================================================
  async function toggleConceptIntervention(conceptId, pillEl) {
    if (!currentAnalysis) return;

    const isCurrentlyEnabled = pillEl.getAttribute("data-enabled") === "true";
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

    // 1. Bay camera 3D tới nút concept
    graph3d.flyToConcept(conceptId);

    // 2. Cập nhật trạng thái enabled trong active_concepts
    if (currentAnalysis.active_concepts) {
      const cObj = currentAnalysis.active_concepts.find(c => c.concept_id === conceptId);
      if (cObj) cObj.enabled = newState;
    }

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
          all_concepts: currentAnalysis.active_concepts
        })
      });

      if (!res.ok) throw new Error("Can thiệp thất bại");
      const data = await res.json();

      // Cập nhật lại state xác suất
      currentAnalysis.probs = data.probs;
      currentAnalysis.prediction = data.prediction;
      currentAnalysis.confidence = data.confidence;

      // Animate mượt mà lại 6 thanh xác suất qua GSAP mà không làm nháy trang
      updateProbabilityBarsSmooth(data.probs, data.prediction, data.confidence);

      // Cập nhật lại Bảng Ma trận Đối sánh Phân biệt & What-If
      if (data.contrastive_analysis) {
        renderContrastiveMatrix(data.contrastive_analysis, data.probs, data.prediction);
      }

      showToast(data.explanation_note);
    } catch (e) {
      console.error(e);
      showToast("Lỗi can thiệp CBM: " + e.message);
    }
  }

  function updateProbabilityBarsSmooth(probs, winnerGenre, confidence) {
    const titleEl = document.getElementById("verdict-genre-title");
    const confEl = document.getElementById("confidence-number");

    if (titleEl) {
      titleEl.innerText = winnerGenre;
    }

    if (confEl && window.gsap) {
      const counter = { val: parseFloat(confEl.innerText) || 50 };
      gsap.to(counter, {
        val: confidence,
        duration: 0.6,
        ease: "power2.out",
        onUpdate: () => {
          confEl.innerText = `${counter.val.toFixed(1)}%`;
        }
      });
    } else if (confEl) {
      confEl.innerText = `${confidence.toFixed(1)}%`;
    }

    animateProbabilityBars(probs, winnerGenre);
  }

  // =========================================================================
  // 8. RENDER LUẬT TƯỢNG TRƯNG KÍCH HOẠT (TRIGGERED RULES)
  // =========================================================================
  function renderTriggeredRules(rules) {
    const container = document.getElementById("rules-list-container");
    if (!container) return;

    container.innerHTML = "";
    if (!rules || rules.length === 0) {
      container.innerHTML = "<p style='color: var(--text-secondary); font-size: 12px;'>Không có luật kích hoạt.</p>";
      return;
    }

    rules.forEach((r, idx) => {
      const box = document.createElement("div");
      box.className = "rule-box";

      const confPct = `${(r.confidence * (r.confidence <= 1.0 ? 100 : 1)).toFixed(1)}%`;

      box.innerHTML = `
        <div class="rule-header-line">
          <span style="font-weight: 600; font-family: var(--font-mono); font-size: 11px; color: var(--accent-vermilion);">
            [LUẬT ${idx + 1}] ➜ ${r.consequent}
          </span>
          <div class="rule-metrics-badges">
            <span class="rule-metric-badge">Tin cậy: ${confPct}</span>
            <span class="rule-metric-badge">Mẫu: ${r.support}</span>
          </div>
        </div>
        <div class="rule-clause-code">IF ${r.rule_text} THEN ${r.consequent}</div>
        <div class="rule-vi-explanation">${r.summary_text || ""}</div>
      `;

      container.appendChild(box);
    });
  }

  // =========================================================================
  // 9. TIMELINE PHÂN ĐOẠN (CHUNKS 3-5s), GỘP ĐOẠN THÔNG MINH & BIỂU ĐỒ DIỄN BIẾN
  // =========================================================================

  function renderTimelineAndSegments(chunks, mode, mergedSegments) {
    currentRawChunks = chunks || [];
    currentMergedSegments = (mergedSegments && mergedSegments.length > 0) ? mergedSegments : currentRawChunks;

    const modeBar = document.getElementById("timeline-mode-bar");
    if (modeBar) {
      modeBar.style.display = (currentTimelineSubtab === "filmstrip") ? "inline-flex" : "none";
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
    const btnReturnOverall = document.getElementById("btn-return-overall-xai");
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
    const isMerged = (currentTimelineMode === "merged" && currentMergedSegments.length > 0);
    const items = isMerged ? currentMergedSegments : currentRawChunks;

    if (countBadge) {
      if (isMerged) {
        countBadge.innerHTML = `<i class="fa-solid fa-layer-group"></i> ${items.length} vùng gộp (${currentRawChunks.length} chunk 4s)`;
      } else {
        countBadge.innerHTML = `<i class="fa-solid fa-bars-staggered"></i> ${items.length} chunk (4s)`;
      }
    }

    if (!items || items.length === 0) {
      strip.innerHTML = "<p style='color: var(--text-secondary); font-size: 12px; grid-column: 1 / -1; padding: 12px 0;'>Không có dữ liệu phân đoạn.</p>";
      return;
    }

    items.forEach((item, idx) => {
      const cardId = (isMerged ? "merged-seg-" : "raw-chunk-") + idx;
      const isActive = (activeSelectedSegmentId === cardId);

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
      const hasVis = item.modalities && item.modalities.visual && item.modalities.visual.prediction;

      if (hasAud) {
        modChipsHtml += `<span class="mod-pill mod-audio" title="Âm thanh: ${item.modalities.audio.prediction} (${item.modalities.audio.confidence}%)"><i class="fa-solid fa-volume-high"></i> ${item.modalities.audio.prediction}</span>`;
      }
      if (hasVis) {
        modChipsHtml += `<span class="mod-pill mod-visual" title="Thị giác: ${item.modalities.visual.prediction} (${item.modalities.visual.confidence}%)"><i class="fa-solid fa-eye"></i> ${item.modalities.visual.prediction}</span>`;
      }
      modChipsHtml += `<span class="mod-pill mod-fused" title="Hợp nhất: ${item.prediction} (${item.confidence}%)"><i class="fa-solid fa-scale-balanced"></i> ${item.prediction}</span>`;

      // Cues phụ
      const cuesText = (item.visual_cues && item.visual_cues.length)
        ? item.visual_cues.join(", ")
        : (item.crisp_axes ? `Cao độ: ${item.crisp_axes.Pitch || 'Vừa'} • Năng lượng: ${item.crisp_axes.Energy || 'Vừa'}` : "");

      seg.innerHTML = `
        <div class="segment-thumb">
          ${thumbImgSrc ? `<img src="${thumbImgSrc}" alt="Segment ${idx}" />` : `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-tertiary);"><i class="fa-solid fa-music"></i></div>`}
          <span class="segment-time-badge">${item.timestamp}</span>
          ${mergedTag}
        </div>
        <div class="segment-meta">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="segment-genre" style="color: ${item.genre_color || 'var(--accent-vermilion)'}; font-weight: 700;">${item.prediction}</span>
            <span class="segment-score">${item.confidence}% tin cậy</span>
          </div>
          <div class="segment-modality-tags">${modChipsHtml}</div>
          <span class="segment-tag" title="${cuesText}">${cuesText}</span>
        </div>
      `;

      // Bấm vào thẻ: Làm nổi bật viền, tua video và cập nhật diễn giải (Requirement 1)
      seg.addEventListener("click", () => {
        activeSelectedSegmentId = cardId;
        document.querySelectorAll(".segment-item").forEach(s => s.classList.remove("active"));
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
        headlineEl.innerText = item.explanation.headline || `Diễn giải phân đoạn ${item.timestamp}`;
      }
      if (bodyEl && item.explanation.narrative) {
        let formatted = item.explanation.narrative
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
          .replace(/\n/g, "<br>");
        bodyEl.innerHTML = formatted;
      }
    }

    // Hiển thị kết quả theo từng phương thức của phân đoạn này (Requirement 2)
    updateModalitiesGrid(item.modalities, item.prediction, item.confidence, true);
  }

  // Khôi phục lại lời lý giải toàn cảnh của cả video
  function restoreOverallExplanation() {
    activeSelectedSegmentId = null;
    document.querySelectorAll(".segment-item").forEach(s => s.classList.remove("active"));

    const banner = document.getElementById("xai-segment-banner");
    if (banner) banner.style.display = "none";

    if (currentOverallExplanation) {
      renderXaiExplanation(currentOverallExplanation);
    }

    if (currentOverallModalities && currentAnalysis) {
      updateModalitiesGrid(currentOverallModalities, currentAnalysis.prediction, currentAnalysis.confidence, false);
    }

    if (currentRawChunks && currentRawChunks[0]) {
      renderMpeg7RadarBars(currentRawChunks[0]);
    }

    document.querySelectorAll(".concept-pill").forEach(p => {
      p.classList.remove("chunk-active");
    });
  }

  // Hiển thị kết quả theo từng phương thức (Requirement 2)
  function updateModalitiesGrid(modalities, fallbackGenre, fallbackConfidence, isSegmentView) {
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
        const aNote = modalities.audio.notes || "Mô hình CNN (EXP-13) + Rừng ngẫu nhiên liên tục";
        modAudio.innerHTML = `<strong>${modalities.audio.prediction}</strong> <span style="color: var(--accent-vermilion); font-family: var(--font-mono); font-weight: 700;">${modalities.audio.confidence}%</span><br><span style="font-size: 11px; color: var(--text-secondary);">${aNote}</span>`;
      } else {
        modAudio.innerHTML = `<strong>${fallbackGenre}</strong> <span style="color: var(--accent-vermilion); font-family: var(--font-mono); font-weight: 700;">${fallbackConfidence}%</span><br><span style="font-size: 11px; color: var(--text-secondary);">Mô hình âm thanh</span>`;
      }
    }

    // 2. Nhánh Thị giác: Chỉ hiển thị khi có khả năng tính toán
    if (visualWrap && modVisual) {
      if (modalities && modalities.visual && modalities.visual.prediction) {
        visualWrap.style.display = "block";
        const vCues = (modalities.visual.cues && modalities.visual.cues.length)
          ? modalities.visual.cues.join(", ")
          : (modalities.visual.notes || "Nhận diện nhạc cụ, trang phục & đạo cụ");
        modVisual.innerHTML = `<strong>${modalities.visual.prediction}</strong> <span style="color: var(--accent-amber); font-family: var(--font-mono); font-weight: 700;">${modalities.visual.confidence}%</span><br><span style="font-size: 11px; color: var(--text-secondary);">${vCues}</span>`;
      } else {
        visualWrap.style.display = "none";
      }
    }

    // 3. Kết quả Hợp nhất
    if (modFused) {
      const fPred = (modalities && modalities.fused && modalities.fused.prediction) ? modalities.fused.prediction : fallbackGenre;
      const fConf = (modalities && modalities.fused && modalities.fused.confidence) ? modalities.fused.confidence : fallbackConfidence;
      const fMethod = (modalities && modalities.fused && modalities.fused.method) ? modalities.fused.method : "Late Fusion Bayesian + Suy luận Mệnh đề Luật Mờ";
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
          videoContainer.scrollIntoView({ behavior: "auto", block: "nearest" });
        }
      }
      if ('fastSeek' in videoPlayer) {
        try { videoPlayer.fastSeek(sec); } catch (e) { videoPlayer.currentTime = sec; }
      } else {
        videoPlayer.currentTime = sec;
      }
      videoPlayer.playbackRate = 1.0;
      const playPromise = videoPlayer.play();
      if (playPromise !== undefined) {
        playPromise.catch(e => console.warn("Video play interrupted:", e));
      }
    } else if (activeMediaType === "audio" && audioPlayer && audioPlayer.src) {
      if (currentActiveTab !== "media" && currentActiveTab !== "mic") switchTab("media");
      if (audioContainer) {
        const rect = audioContainer.getBoundingClientRect();
        if (rect.bottom < 50 || rect.top > window.innerHeight - 50) {
          audioContainer.scrollIntoView({ behavior: "auto", block: "nearest" });
        }
      }
      if ('fastSeek' in audioPlayer) {
        try { audioPlayer.fastSeek(sec); } catch (e) { audioPlayer.currentTime = sec; }
      } else {
        audioPlayer.currentTime = sec;
      }
      audioPlayer.playbackRate = 1.0;
      const playPromise = audioPlayer.play();
      if (playPromise !== undefined) {
        playPromise.catch(e => console.warn("Audio play interrupted:", e));
      }
    }
  }

  function highlightChunkConcepts(chunk) {
    const activeIds = [...(chunk.active_audio_concepts || []), ...(chunk.active_visual_concepts || [])];
    document.querySelectorAll(".concept-pill").forEach(p => {
      const cid = p.getAttribute("data-concept-id");
      p.style.opacity = "1";
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
    const axesOrder = ["Energy", "Harmony", "Pitch", "Spectral", "Tempo", "Timbre", "Vibrato"];

    axesOrder.forEach(ax => {
      const val = (chunk.radar && chunk.radar[ax] !== undefined) ? chunk.radar[ax] : 0.5;
      const crispVal = (chunk.crisp_axes && chunk.crisp_axes[ax]) ? chunk.crisp_axes[ax] : "Vừa";

      const row = document.createElement("div");
      row.className = "axis-bar-row";
      row.innerHTML = `
        <span class="axis-name">${ax}</span>
        <div class="axis-progress-track">
          <div class="axis-progress-fill" style="width: ${val * 100}%;"></div>
        </div>
        <span class="axis-crisp-badge">${crispVal}</span>
        <span class="axis-val-num">${(val * 100).toFixed(0)}%</span>
      `;
      radarList.appendChild(row);
    });
  }

  // Chọn phân đoạn tương ứng với mốc thời gian và làm nổi bật
  function selectSegmentAtTime(targetSec) {
    const isMerged = (currentTimelineMode === "merged" && currentMergedSegments.length > 0);
    const items = isMerged ? currentMergedSegments : currentRawChunks;
    if (!items || items.length === 0) return;

    let matchedIdx = items.findIndex(item => targetSec >= item.start_time && targetSec <= item.end_time);
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
      const cardId = (isMerged ? "merged-seg-" : "raw-chunk-") + matchedIdx;
      activeSelectedSegmentId = cardId;
      document.querySelectorAll(".segment-item").forEach(s => s.classList.remove("active"));
      const card = document.getElementById(cardId);
      if (card) {
        card.classList.add("active");
        card.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
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
    const ribbonWrap = document.getElementById("genre-timeline-ribbon-wrap");
    const ribbon = document.getElementById("genre-timeline-ribbon");
    const ribbonPlayhead = document.getElementById("ribbon-playhead");
    if (!ribbonWrap || !ribbon || !chunks || chunks.length === 0) return;

    ribbonWrap.style.display = "block";
    ribbon.innerHTML = "";

    const activePlayer = (activeMediaType === "video" && videoPlayer && videoPlayer.src) ? videoPlayer : audioPlayer;
    const totalDur = (activePlayer && activePlayer.duration > 0) ? activePlayer.duration : (totalDuration || chunks[chunks.length - 1].end_time || 1);

    chunks.forEach((chk, idx) => {
      const segDur = chk.duration || (chk.end_time - chk.start_time) || 4;
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
      const curPlayer = (activeMediaType === "video" && videoPlayer && videoPlayer.src) ? videoPlayer : audioPlayer;
      const realDur = (curPlayer && curPlayer.duration > 0) ? curPlayer.duration : totalDur;
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
    const legendContainer = document.getElementById("chart-legend-container");
    const svg = document.getElementById("timeline-svg-chart");
    if (!chartView || !svg) return;

    const series = (timelineSeries && timelineSeries.length > 0) ? timelineSeries : currentRawChunks;
    if (!series || series.length === 0) return;

    const activePlayer = (activeMediaType === "video" && videoPlayer && videoPlayer.src) ? videoPlayer : audioPlayer;
    const totalDur = (activePlayer && activePlayer.duration > 0) ? activePlayer.duration : (totalDuration || series[series.length - 1].end_time || (series.length * 4));

    // 1. Tìm các thể loại có xác suất đáng kể trong chuỗi thời gian để tạo Legend
    const activeGenres = new Set();
    series.forEach(pt => {
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
      Array.from(activeGenres).forEach(g => {
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
    yLevels.forEach(pct => {
      const y = pad.top + plotH - (pct / 100) * plotH;
      const isBase = (pct === 0);
      svgInner += `
        <line x1="${pad.left}" y1="${y}" x2="${W - pad.right}" y2="${y}" 
              stroke="${isBase ? 'rgba(0,0,0,0.22)' : 'rgba(0,0,0,0.07)'}" 
              stroke-width="${isBase ? '1.5' : '1'}" 
              stroke-dasharray="${isBase ? '' : '3,3'}" />
        <text x="${pad.left - 8}" y="${y + 4}" fill="#777" font-size="10" font-family="monospace" text-anchor="end">${pct}%</text>
      `;
    });

    // 2. Trục X & Các mốc thời gian (Ticks dạng số nguyên, đơn vị (s) ở cuối trục)
    const step = dur <= 25 ? 5 : (dur <= 60 ? 10 : 20);
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
    activeGenres.forEach(genre => {
      const col = GENRE_COLORS[genre] || "#888";

      // Chuẩn hóa xác suất sang thang 0..100
      const rawPts = series.map(pt => {
        const t = (pt.time !== undefined) ? pt.time : ((pt.start_time + pt.end_time) / 2);
        let pVal = 0;
        if (pt.probs && pt.probs[genre] !== undefined) {
          pVal = pt.probs[genre];
        } else if (pt.prediction === genre) {
          pVal = pt.confidence || 60;
        }
        if (pVal <= 1.0) pVal = pVal * 100;
        return { time: t, prob: Math.max(0, Math.min(100, pVal)), item: pt };
      });

      if (rawPts.length === 0) return;

      // Thêm điểm neo ở t = 0 và t = dur để đường cong bao phủ toàn bộ thời lượng
      const pts = [];
      pts.push({
        x: pad.left,
        y: pad.top + plotH - (rawPts[0].prob / 100) * plotH,
        prob: rawPts[0].prob,
        time: 0
      });

      rawPts.forEach(p => {
        pts.push({
          x: pad.left + (Math.max(0, Math.min(dur, p.time)) / dur) * plotW,
          y: pad.top + plotH - (p.prob / 100) * plotH,
          prob: p.prob,
          time: p.time
        });
      });

      const lastRaw = rawPts[rawPts.length - 1];
      pts.push({
        x: pad.left + plotW,
        y: pad.top + plotH - (lastRaw.prob / 100) * plotH,
        prob: lastRaw.prob,
        time: dur
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
      const t = (pt.time !== undefined) ? pt.time : ((pt.start_time + pt.end_time) / 2);
      const x = pad.left + (Math.max(0, Math.min(dur, t)) / dur) * plotW;
      const topGenre = pt.prediction;
      let topProb = (pt.probs && pt.probs[topGenre] !== undefined)
        ? pt.probs[topGenre]
        : (pt.confidence || 60);
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
    points.forEach(pt => {
      const idx = parseInt(pt.getAttribute("data-chunk-idx"), 10);
      const chk = series[idx];
      if (!chk) return;

      pt.addEventListener("mouseenter", (e) => {
        pt.setAttribute("r", "8");
        if (tooltip) {
          const topG = chk.prediction;
          let conf = chk.confidence;
          if (conf <= 1.0) conf = conf * 100;
          const cues = (chk.visual_cues && chk.visual_cues.length) ? chk.visual_cues.join(", ") : "";

          // Liệt kê xác suất các lớp
          let probsListHtml = "";
          if (chk.probs) {
            const sorted = Object.entries(chk.probs)
              .map(([g, p]) => [g, p <= 1.0 ? p * 100 : p])
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3);
            probsListHtml = sorted.map(([g, p]) => `
              <div style="display: flex; justify-content: space-between; gap: 8px; font-size: 10px; margin-top: 2px;">
                <span style="color: ${GENRE_COLORS[g] || '#666'}">${g}:</span>
                <span style="font-weight: 700;">${p.toFixed(1)}%</span>
              </div>
            `).join("");
          }

          tooltip.innerHTML = `
            <div style="font-weight: 700; color: ${GENRE_COLORS[topG] || '#C84B31'}; margin-bottom: 3px; font-size: 11.5px;">
              ${topG} (${conf.toFixed(1)}%)
            </div>
            <div style="color: var(--text-secondary); font-size: 10px; margin-bottom: 4px;">
              Mốc: ${chk.timestamp}
            </div>
            ${probsListHtml}
            ${cues ? `<div style="color: var(--accent-amber); font-size: 10px; margin-top: 4px; border-top: 1px dashed rgba(0,0,0,0.1); padding-top: 2px;">Vật thể: ${cues}</div>` : ''}
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
        const nodeTime = (chk.time !== undefined) ? chk.time : ((chk.start_time + chk.end_time) / 2);
        seekToTimestamp(nodeTime);
        selectSegmentAtTime(nodeTime);

        // Đồng bộ dải ribbon playhead và chart scrub line tới đúng mốc thời gian của node
        const ribbonPlayhead = document.getElementById("ribbon-playhead");
        const activePlayer = (activeMediaType === "video" && videoPlayer && videoPlayer.src) ? videoPlayer : audioPlayer;
        const dur = (activePlayer && activePlayer.duration > 0) ? activePlayer.duration : totalDuration;
        if (ribbonPlayhead && dur > 0) {
          ribbonPlayhead.style.left = `${(nodeTime / dur) * 100}%`;
        }
        if (scrubLine && dur > 0) {
          const chartPct = (pad.left + (nodeTime / dur) * plotW) / 600 * 100;
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
      } else if (viewX >= (600 - pad.right)) {
        targetSec = totalDuration;
      } else {
        const timeFraction = (viewX - pad.left) / plotW;
        targetSec = Math.max(0, Math.min(totalDuration, timeFraction * totalDuration));
      }

      seekToTimestamp(targetSec);
      selectSegmentAtTime(targetSec);

      const ribbonPlayhead = document.getElementById("ribbon-playhead");
      if (ribbonPlayhead) {
        const activePlayer = (activeMediaType === "video" && videoPlayer && videoPlayer.src) ? videoPlayer : audioPlayer;
        const dur = (activePlayer && activePlayer.duration > 0) ? activePlayer.duration : totalDuration;
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
      const dur = (player.duration && player.duration > 0) ? player.duration : (totalDuration || 1);
      if (!dur || dur <= 0) return;

      const cur = player.currentTime;
      const pct = Math.min(100, Math.max(0, (cur / dur) * 100));

      if (ribbonPlayhead) {
        ribbonPlayhead.style.left = `${pct}%`;
      }

      if (chartScrubLine) {
        // Tọa độ biểu đồ: pad.left = 45, plotW = 525, tổng W = 600
        const chartPct = (45 + (pct / 100) * 525) / 600 * 100;
        chartScrubLine.style.left = `${chartPct}%`;
        chartScrubLine.style.display = "block";
      }
    };

    player.addEventListener("timeupdate", playbackSyncHandler);
  }

  const SUBTAB_METAS = {
    "filmstrip": { icon: "fa-solid fa-film", label: "Dải Keyframes" },
    "chart": { icon: "fa-solid fa-chart-line", label: "Biểu đồ Thời gian" },
    "spectrogram": { icon: "fa-solid fa-wave-square", label: "Ảnh Phổ Mel" },
    "radar": { icon: "fa-solid fa-sliders", label: "7 Trục MPEG-7" }
  };

  // Hàm chuyển đổi chế độ xem phân đoạn tập trung (Keyframes, Biểu đồ, Ảnh phổ, 7 trục)
  function switchTimelineSubtab(target) {
    currentTimelineSubtab = target;
    const filmstripView = document.getElementById("timeline-scroll-strip");
    const chartView = document.getElementById("timeline-chart-container");
    const specView = document.getElementById("spectrogram-grid-container");
    const radarView = document.getElementById("radar-bars-container");
    const modeBar = document.getElementById("timeline-mode-bar");

    // Cập nhật custom dropdown trigger
    const triggerIcon = document.getElementById("subtab-trigger-icon");
    const triggerText = document.getElementById("subtab-trigger-text");
    const triggerBtn = document.getElementById("custom-subtab-trigger");
    const menu = document.getElementById("custom-subtab-menu");

    if (SUBTAB_METAS[target]) {
      if (triggerIcon) triggerIcon.innerHTML = `<i class="${SUBTAB_METAS[target].icon}"></i>`;
      if (triggerText) triggerText.textContent = SUBTAB_METAS[target].label;
    }

    // Đóng dropdown menu nếu đang mở
    if (triggerBtn) {
      triggerBtn.classList.remove("open");
      triggerBtn.setAttribute("aria-expanded", "false");
    }
    if (menu) menu.classList.remove("show");

    // Cập nhật active class trong custom dropdown menu
    document.querySelectorAll(".custom-dropdown-item").forEach(item => {
      item.classList.toggle("active", item.getAttribute("data-subtab") === target);
    });

    // Đồng bộ nút tab nếu giao diện còn dùng
    document.querySelectorAll(".subtab-btn").forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-subtab") === target);
    });

    if (filmstripView) filmstripView.style.display = (target === "filmstrip") ? "grid" : "none";
    if (chartView) {
      chartView.style.display = (target === "chart") ? "flex" : "none";
      if (target === "chart" && currentTimelineSeries && currentTimelineSeries.length > 0) {
        renderTimelineChart(currentTimelineSeries, (currentAnalysis ? currentAnalysis.duration_seconds : 0));
      }
    }
    if (specView) specView.style.display = (target === "spectrogram") ? "grid" : "none";
    if (radarView) radarView.style.display = (target === "radar") ? "flex" : "none";

    // Thanh điều khiển Gộp vùng tương đồng / Từng chunk CHỈ hiển thị ở giữa khi ở tab Dải Keyframes
    if (modeBar) {
      const hasData = (currentRawChunks && currentRawChunks.length > 0);
      modeBar.style.display = (hasData && target === "filmstrip") ? "inline-flex" : "none";
    }
  }

  // Chuyển đổi giữa 4 Subtab: Dải Keyframes, Biểu đồ Thời gian, Ảnh Phổ Mel, 7 Trục MPEG-7
  function setupTimelineSubtabs() {
    const triggerBtn = document.getElementById("custom-subtab-trigger");
    const menu = document.getElementById("custom-subtab-menu");
    const dropdownWrap = document.getElementById("custom-subtab-dropdown");

    if (triggerBtn && menu) {
      triggerBtn.onclick = (e) => {
        e.stopPropagation();
        const isOpen = menu.classList.contains("show");
        if (isOpen) {
          menu.classList.remove("show");
          triggerBtn.classList.remove("open");
          triggerBtn.setAttribute("aria-expanded", "false");
        } else {
          menu.classList.add("show");
          triggerBtn.classList.add("open");
          triggerBtn.setAttribute("aria-expanded", "true");
        }
      };

      // Đóng khi click ra ngoài dropdown
      document.addEventListener("click", (e) => {
        if (dropdownWrap && !dropdownWrap.contains(e.target)) {
          menu.classList.remove("show");
          triggerBtn.classList.remove("open");
          triggerBtn.setAttribute("aria-expanded", "false");
        }
      });

      // Đóng khi nhấn Escape
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && menu.classList.contains("show")) {
          menu.classList.remove("show");
          triggerBtn.classList.remove("open");
          triggerBtn.setAttribute("aria-expanded", "false");
        }
      });
    }

    // Gắn sự kiện click cho các item trong custom dropdown
    document.querySelectorAll(".custom-dropdown-item").forEach(item => {
      item.onclick = (e) => {
        e.stopPropagation();
        const target = item.getAttribute("data-subtab");
        switchTimelineSubtab(target);
      };
    });

    const subtabs = document.querySelectorAll(".subtab-btn");
    subtabs.forEach(btn => {
      btn.onclick = () => {
        const target = btn.getAttribute("data-subtab");
        switchTimelineSubtab(target);
      };
    });
  }

  // =========================================================================
  // 10. TIỆN ÍCH GIAO DIỆN: LOADING OVERLAY & TOAST NOTIFICATION
  // =========================================================================
  function showLoading(title, subtitle) {
    if (loadingOverlay) {
      const h3 = loadingOverlay.querySelector("h3");
      const p = loadingOverlay.querySelector("p");
      if (h3) h3.innerText = title;
      if (p) p.innerText = subtitle;
      loadingOverlay.style.display = "flex";
    }
  }

  function hideLoading() {
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
  // 11. MA TRẬN ĐỐI SÁNH PHÂN BIỆT DI SẢN & THỬ NGHIỆM WHAT-IF (PHƯƠNG ÁN 3)
  // =========================================================================
  let whatIfState = {
    top1Genre: "",
    top2Genre: "",
    top1BaseProb: 50,
    top2BaseProb: 20
  };

  function renderContrastiveMatrix(contrastive, probs, currentPrediction) {
    if (!contrastive) return;

    const top1 = contrastive.top1_genre || currentPrediction;
    const top2 = contrastive.top2_genre || "Quan họ";
    const top1Prob = contrastive.top1_prob || 60;
    const top2Prob = contrastive.top2_prob || 20;
    const margin = contrastive.margin_pct || (top1Prob - top2Prob);

    whatIfState = {
      top1Genre: top1,
      top2Genre: top2,
      top1BaseProb: top1Prob,
      top2BaseProb: top2Prob
    };

    // Header metadata tags
    const winnerTag = document.getElementById("meta-winner-tag");
    const runnerTag = document.getElementById("meta-runnerup-tag");
    if (winnerTag) winnerTag.innerText = `Top 1: ${top1} (${top1Prob}%)`;
    if (runnerTag) runnerTag.innerText = `Top 2: ${top2} (${top2Prob}%)`;

    // Table Column Headers
    const th1Name = document.getElementById("th-top1-name");
    const th1Prob = document.getElementById("th-top1-prob");
    const th2Name = document.getElementById("th-top2-name");
    const th2Prob = document.getElementById("th-top2-prob");

    if (th1Name) th1Name.innerText = top1;
    if (th1Prob) th1Prob.innerText = `${top1Prob}%`;
    if (th2Name) th2Name.innerText = top2;
    if (th2Prob) th2Prob.innerText = `${top2Prob}%`;

    // Render 5 Comparative Rows
    const tbody = document.getElementById("contrastive-table-body");
    if (tbody && contrastive.dimensions) {
      tbody.innerHTML = "";
      contrastive.dimensions.forEach((dim, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="dim-header-cell">
            <div class="contrastive-dim-title">
              <i class="fa-solid ${dim.icon || 'fa-check'}"></i>
              <span>${dim.name}</span>
            </div>
            <div style="font-size: 11px; font-family: var(--font-mono); color: var(--text-tertiary);">Tiêu chí ${idx + 1}/5</div>
          </td>
          <td class="cell-winner">
            <div class="cell-desc-text">${dim.top1_desc}</div>
            <div class="cell-evidence-badge favors-top1">
              <i class="fa-solid fa-circle-check"></i> ${dim.evidence}
            </div>
          </td>
          <td class="cell-runner">
            <div class="cell-desc-text">${dim.top2_desc}</div>
            <div class="cell-evidence-badge">
              <i class="fa-solid fa-arrow-right-arrow-left"></i> Đối trọng cạnh tranh
            </div>
          </td>
          <td class="decision-cue-cell">
            <span class="winner-seal-badge">
              <i class="fa-solid fa-trophy"></i> ${top1}
            </span>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Render What-If Sandbox
    const whatIf = contrastive.what_if;
    const wName = document.getElementById("whatif-winner-name");
    const rName = document.getElementById("whatif-runner-name");
    const condGrid = document.getElementById("whatif-conditions-grid");
    const slider = document.getElementById("whatif-intervene-slider");
    const sliderVal = document.getElementById("whatif-slider-val");
    const simResult = document.getElementById("whatif-sim-result");

    if (wName) wName.innerText = top1;
    if (rName) rName.innerText = top2;

    if (condGrid && whatIf && whatIf.flip_conditions) {
      condGrid.innerHTML = "";
      whatIf.flip_conditions.forEach((cond, i) => {
        const cCard = document.createElement("div");
        cCard.className = "whatif-cond-card";
        cCard.innerHTML = `
          <span class="cond-step-num">${i + 1}</span>
          <span class="cond-text">${cond}</span>
        `;
        condGrid.appendChild(cCard);
      });
    }

    // Reset slider to 50%
    if (slider) {
      slider.value = 50;
      if (sliderVal) {
        sliderVal.innerText = `50% (Cân bằng thực tế)`;
        sliderVal.style.color = "var(--text-secondary)";
      }
      if (simResult) {
        simResult.innerHTML = `
          <i class="fa-solid fa-circle-info" style="color: var(--accent-indigo);"></i>
          <span>Thực tế: <strong>${top1}</strong> dẫn trước <strong>${top2}</strong> với chênh lệch an toàn <strong>+${margin}%</strong>. Kéo thanh trượt sang phải để mô phỏng lật ngược tình thế.</span>
        `;
        simResult.style.backgroundColor = "#FAF9F6";
        simResult.style.borderColor = "var(--border-hairline)";
      }
    }
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
      let simP1 = Math.max(2.0, Math.min(96.0, whatIfState.top1BaseProb - delta));
      let simP2 = Math.max(2.0, Math.min(96.0, whatIfState.top2BaseProb + delta));

      const sumSim = simP1 + simP2;
      const basePool = whatIfState.top1BaseProb + whatIfState.top2BaseProb;
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
          statusBadge.style.backgroundColor = "var(--accent-vermilion-subtle)";
          statusBadge.style.color = "var(--accent-vermilion)";
        }
        if (simResult) {
          simResult.innerHTML = `
            <i class="fa-solid fa-triangle-exclamation" style="color: var(--accent-vermilion); font-size: 14px;"></i>
            <span><strong>KỊCH BẢN ĐẢO CHIỀU:</strong> Khi tăng cường độ đặc trưng phân cực của ${top2} lên ${val}%, <strong>${top2} (${p2Norm}%)</strong> đã chính thức vượt qua <strong>${top1} (${p1Norm}%)</strong> và trở thành dự đoán hàng đầu!</span>
          `;
          simResult.style.backgroundColor = "var(--accent-vermilion-subtle)";
          simResult.style.borderColor = "rgba(200, 75, 49, 0.3)";
        }
      } else {
        if (statusBadge) {
          statusBadge.innerText = "Kịch bản Phản Sự Thực";
          statusBadge.style.backgroundColor = "rgba(30, 107, 91, 0.12)";
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
});

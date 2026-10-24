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

  // State quản lý kết quả hiện tại & Nguồn đầu vào duy nhất (Single Active Input)
  let currentAnalysis = null;
  let currentActiveTab = "video"; // "video" | "audio" | "mic" | "url"
  let activeMediaType = null;     // "video" | "audio" | null (Chỉ cho phép 1 input tồn tại tại 1 thời điểm)

  // Tham chiếu các phần tử DOM chính
  const videoInput = document.getElementById("video-file-input");
  const audioInput = document.getElementById("audio-file-input");
  const videoPlayer = document.getElementById("main-video-player");
  const audioPlayer = document.getElementById("main-audio-player");
  const videoContainer = document.getElementById("video-player-container");
  const audioContainer = document.getElementById("audio-player-container");
  const videoDropzone = document.getElementById("video-dropzone");
  const audioDropzone = document.getElementById("audio-dropzone");
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
      if (videoDropzone) videoDropzone.style.display = "block";
      if (videoInput) videoInput.value = "";
      if (videoFileName) videoFileName.innerHTML = "";
    }

    if (keepType !== "audio") {
      if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.removeAttribute("src");
        audioPlayer.load();
      }
      if (audioContainer) audioContainer.style.display = "none";
      if (audioDropzone) audioDropzone.style.display = "block";
      if (audioInput) audioInput.value = "";
      if (audioFileName) audioFileName.innerHTML = "";
    }

    // Nếu đang ghi âm từ microphone mà nạp input khác -> dừng thu âm
    if (recorder && recorder.isRecording) {
      if (micStopBtn) micStopBtn.click();
    }

    audioWave.disconnect();
    activeMediaType = keepType;
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

  // Sự kiện nút "Thay đổi video" -> Xóa input hiện tại và mở lại dropzone
  if (btnChangeVideo) {
    btnChangeVideo.addEventListener("click", () => {
      clearAllInputsExcept(null);
    });
  }

  // Sự kiện nút "Thay đổi âm thanh" -> Xóa input hiện tại và mở lại dropzone
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

        // Tự động phân tích mẫu đầu tiên (Ca trù) để người dùng chiêm ngưỡng kết quả ngay khi mở trang
        setTimeout(() => {
          if (!currentAnalysis && data.samples.length > 0) {
            analyzeSampleFile(data.samples[0]);
          }
        }, 500);
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
  // 4. XỬ LÝ UPLOAD TỆP VIDEO / AUDIO (DRAG & DROP)
  // =========================================================================
  setupDropzone("video-dropzone", videoInput, (file) => uploadAndAnalyze(file, "video"));
  setupDropzone("audio-dropzone", audioInput, (file) => uploadAndAnalyze(file, "audio"));

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
    const targetMode = (mode === "video") ? "video" : "audio";
    clearAllInputsExcept(targetMode);

    if (targetMode === "video") {
      switchTab("video");
      localBlobUrl = URL.createObjectURL(file);
      videoPlayer.src = localBlobUrl;
      videoPlayer.load();
      if (videoDropzone) videoDropzone.style.display = "none";
      if (videoContainer) videoContainer.style.display = "block";
      if (videoFileName) videoFileName.innerHTML = `<i class="fa-solid fa-film"></i> ${file.name}`;
    } else {
      switchTab("audio");
      localBlobUrl = URL.createObjectURL(file);
      audioPlayer.src = localBlobUrl;
      audioPlayer.load();
      if (audioDropzone) audioDropzone.style.display = "none";
      if (audioContainer) audioContainer.style.display = "block";
      const displayName = (mode === "mic") ? "Bản ghi âm trực tiếp" : file.name;
      if (audioFileName) audioFileName.innerHTML = `<i class="fa-solid fa-file-audio"></i> ${displayName}`;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("mode", mode);

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

    // 1. Kích hoạt media player và gắn Web Audio API
    setupMediaPlayer(result);

    // 2. Điền thông tin Verdict chính
    renderVerdict(result);

    // 3. Animate 6 thanh xác suất bằng GSAP
    animateProbabilityBars(result.probs, result.prediction);

    // 4. Render lời lý giải tự nhiên XAI
    renderXaiExplanation(result.explanation);

    // 5. Render CBM Concept Pills tương tác
    renderConceptPills(result.active_concepts);

    // 6. Render Triggered Symbolic Rules
    renderTriggeredRules(result.triggered_rules);

    // 7. Render Segment Timeline & Filmstrip
    renderTimelineAndSegments(result.chunks, result.mode);

    // 8. Render 3D Semantic Graph
    if (result.graph_3d) {
      graph3d.loadGraphData(result.graph_3d);
    }
  }

  function setupMediaPlayer(result) {
    if (!result.media_url && !result._localBlobUrl) return;

    if (result.mode === "video") {
      clearAllInputsExcept("video");
      switchTab("video");

      // Ẩn dropzone kéo thả, chỉ hiển thị trình phát video đã nạp
      if (videoDropzone) videoDropzone.style.display = "none";
      if (videoContainer) videoContainer.style.display = "block";

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
    } else {
      clearAllInputsExcept("audio");
      switchTab("audio");

      // Ẩn dropzone kéo thả, chỉ hiển thị trình phát audio đã nạp
      if (audioDropzone) audioDropzone.style.display = "none";
      if (audioContainer) audioContainer.style.display = "block";

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

    if (!concepts || concepts.length === 0) return;

    let hasVisual = false;

    concepts.forEach((c) => {
      const isAudio = (c.type === "audio");
      if (!isAudio) hasVisual = true;

      const pill = document.createElement("button");
      pill.className = `concept-pill ${isAudio ? "audio-pill" : "visual-pill"} active`;
      pill.id = `pill-${c.concept_id}`;
      pill.setAttribute("data-concept-id", c.concept_id);
      pill.setAttribute("data-enabled", "true");

      const shortAspect = c.aspect ? c.aspect.replace(/_/g, " ") : "Đặc trưng";
      const tsHint = c.timestamps && c.timestamps.length ? `• ${c.timestamps[0]}` : "";

      pill.innerHTML = `
        <span class="pill-check">✓</span>
        <span style="font-weight: 600;">${c.concept_id}</span>
        <span>${shortAspect}</span>
        <span class="pill-pct">${tsHint}</span>
      `;

      // Click: Can thiệp CBM & Fly-to 3D
      pill.addEventListener("click", () => {
        toggleConceptIntervention(c.concept_id, pill);
      });

      if (isAudio && audioContainer) {
        audioContainer.appendChild(pill);
      } else if (!isAudio && visualContainer) {
        visualContainer.appendChild(pill);
      }
    });

    if (visualGroup) {
      visualGroup.style.display = hasVisual ? "block" : "none";
    }

    // GSAP Staggered Reveal cho Pills
    if (window.gsap) {
      gsap.from(".concept-pill", {
        opacity: 0,
        y: 12,
        duration: 0.45,
        stagger: 0.04,
        ease: "power2.out"
      });
    }
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
  // 9. TIMELINE PHÂN ĐOẠN (CHUNKS 3-5s) & FILMSTRIP KEYFRAMES
  // =========================================================================
  function renderTimelineAndSegments(chunks, mode) {
    const strip = document.getElementById("timeline-scroll-strip");
    const specGrid = document.getElementById("spectrogram-grid-container");
    const radarList = document.getElementById("radar-bars-container");

    if (strip) strip.innerHTML = "";
    if (specGrid) specGrid.innerHTML = "";
    if (radarList) radarList.innerHTML = "";

    if (!chunks || chunks.length === 0) return;

    chunks.forEach((chk, idx) => {
      // 1. Card phân đoạn trên Timeline Strip
      const seg = document.createElement("div");
      seg.className = `segment-item ${idx === 0 ? "active" : ""}`;
      seg.id = `segment-chunk-${idx}`;

      const thumbImgSrc = chk.keyframe_img || chk.spectrogram_img || "";

      seg.innerHTML = `
        <div class="segment-thumb">
          ${thumbImgSrc ? `<img src="${thumbImgSrc}" alt="Segment ${idx}" />` : ""}
          <span class="segment-time-badge">${chk.timestamp}</span>
        </div>
        <div class="segment-meta">
          <span class="segment-genre">${chk.prediction}</span>
          <span class="segment-score">${chk.confidence}% tin cậy</span>
          <span class="segment-tag">${chk.visual_cues && chk.visual_cues.length ? chk.visual_cues.join(", ") : chk.crisp_axes ? `Pitch: ${chk.crisp_axes.Pitch}` : ""}</span>
        </div>
      `;

      // Bấm vào phân đoạn: Tua video/audio đến giây tương ứng và làm nổi bật concept
      seg.addEventListener("click", () => {
        document.querySelectorAll(".segment-item").forEach(s => s.classList.remove("active"));
        seg.classList.add("active");
        seekToTimestamp(chk.start_time);
        highlightChunkConcepts(chk);
      });

      if (strip) strip.appendChild(seg);

      // 2. Card ảnh phổ Mel & CQT trong Subtab
      if (specGrid && chk.spectrogram_img) {
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

    // 3. Render 7 Trục MPEG-7 từ chunk đầu tiên
    if (radarList && chunks[0] && chunks[0].radar) {
      renderMpeg7RadarBars(chunks[0]);
    }

    // Thiết lập subtab chuyển đổi giữa Filmstrip, Phổ tần và Radar
    setupTimelineSubtabs();
  }

  function seekToTimestamp(seconds) {
    if (activeMediaType === "video" && videoPlayer && videoPlayer.src) {
      // Nếu người dùng đang xem tab khác (URL YouTube, ghi âm, tải tệp âm thanh), tự động nhảy về tab Video
      if (currentActiveTab !== "video") {
        switchTab("video");
      }
      if (videoContainer) {
        videoContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
      videoPlayer.currentTime = seconds;
      const playPromise = videoPlayer.play();
      if (playPromise !== undefined) {
        playPromise.catch(e => console.warn("Video play interrupted/aborted:", e));
      }
    } else if (activeMediaType === "audio" && audioPlayer && audioPlayer.src) {
      // Nếu người dùng đang xem tab khác (Video, URL, Mic), tự động nhảy về tab Audio
      if (currentActiveTab !== "audio") {
        switchTab("audio");
      }
      if (audioContainer) {
        audioContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
      audioPlayer.currentTime = seconds;
      const playPromise = audioPlayer.play();
      if (playPromise !== undefined) {
        playPromise.catch(e => console.warn("Audio play interrupted/aborted:", e));
      }
    }
  }

  function highlightChunkConcepts(chunk) {
    // Làm nổi bật các concept được kích hoạt trong chunk này
    document.querySelectorAll(".concept-pill").forEach(p => p.style.opacity = "0.5");

    const activeIds = [...(chunk.active_audio_concepts || []), ...(chunk.active_visual_concepts || [])];
    activeIds.forEach(cid => {
      const pill = document.getElementById(`pill-${cid}`);
      if (pill) {
        pill.style.opacity = "1";
        if (window.gsap) {
          gsap.fromTo(pill, { scale: 1.0 }, { scale: 1.08, duration: 0.25, yoyo: true, repeat: 1 });
        }
      }
    });
  }

  function renderMpeg7RadarBars(chunk) {
    const radarList = document.getElementById("radar-bars-container");
    if (!radarList) return;

    radarList.innerHTML = "";
    const axesOrder = ["Energy", "Harmony", "Pitch", "Spectral", "Tempo", "Timbre", "Vibrato"];

    axesOrder.forEach(ax => {
      const val = chunk.radar[ax] || 0.5;
      const crispVal = chunk.crisp_axes[ax] || "Vừa";

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

  function setupTimelineSubtabs() {
    const subtabs = document.querySelectorAll(".subtab-btn");
    const filmstripView = document.getElementById("timeline-scroll-strip");
    const specView = document.getElementById("spectrogram-grid-container");
    const radarView = document.getElementById("radar-bars-container");

    subtabs.forEach(btn => {
      btn.addEventListener("click", () => {
        subtabs.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        const target = btn.getAttribute("data-subtab");
        if (filmstripView) filmstripView.style.display = (target === "filmstrip") ? "grid" : "none";
        if (specView) specView.style.display = (target === "spectrogram") ? "grid" : "none";
        if (radarView) radarView.style.display = (target === "radar") ? "flex" : "none";
      });
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

  // Nút Reset góc nhìn Camera 3D
  const resetCameraBtn = document.getElementById("btn-reset-graph-cam");
  if (resetCameraBtn) {
    resetCameraBtn.addEventListener("click", () => {
      graph3d.resetCameraView();
    });
  }
});

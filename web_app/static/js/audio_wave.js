/**
 * Heritage XAI - 3D Audio-Reactive Visualizer (Three.js WebGL)
 * ------------------------------------------------------------
 * Thiết kế chuẩn mực cao cấp đáp ứng 2 chế độ trực quan nghệ thuật:
 * 
 * 1. Chế độ "Đường sóng ngang" (Line Waveform - Mặc định):
 *    - Baseline là một đường thẳng ngang tĩnh lặng, thanh lịch tiệp màu nền giấy #FAF9F6.
 *    - Khi phát âm thanh: Nảy các đỉnh nhọn sắc nét (sharp needle peaks) theo nhịp Bass
 *      ở tâm và giai điệu Melody ở hai cánh, không nhảy loạn xạ hay phân tán ngẫu nhiên.
 *    - Cửa sổ Gaussian ghim êm ở hai đầu mép, làm mượt phản hồi bằng bộ lọc EMA.
 * 
 * 2. Chế độ "Quả cầu âm thanh 3D xoay" (3D Rotating Spiky Audio Orb):
 *    - Quả cầu phân bố hạt 3D xoay đều êm dịu trên trục Y và X.
 *    - Khi phát âm thanh: Bùng nổ các đỉnh gai nhọn vươn ra xung quanh theo Bass và Melody.
 * 
 * Tối ưu 60 FPS, tương thích hoàn toàn Web Audio API.
 */

class AudioWaveRibbon {
  constructor(containerId = "wave-canvas-container") {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.warn(`[AudioWaveVisualizer] Không tìm thấy phần tử #${containerId}`);
      return;
    }

    // Chế độ trực quan: Duy nhất Quả cầu 3D xoay phản hồi âm thanh
    this.mode = "sphere";

    this.scene = null;
    this.camera = null;
    this.renderer = null;

    // Hệ thống quả cầu âm thanh 3D xoay nảy gai nhọn
    this.sphereGroup = null;
    this.sphereParticles = null;
    this.sphereParticleMeta = [];
    this.sphereParticleCount = 2400;

    // Web Audio API members
    this.audioCtx = null;
    this.analyser = null;
    this.audioSourceNode = null;
    this.freqData = null;
    this.smoothedData = null;
    this.smoothedBass = 0;
    this.smoothedMid = 0;
    this.smoothedTreble = 0;

    this.isPlaying = false;
    this.isLiveMic = false;

    // Animation & Observers
    this.animId = null;
    this.clock = null;
    this.isVisible = true;

    this.initThree();
    this.initAudioSphere();
    this.initObservers();
    this.startAnimationLoop();
  }

  initThree() {
    const width = this.container.clientWidth || 700;
    const height = this.container.clientHeight || 260;

    this.scene = new THREE.Scene();
    this.scene.background = null; // Nền trong suốt tiệp nền #FAF9F6

    // Camera phối cảnh tập trung vào Quả cầu 3D, mở rộng headroom cho các đỉnh gai âm thanh
    this.camera = new THREE.PerspectiveCamera(36, width / height, 0.1, 100);
    this.camera.position.set(0, 0, 10.5);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.container.innerHTML = "";
    this.container.appendChild(this.renderer.domElement);

    this.clock = new THREE.Clock();
  }

  // ---------------------------------------------------------------------------
  // 1. KHỞI TẠO QUẢ CẦU ÂM THANH 3D XOAY (3D ROTATING SPIKY ORB)
  // ---------------------------------------------------------------------------
  initAudioSphere() {
    this.sphereGroup = new THREE.Group();
    this.sphereGroup.name = "sphereVisualizerGroup";

    const count = this.sphereParticleCount;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    this.sphereParticleMeta = [];

    const baseRadius = 2.4;
    const colCore = new THREE.Color(0x1E6B5B);
    const colMid = new THREE.Color(0xC97A1B);
    const colSpike = new THREE.Color(0xC84B31);

    // Phân bố đều các hạt trên mặt cầu bằng thuật toán Fibonacci Sphere
    const goldenRatio = (1 + Math.sqrt(5)) / 2;

    for (let i = 0; i < count; i++) {
      const theta = 2 * Math.PI * i / goldenRatio;
      const phi = Math.acos(1 - 2 * (i + 0.5) / count);

      const nx = Math.sin(phi) * Math.cos(theta);
      const ny = Math.sin(phi) * Math.sin(theta);
      const nz = Math.cos(phi);

      const x = nx * baseRadius;
      const y = ny * baseRadius;
      const z = nz * baseRadius;

      const pIdx = i * 3;
      positions[pIdx] = x;
      positions[pIdx + 1] = y;
      positions[pIdx + 2] = z;

      this.sphereParticleMeta.push({
        nx: nx,
        ny: ny,
        nz: nz,
        baseRadius: baseRadius,
        theta: theta,
        phi: phi
      });

      // Màu khởi tạo cho quả cầu
      const col = new THREE.Color().copy(colCore).lerp(colMid, (ny + 1) / 2);
      colors[pIdx] = col.r;
      colors[pIdx + 1] = col.g;
      colors[pIdx + 2] = col.b;
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const particleTexture = this.createCircleTexture();

    const material = new THREE.PointsMaterial({
      size: 0.22,
      map: particleTexture,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      blending: THREE.NormalBlending,
      depthWrite: false
    });

    this.sphereParticles = new THREE.Points(geometry, material);
    this.sphereGroup.add(this.sphereParticles);
    this.scene.add(this.sphereGroup);
  }

  createCircleTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext("2d");

    const gradient = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
    gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
    gradient.addColorStop(0.35, "rgba(255, 255, 255, 0.85)");
    gradient.addColorStop(0.7, "rgba(255, 255, 255, 0.3)");
    gradient.addColorStop(1, "rgba(255, 255, 255, 0)");

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 32, 32);

    const texture = new THREE.Texture(canvas);
    texture.needsUpdate = true;
    return texture;
  }

  // ---------------------------------------------------------------------------
  // 3. CHUYỂN ĐỔI CHẾ ĐỘ (LINE <-> SPHERE)
  // ---------------------------------------------------------------------------
  setupToggleButton() {
    const toggleBtn = document.getElementById("btn-toggle-wave-mode");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        this.toggleMode();
      });
    }
  }

  toggleMode() {
    this.mode = (this.mode === "line") ? "sphere" : "line";
    this.applyModeVisibility();
  }

  initObservers() {
    if (window.ResizeObserver) {
      const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          if (width > 0 && height > 0) {
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
          }
        }
      });
      resizeObserver.observe(this.container);
    }

    if (window.IntersectionObserver) {
      const observer = new IntersectionObserver((entries) => {
        this.isVisible = entries[0].isIntersecting;
      }, { threshold: 0.1 });
      observer.observe(this.container);
    }
  }

  // ---------------------------------------------------------------------------
  // 4. KẾT NỐI NGUỒN ÂM THANH (HTML5 VIDEO / AUDIO / MIC)
  // ---------------------------------------------------------------------------
  ensureAudioContext() {
    if (!window.__HERITAGE_AUDIO_CTX__) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      window.__HERITAGE_AUDIO_CTX__ = new AudioContextClass();
    }
    this.audioCtx = window.__HERITAGE_AUDIO_CTX__;

    if (!window.__HERITAGE_ANALYSER__) {
      window.__HERITAGE_ANALYSER__ = this.audioCtx.createAnalyser();
      window.__HERITAGE_ANALYSER__.fftSize = 256;
      window.__HERITAGE_ANALYSER__.smoothingTimeConstant = 0.85;
    }
    this.analyser = window.__HERITAGE_ANALYSER__;

    if (!this.freqData) {
      this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
      this.smoothedData = new Float32Array(this.analyser.frequencyBinCount);
    }

    if (this.audioCtx.state === "suspended") {
      this.audioCtx.resume().catch(() => {});
    }
  }

  connectMediaElement(mediaElement) {
    if (!mediaElement) return;
    this.ensureAudioContext();

    try {
      if (!mediaElement._webAudioSourceNode) {
        mediaElement._webAudioSourceNode = this.audioCtx.createMediaElementSource(mediaElement);
      }
      this.audioSourceNode = mediaElement._webAudioSourceNode;

      try { this.audioSourceNode.disconnect(); } catch (e) {}
      this.audioSourceNode.connect(this.analyser);
      try { this.analyser.disconnect(); } catch (e) {}
      this.analyser.connect(this.audioCtx.destination);

      this.isPlaying = true;
      this.isLiveMic = false;
    } catch (e) {
      console.warn("[AudioWaveVisualizer] Fallback connection for media element:", e);
      this.isPlaying = true;
      this.isLiveMic = false;
    }
  }

  connectMicrophoneStream(stream) {
    if (!stream) return;
    this.ensureAudioContext();

    try {
      if (this.audioSourceNode) {
        try { this.audioSourceNode.disconnect(); } catch (e) {}
      }

      this.audioSourceNode = this.audioCtx.createMediaStreamSource(stream);
      this.audioSourceNode.connect(this.analyser);

      this.isPlaying = true;
      this.isLiveMic = true;
    } catch (e) {
      console.error("[AudioWaveVisualizer] Không thể kết nối Mic Stream:", e);
    }
  }

  disconnect() {
    this.isPlaying = false;
    this.isLiveMic = false;
  }

  // ---------------------------------------------------------------------------
  // 5. VÒNG LẶP RENDER CHÍNH (60 FPS)
  // ---------------------------------------------------------------------------
  startAnimationLoop() {
    const animate = () => {
      this.animId = requestAnimationFrame(animate);

      if (!this.isVisible) return;

      const elapsedTime = this.clock.getElapsedTime();

      let bass = 0;
      let mid = 0;
      let treble = 0;

      if (this.analyser && (this.isPlaying || this.isLiveMic)) {
        this.analyser.getByteFrequencyData(this.freqData);

        // EMA Smoothing trên từng bin tần số (Tránh giật cục, tạo đỉnh nhọn thanh thoát)
        const alpha = 0.24;
        for (let i = 0; i < this.freqData.length; i++) {
          const normVal = this.freqData[i] / 255.0;
          this.smoothedData[i] = this.smoothedData[i] * (1.0 - alpha) + normVal * alpha;
        }

        // Tính bass, mid, treble đã làm mượt
        let sumBass = 0, sumMid = 0, sumTreble = 0;
        for (let i = 0; i < 10; i++) sumBass += this.smoothedData[i];
        for (let i = 10; i < 45; i++) sumMid += this.smoothedData[i];
        for (let i = 45; i < 95; i++) sumTreble += this.smoothedData[i];

        const rawBass = sumBass / 10;
        const rawMid = sumMid / 35;
        const rawTreble = sumTreble / 50;

        this.smoothedBass += (rawBass - this.smoothedBass) * 0.28;
        this.smoothedMid += (rawMid - this.smoothedMid) * 0.28;
        this.smoothedTreble += (rawTreble - this.smoothedTreble) * 0.28;

        bass = this.smoothedBass;
        mid = this.smoothedMid;
        treble = this.smoothedTreble;

        this.updateVuMeters(bass, mid, treble);
      } else {
        // Chế độ chờ: Dải phẳng tĩnh lặng, thở rất nhẹ (Idle breathing)
        bass = 0.03 * (Math.sin(elapsedTime * 1.5) * 0.5 + 0.5);
        mid = 0.02 * (Math.cos(elapsedTime * 1.8) * 0.5 + 0.5);
        treble = 0.01;

        if (this.smoothedData) {
          for (let i = 0; i < this.smoothedData.length; i++) {
            this.smoothedData[i] *= 0.90; // Tắt dần êm dịu
          }
        }
      }

      // Cập nhật quả cầu âm thanh 3D xoay nảy gai nhọn theo nhịp Bass & Melody
      this.updateAudioSphereGeometry(elapsedTime, bass, mid, treble);

      this.renderer.render(this.scene, this.camera);
    };

    animate();
  }

  // ---------------------------------------------------------------------------
  // 7. CẬP NHẬT QUẢ CẦU 3D XOAY: NẢY CÁC ĐỈNH NHỌN XUNG QUANH THEO BASS & MELODY
  // ---------------------------------------------------------------------------
  updateAudioSphereGeometry(time, bass, mid, treble) {
    if (!this.sphereParticles || !this.sphereParticleMeta.length) return;

    // Xoay quả cầu đều đặn trong không gian 3D
    this.sphereGroup.rotation.y += 0.009;
    this.sphereGroup.rotation.x += 0.003;

    const positions = this.sphereParticles.geometry.attributes.position.array;
    const colors = this.sphereParticles.geometry.attributes.color.array;
    const len = this.sphereParticleMeta.length;

    const colCore = new THREE.Color(0x1E6B5B);
    const colMid = new THREE.Color(0xC97A1B);
    const colSpike = new THREE.Color(0xC84B31);

    const binCount = (this.smoothedData && this.smoothedData.length) ? this.smoothedData.length : 64;

    for (let i = 0; i < len; i++) {
      const p = this.sphereParticleMeta[i];
      const pIdx = i * 3;

      // Ánh xạ góc cực (phi) và vĩ độ sang các dải tần số FFT
      const normalizedLat = Math.abs(p.ny);
      const binIdx = Math.min(binCount - 1, Math.floor(normalizedLat * 40));
      const freqAmp = this.smoothedData ? this.smoothedData[binIdx] : 0.0;

      // Tạo đỉnh nhọn vươn ra (Sharp Radial Spikes)
      const sharpSpike = Math.pow(freqAmp, 1.8) * 1.5;

      // Nhịp đập toàn phần theo Bass (Pulsing Radius)
      const bassPulse = bass * 0.75;

      // Độ gợn sóng bề mặt theo Melody
      const surfaceRipple = Math.sin(p.theta * 4.0 + time * 3.0) * (mid * 0.35);

      const currentRadius = p.baseRadius + bassPulse + sharpSpike + surfaceRipple;

      // Vị trí 3D mới vươn nhọn dọc theo vector pháp tuyến
      positions[pIdx] = p.nx * currentRadius;
      positions[pIdx + 1] = p.ny * currentRadius;
      positions[pIdx + 2] = p.nz * currentRadius;

      // Đổi màu gai nhọn rực đỏ son khi vươn cao
      const spikeFactor = Math.min(1.0, (sharpSpike + bassPulse) / 0.9);
      const col = new THREE.Color().copy(colCore).lerp(colMid, (p.ny + 1) / 2).lerp(colSpike, spikeFactor);

      colors[pIdx] = col.r;
      colors[pIdx + 1] = col.g;
      colors[pIdx + 2] = col.b;
    }

    this.sphereParticles.geometry.attributes.position.needsUpdate = true;
    this.sphereParticles.geometry.attributes.color.needsUpdate = true;
  }

  updateVuMeters(bass, mid, treble) {
    const bBar = document.getElementById("vu-bass");
    const mBar = document.getElementById("vu-mid");
    const tBar = document.getElementById("vu-treble");

    if (bBar) bBar.style.height = `${Math.max(3, Math.min(14, bass * 18))}px`;
    if (mBar) mBar.style.height = `${Math.max(3, Math.min(14, mid * 18))}px`;
    if (tBar) tBar.style.height = `${Math.max(3, Math.min(14, treble * 18))}px`;
  }

  dispose() {
    if (this.animId) cancelAnimationFrame(this.animId);
    this.disconnect();
    if (this.lineParticles) {
      this.lineParticles.geometry.dispose();
      this.lineParticles.material.dispose();
    }
    if (this.sphereParticles) {
      this.sphereParticles.geometry.dispose();
      this.sphereParticles.material.dispose();
    }
    if (this.renderer) {
      this.renderer.dispose();
    }
  }
}

// Global instance helper
window.AudioWaveRibbon = AudioWaveRibbon;

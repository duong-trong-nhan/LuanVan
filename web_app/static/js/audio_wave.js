/**
 * Heritage XAI - Animated Audio-Reactive 3D Brand Identity (Three.js WebGL)
 * -------------------------------------------------------------------------
 * Khôi phục và nâng cấp quả cầu hạt 3D Fibonacci (AudioWaveRibbon) làm biểu tượng
 * nhận diện thương hiệu động ở góc trên bên trái của Header (#brand-orb-emblem):
 *
 * 1. Bốn trạng thái chuyển động mượt mà (Smooth Lerped State Machine):
 *    - "idle"           : Xoay nhẹ, nhịp thở tĩnh lặng khi chưa phát video.
 *    - "processing"     : Xoay giàu năng lượng với gợn sóng xoắn ốc đồng tâm khi AI đang phân tích.
 *    - "completed"      : Nhịp nở hoàn tất mượt mà và duy trì chuyển động ổn định sau khi phân tích xong.
 *    - "audio-reactive" : Phản hồi trực tiếp theo tín hiệu âm thanh thực tế (Bass, Mid, Treble & FFT bins)
 *                         của video/audio đang phát thông qua Web Audio API (không giả lập ngẫu nhiên).
 *
 * 2. Hiệu năng & An toàn tài nguyên:
 *    - Duy nhất 1 WebGL context, hỗ trợ mount() / unmount() / dispose() đầy đủ.
 *    - Tự động tạm dừng vòng lặp khi tab trình duyệt ẩn (visibilitychange) hoặc khuất tầm nhìn.
 *    - Tự động kích hoạt Fallback tĩnh (#brand-orb-fallback) nếu WebGL không khả dụng.
 *    - Tôn trọng thiết lập giảm chuyển động của hệ điều hành (prefers-reduced-motion).
 */

class AudioWaveRibbon {
    constructor(containerId = "wave-canvas-container") {
        this.containerId = containerId;
        this.container = null;
        this.emblemWrapper = null;

        // Trạng thái hoạt ảnh: "idle" | "processing" | "completed" | "audio-reactive"
        this.state = "idle";
        this.hasCompletedAnalysis = false;
        this.completionBloom = 0; // Nhịp nở mượt khi vừa hoàn tất phân tích

        // Trọng số nội suy mượt giữa các trạng thái (tránh giật cục khi chuyển trạng thái)
        this.stateWeights = {
            idle: 1.0,
            processing: 0.0,
            completed: 0.0,
            reactive: 0.0,
        };

        // Three.js core objects (duy nhất 1 WebGL context)
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.particleTexture = null;

        // Hệ thống Quả cầu 3D Fibonacci & Lõi đồng tâm
        this.sphereGroup = null;
        this.sphereParticles = null;
        this.innerCoreMesh = null;
        this.sphereParticleMeta = [];
        this.sphereParticleCount = 2400;

        // Màu sắc chủ đạo theo chuẩn Editorial Light Mode
        this.colCore = new THREE.Color(0x1e6b5b); // Xanh ngọc (Jade Emerald)
        this.colMid = new THREE.Color(0xc97a1b); // Vàng hổ phách (Amber Ochre)
        this.colSpike = new THREE.Color(0xc84b31); // Đỏ son (Vermilion Red)
        this.colAccentGenre = new THREE.Color(0x1e6b5b);

        // Web Audio API (chỉ phân tích âm thanh thực từ phần tử media đang phát)
        this.audioCtx = null;
        this.analyser = null;
        this.audioSourceNode = null;
        this.connectedMediaElement = null;
        this.freqData = null;
        this.smoothedData = null;
        this.smoothedBass = 0;
        this.smoothedMid = 0;
        this.smoothedTreble = 0;
        this.smoothedEnergy = 0;

        this.isPlaying = false;
        this.isMounted = false;
        this.isVisible = true;
        this.isTabActive = !document.hidden;
        this.prefersReducedMotion = false;

        // Observers & Event handlers
        this.animId = null;
        this.clock = null;
        this.resizeObserver = null;
        this.intersectionObserver = null;
        this._boundVisibilityChange = null;
        this._boundReducedMotionChange = null;
        this._reducedMotionQuery = null;

        this.mount(containerId);
    }

    // -------------------------------------------------------------------------
    // 1. KIỂM TRA HỖ TRỢ WEBGL & VÒNG ĐỜI MOUNT / UNMOUNT / DISPOSE
    // -------------------------------------------------------------------------
    isWebGLAvailable() {
        if (typeof window === "undefined" || typeof THREE === "undefined") {
            return false;
        }
        try {
            const testCanvas = document.createElement("canvas");
            return !!(
                window.WebGLRenderingContext &&
                (testCanvas.getContext("webgl") ||
                    testCanvas.getContext("experimental-webgl"))
            );
        } catch (e) {
            return false;
        }
    }

    mount(containerId = this.containerId) {
        if (this.isMounted) {
            this.unmount();
        }

        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.emblemWrapper = document.getElementById("brand-orb-emblem");

        if (!this.container) {
            console.warn(
                `[AudioWaveRibbon] Không tìm thấy phần tử #${containerId}`,
            );
            return false;
        }

        this.setupAccessibilityListeners();

        if (!this.isWebGLAvailable()) {
            this.activateFallback("Trình duyệt không hỗ trợ WebGL");
            return false;
        }

        const initialized = this.initThree();
        if (!initialized) {
            this.activateFallback("Không thể khởi tạo WebGLRenderer");
            return false;
        }

        if (this.emblemWrapper) {
            this.emblemWrapper.classList.remove("webgl-fallback");
        }
        const fallbackEl = document.getElementById("brand-orb-fallback");
        if (fallbackEl) {
            fallbackEl.style.display = "none";
        }

        this.initAudioSphere();
        this.initObservers();
        this.isMounted = true;
        this.setState(this.state);
        this.startAnimationLoop();
        return true;
    }

    activateFallback(reason) {
        console.warn(`[AudioWaveRibbon] Kích hoạt chế độ Fallback: ${reason}`);
        if (this.emblemWrapper) {
            this.emblemWrapper.classList.add("webgl-fallback");
        }
        const fallbackEl = document.getElementById("brand-orb-fallback");
        if (fallbackEl) {
            fallbackEl.style.display = "block";
        }
    }

    setupAccessibilityListeners() {
        if (window.matchMedia) {
            this._reducedMotionQuery = window.matchMedia(
                "(prefers-reduced-motion: reduce)",
            );
            this.prefersReducedMotion = this._reducedMotionQuery.matches;
            this._boundReducedMotionChange = (e) => {
                this.prefersReducedMotion = !!e.matches;
            };
            if (this._reducedMotionQuery.addEventListener) {
                this._reducedMotionQuery.addEventListener(
                    "change",
                    this._boundReducedMotionChange,
                );
            }
        }

        this._boundVisibilityChange = () => {
            this.isTabActive = !document.hidden;
            if (this.isTabActive && this.clock) {
                this.clock.getDelta();
            }
        };
        document.addEventListener(
            "visibilitychange",
            this._boundVisibilityChange,
        );
    }

    initThree() {
        try {
            const rect = this.container.getBoundingClientRect();
            const width = this.container.clientWidth || Math.round(rect.width) || 600;
            const height = this.container.clientHeight || Math.round(rect.height) || 550;

            this.scene = new THREE.Scene();
            this.scene.background = null;

            // Camera phối cảnh tập trung vào Quả cầu 3D trong background
            this.camera = new THREE.PerspectiveCamera(
                38,
                width / height,
                0.1,
                100,
            );
            this.camera.position.set(0, 0, 10.0);
            this.camera.lookAt(0, 0, 0);

            this.renderer = new THREE.WebGLRenderer({
                antialias: true,
                alpha: true,
                powerPreference: "high-performance",
            });
            this.renderer.setSize(width, height);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

            const domCanvas = this.renderer.domElement;
            domCanvas.addEventListener(
                "webglcontextlost",
                (e) => {
                    e.preventDefault();
                    if (this.animId) cancelAnimationFrame(this.animId);
                    this.activateFallback("WebGL context lost");
                },
                false,
            );

            this.container.innerHTML = "";
            this.container.appendChild(domCanvas);

            this.clock = new THREE.Clock();
            return true;
        } catch (err) {
            console.error("[AudioWaveRibbon] Lỗi initThree:", err);
            return false;
        }
    }

    // -------------------------------------------------------------------------
    // 2. KHÔI PHỤC QUẢ CẦU HẠT 3D FIBONACCI (3D ROTATING SPIKY ORB)
    // -------------------------------------------------------------------------
    initAudioSphere() {
        this.sphereGroup = new THREE.Group();
        this.sphereGroup.name = "brandAudioSphereGroup";
        // Dời quả cầu 3D cao lên một chút (y = 0.70) để nằm trọn cân đối sau thẻ lời giải thích
        this.sphereGroup.position.set(0, 0.70, 0);

        // Lõi cầu mờ bên trong tạo chiều sâu thị giác
        const coreGeo = new THREE.SphereGeometry(0.95, 24, 24);
        const coreMat = new THREE.MeshBasicMaterial({
            color: 0x1e6b5b,
            transparent: true,
            opacity: 0.14,
            wireframe: true,
        });
        this.innerCoreMesh = new THREE.Mesh(coreGeo, coreMat);
        this.sphereGroup.add(this.innerCoreMesh);

        // Phân bố hạt trên mặt cầu bằng thuật toán Fibonacci Sphere nguyên bản
        const count = this.sphereParticleCount;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);

        this.sphereParticleMeta = [];

        const baseRadius = 1.25;
        const goldenRatio = (1 + Math.sqrt(5)) / 2;

        for (let i = 0; i < count; i++) {
            const theta = (2 * Math.PI * i) / goldenRatio;
            const phi = Math.acos(1 - (2 * (i + 0.5)) / count);

            const nx = Math.sin(phi) * Math.cos(theta);
            const ny = Math.sin(phi) * Math.sin(theta);
            const nz = Math.cos(phi);

            const pIdx = i * 3;
            positions[pIdx] = nx * baseRadius;
            positions[pIdx + 1] = ny * baseRadius;
            positions[pIdx + 2] = nz * baseRadius;

            this.sphereParticleMeta.push({
                nx: nx,
                ny: ny,
                nz: nz,
                baseRadius: baseRadius,
                theta: theta,
                phi: phi,
            });

            const col = new THREE.Color()
                .copy(this.colCore)
                .lerp(this.colMid, (ny + 1) * 0.5);
            colors[pIdx] = col.r;
            colors[pIdx + 1] = col.g;
            colors[pIdx + 2] = col.b;
        }

        geometry.setAttribute(
            "position",
            new THREE.BufferAttribute(positions, 3),
        );
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

        this.particleTexture = this.createCircleTexture();

        const material = new THREE.PointsMaterial({
            size: 0.13,
            map: this.particleTexture,
            vertexColors: true,
            transparent: true,
            opacity: 0.95,
            blending: THREE.NormalBlending,
            depthWrite: false,
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
        gradient.addColorStop(0.4, "rgba(255, 255, 255, 0.9)");
        gradient.addColorStop(0.75, "rgba(255, 255, 255, 0.35)");
        gradient.addColorStop(1, "rgba(255, 255, 255, 0)");

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 32, 32);

        const texture = new THREE.Texture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    initObservers() {
        if (window.ResizeObserver && this.container) {
            this.resizeObserver = new ResizeObserver((entries) => {
                for (const entry of entries) {
                    const width = Math.round(entry.contentRect.width);
                    const height = Math.round(entry.contentRect.height);
                    if (width > 0 && height > 0 && this.renderer && this.camera) {
                        this.camera.aspect = width / height;
                        this.camera.updateProjectionMatrix();
                        this.renderer.setSize(width, height);
                    }
                }
            });
            this.resizeObserver.observe(this.container);
        }

        if (window.IntersectionObserver && this.container) {
            this.intersectionObserver = new IntersectionObserver(
                (entries) => {
                    if (entries && entries[0]) {
                        this.isVisible = entries[0].isIntersecting;
                    }
                },
                { threshold: 0.05 },
            );
            this.intersectionObserver.observe(this.container);
        }
    }

    // -------------------------------------------------------------------------
    // 3. QUẢN LÝ 4 TRẠNG THÁI HOẠT ẢNH (IDLE, PROCESSING, COMPLETED, AUDIO-REACTIVE)
    // -------------------------------------------------------------------------
    setState(newState, options = {}) {
        const validStates = [
            "idle",
            "processing",
            "completed",
            "audio-reactive",
        ];
        const target = validStates.includes(newState) ? newState : "idle";

        if (target === "completed") {
            this.hasCompletedAnalysis = true;
            this.completionBloom = 1.0;
            if (options.genreColor) {
                try {
                    this.colAccentGenre.set(options.genreColor);
                } catch (e) {}
            }
        } else if (target === "idle" && options.resetCompletion) {
            this.hasCompletedAnalysis = false;
            this.colAccentGenre.set(0x1e6b5b);
        }

        this.state = target;

        if (this.emblemWrapper) {
            this.emblemWrapper.setAttribute("data-orb-state", target);
            const stateLabels = {
                idle: "Biểu tượng 3D Nhận diện Hệ thống (Trạng thái: Chờ)",
                processing:
                    "Biểu tượng 3D Nhận diện Hệ thống (Trạng thái: Đang phân tích đa phương thức...)",
                completed:
                    "Biểu tượng 3D Nhận diện Hệ thống (Trạng thái: Đã hoàn tất phân tích)",
                "audio-reactive":
                    "Biểu tượng 3D Nhận diện Hệ thống (Trạng thái: Đang phản hồi âm thanh thời gian thực)",
            };
            this.emblemWrapper.title =
                stateLabels[target] || stateLabels.idle;
        }
    }

    setProcessingState(isProcessing) {
        if (isProcessing) {
            this.setState("processing");
        } else if (this.isPlaying) {
            this.setState("audio-reactive");
        } else if (this.hasCompletedAnalysis) {
            this.setState("completed");
        } else {
            this.setState("idle");
        }
    }

    setCompletedState(genreColor) {
        this.hasCompletedAnalysis = true;
        if (this.isPlaying) {
            if (genreColor) {
                try {
                    this.colAccentGenre.set(genreColor);
                } catch (e) {}
            }
            this.completionBloom = 0.85;
            this.setState("audio-reactive");
        } else {
            this.setState("completed", { genreColor });
        }
    }

    // -------------------------------------------------------------------------
    // 4. KẾT NỐI ÂM THANH THỰC TẾ CỦA VIDEO / AUDIO ĐANG PHÁT (WEB AUDIO API)
    // -------------------------------------------------------------------------
    ensureAudioContext() {
        if (!window.__HERITAGE_AUDIO_CTX__) {
            const AudioContextClass =
                window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) return false;
            window.__HERITAGE_AUDIO_CTX__ = new AudioContextClass();
        }
        this.audioCtx = window.__HERITAGE_AUDIO_CTX__;

        if (!window.__HERITAGE_ANALYSER__) {
            window.__HERITAGE_ANALYSER__ = this.audioCtx.createAnalyser();
            window.__HERITAGE_ANALYSER__.fftSize = 256;
            window.__HERITAGE_ANALYSER__.smoothingTimeConstant = 0.82;
        }
        this.analyser = window.__HERITAGE_ANALYSER__;

        if (!this.freqData) {
            this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
            this.smoothedData = new Float32Array(
                this.analyser.frequencyBinCount,
            );
        }

        if (this.audioCtx.state === "suspended") {
            this.audioCtx.resume().catch(() => {});
        }
        return true;
    }

    connectMediaElement(mediaElement) {
        if (!mediaElement) return;
        if (!this.ensureAudioContext()) return;

        try {
            if (!mediaElement._webAudioSourceNode) {
                mediaElement._webAudioSourceNode =
                    this.audioCtx.createMediaElementSource(mediaElement);
            }
            this.audioSourceNode = mediaElement._webAudioSourceNode;
            this.connectedMediaElement = mediaElement;

            try {
                this.audioSourceNode.disconnect();
            } catch (e) {}
            this.audioSourceNode.connect(this.analyser);
            try {
                this.analyser.disconnect();
            } catch (e) {}
            this.analyser.connect(this.audioCtx.destination);

            this.isPlaying = !mediaElement.paused && !mediaElement.ended;
            if (this.isPlaying) {
                this.setState("audio-reactive");
            }
        } catch (e) {
            console.warn(
                "[AudioWaveRibbon] Kết nối phần tử media với Web Audio API:",
                e,
            );
            this.connectedMediaElement = mediaElement;
            this.isPlaying = !mediaElement.paused && !mediaElement.ended;
            if (this.isPlaying) {
                this.setState("audio-reactive");
            }
        }
    }

    /**
     * Giữ hàm tương thích nếu được gọi từ module ghi âm, nhưng KHÔNG tự yêu cầu quyền mic.
     */
    connectMicrophoneStream(stream) {
        if (!stream) return;
        if (!this.ensureAudioContext()) return;
        try {
            if (this.audioSourceNode) {
                try {
                    this.audioSourceNode.disconnect();
                } catch (e) {}
            }
            this.audioSourceNode = this.audioCtx.createMediaStreamSource(stream);
            this.audioSourceNode.connect(this.analyser);
            this.isPlaying = true;
            this.setState("audio-reactive");
        } catch (e) {
            console.warn("[AudioWaveRibbon] Không kết nối luồng stream:", e);
        }
    }

    disconnect() {
        this.isPlaying = false;
        if (this.state === "audio-reactive") {
            this.setState(this.hasCompletedAnalysis ? "completed" : "idle");
        }
    }

    // -------------------------------------------------------------------------
    // 5. VÒNG LẶP HOẠT ẢNH 60 FPS & TRÍCH XUẤT DẢI TẦN THỰC TẾ
    // -------------------------------------------------------------------------
    startAnimationLoop() {
        if (this.animId) cancelAnimationFrame(this.animId);

        const animate = () => {
            this.animId = requestAnimationFrame(animate);

            // Tạm dừng tính toán & vẽ khi tab trình duyệt ẩn hoặc biểu tượng khuất tầm nhìn
            if (!this.isMounted || !this.isVisible || !this.isTabActive) {
                return;
            }

            const dt = Math.min(0.05, this.clock.getDelta());
            const elapsedTime = this.clock.getElapsedTime();

            // Kiểm tra trạng thái thực tế của phần tử media đang kết nối
            if (this.connectedMediaElement) {
                const mediaActuallyPlaying =
                    !this.connectedMediaElement.paused &&
                    !this.connectedMediaElement.ended &&
                    this.connectedMediaElement.readyState >= 2;
                if (mediaActuallyPlaying && !this.isPlaying) {
                    this.isPlaying = true;
                    this.setState("audio-reactive");
                } else if (!mediaActuallyPlaying && this.isPlaying) {
                    this.isPlaying = false;
                    if (this.state === "audio-reactive") {
                        this.setState(
                            this.hasCompletedAnalysis ? "completed" : "idle",
                        );
                    }
                }
            }

            // Nội suy mượt trọng số 4 trạng thái (State Weight Lerping)
            const lerpRate = Math.min(1.0, dt * 6.5);
            const targetIdle = this.state === "idle" ? 1.0 : 0.0;
            const targetProc = this.state === "processing" ? 1.0 : 0.0;
            const targetComp = this.state === "completed" ? 1.0 : 0.0;
            const targetReac = this.state === "audio-reactive" ? 1.0 : 0.0;

            this.stateWeights.idle +=
                (targetIdle - this.stateWeights.idle) * lerpRate;
            this.stateWeights.processing +=
                (targetProc - this.stateWeights.processing) * lerpRate;
            this.stateWeights.completed +=
                (targetComp - this.stateWeights.completed) * lerpRate;
            this.stateWeights.reactive +=
                (targetReac - this.stateWeights.reactive) * lerpRate;

            // Khi phân tích (processing), quả cầu 3D định vị ở chính giữa màn hình (y = 0.0)
            // Khi hoạt động thông thường, quả cầu dời cao lên một chút (y = 0.70) phía sau thẻ diễn giải
            const targetPosY = this.state === "processing" ? 0.0 : -0.70;
            if (this.sphereGroup) {
                this.sphereGroup.position.y +=
                    (targetPosY - this.sphereGroup.position.y) * Math.min(1.0, dt * 5.0);
            }

            if (this.completionBloom > 0.001) {
                this.completionBloom = Math.max(
                    0,
                    this.completionBloom - dt * 0.9,
                );
            } else {
                this.completionBloom = 0;
            }

            // Phân tích tín hiệu âm thanh thực tế từ AnalyserNode
            let bass = 0;
            let mid = 0;
            let treble = 0;

            if (this.analyser && this.isPlaying && this.freqData) {
                this.analyser.getByteFrequencyData(this.freqData);

                // Làm mượt phổ tần số bằng bộ lọc EMA
                const alpha = 0.28;
                for (let i = 0; i < this.freqData.length; i++) {
                    const normVal = this.freqData[i] / 255.0;
                    this.smoothedData[i] =
                        this.smoothedData[i] * (1.0 - alpha) + normVal * alpha;
                }

                // Tách 3 dải tần số thực tế:
                // - Bass (dải trầm: bins 0..9)
                // - Mid (dải trung / giọng hát & nhạc cụ chính: bins 10..44)
                // - Treble (dải cao / âm sắc & phách: bins 45..94)
                let sumBass = 0;
                let sumMid = 0;
                let sumTreble = 0;
                for (let i = 0; i < 10; i++) sumBass += this.smoothedData[i];
                for (let i = 10; i < 45; i++) sumMid += this.smoothedData[i];
                for (let i = 45; i < 95; i++) sumTreble += this.smoothedData[i];

                const rawBass = sumBass / 10;
                const rawMid = sumMid / 35;
                const rawTreble = sumTreble / 50;

                this.smoothedBass += (rawBass - this.smoothedBass) * 0.3;
                this.smoothedMid += (rawMid - this.smoothedMid) * 0.3;
                this.smoothedTreble += (rawTreble - this.smoothedTreble) * 0.3;
            } else {
                // Khi video pause hoặc không phát âm thanh: giảm dần mượt mà về 0
                this.smoothedBass *= 0.88;
                this.smoothedMid *= 0.88;
                this.smoothedTreble *= 0.88;

                if (this.smoothedData) {
                    for (let i = 0; i < this.smoothedData.length; i++) {
                        this.smoothedData[i] *= 0.88;
                    }
                }
            }

            bass = this.smoothedBass;
            mid = this.smoothedMid;
            treble = this.smoothedTreble;
            this.smoothedEnergy = bass * 0.45 + mid * 0.35 + treble * 0.2;

            this.updateAudioSphereGeometry(
                elapsedTime,
                dt,
                bass,
                mid,
                treble,
            );

            if (this.renderer && this.scene && this.camera) {
                this.renderer.render(this.scene, this.camera);
            }
        };

        animate();
    }

    // -------------------------------------------------------------------------
    // 6. CẬP NHẬT HÌNH HỌC & MÀU SẮC QUẢ CẦU 3D THEO 4 TRẠNG THÁI & ÂM THANH
    // -------------------------------------------------------------------------
    updateAudioSphereGeometry(time, dt, bass, mid, treble) {
        if (!this.sphereParticles || !this.sphereParticleMeta.length) return;

        const wIdle = this.stateWeights.idle;
        const wProc = this.stateWeights.processing;
        const wComp = this.stateWeights.completed;
        const wReac = this.stateWeights.reactive;

        // Tôn trọng thiết lập giảm chuyển động của hệ điều hành (prefers-reduced-motion)
        const motionScale = this.prefersReducedMotion ? 0.22 : 1.0;

        // Tốc độ xoay thích ứng mượt giữa 4 trạng thái + cường độ âm thanh thực
        const rotSpeedY =
            (0.45 * wIdle +
                2.2 * wProc +
                0.65 * wComp +
                (0.75 + mid * 2.1 + bass * 1.2) * wReac) *
            motionScale;
        const rotSpeedX =
            (0.16 * wIdle +
                0.85 * wProc +
                0.22 * wComp +
                (0.28 + treble * 1.4) * wReac) *
            motionScale;

        this.sphereGroup.rotation.y += rotSpeedY * dt;
        this.sphereGroup.rotation.x += rotSpeedX * dt;

        // Cập nhật lõi cầu bên trong
        if (this.innerCoreMesh) {
            const coreScale =
                1.0 +
                (bass * 0.42 * wReac +
                    Math.sin(time * 4.5) * 0.08 * wProc +
                    this.completionBloom * 0.16) *
                    motionScale;
            this.innerCoreMesh.scale.set(coreScale, coreScale, coreScale);
            this.innerCoreMesh.material.opacity =
                0.12 + bass * 0.22 * wReac + 0.08 * wProc;
        }

        const positions =
            this.sphereParticles.geometry.attributes.position.array;
        const colors = this.sphereParticles.geometry.attributes.color.array;
        const len = this.sphereParticleMeta.length;
        const binCount =
            this.smoothedData && this.smoothedData.length
                ? this.smoothedData.length
                : 64;

        const workColor = new THREE.Color();
        const baseTopColor = new THREE.Color()
            .copy(this.colMid)
            .lerp(this.colAccentGenre, wComp * 0.45);

        for (let i = 0; i < len; i++) {
            const p = this.sphereParticleMeta[i];
            const pIdx = i * 3;

            // 1. Thành phần Idle & Completed: nhịp thở điều hòa rất nhẹ
            const idleBreathing =
                Math.sin(time * 1.6 + p.phi * 1.5) *
                0.025 *
                (wIdle + wComp * 1.2);
            const bloomOffset =
                Math.sin(p.phi * 3.0 + time * 4.0) *
                this.completionBloom *
                0.12;

            // 2. Thành phần Processing: sóng xoắn ốc đồng tâm lan tỏa khi AI đang phân tích
            const procWave =
                wProc > 0.001
                    ? Math.sin(p.theta * 3.0 - time * 5.5 + p.phi * 2.5) *
                      0.14 *
                      wProc
                    : 0;

            // 3. Thành phần Audio-Reactive: phản hồi trực tiếp theo phổ tần số FFT thực tế
            let audioDisplacement = 0;
            let spikeFactor = 0;

            if (wReac > 0.001 || bass > 0.005) {
                const normalizedLat = Math.abs(p.ny);
                const binIdx = Math.min(
                    binCount - 1,
                    Math.floor(normalizedLat * 42),
                );
                const freqAmp = this.smoothedData
                    ? this.smoothedData[binIdx]
                    : 0.0;

                // Gai nhọn hướng tâm theo cường độ tần số tại vĩ độ tương ứng + Treble
                const sharpSpike =
                    Math.pow(freqAmp, 1.65) * (0.38 + treble * 0.22);

                // Độ giãn nở toàn cầu theo dải trầm (Bass)
                const bassPulse = bass * 0.22;

                // Gợn sóng bề mặt theo dải trung (Mid)
                const surfaceRipple =
                    Math.sin(p.theta * 4.0 + time * 3.2) * (mid * 0.11);

                audioDisplacement =
                    (bassPulse + sharpSpike + surfaceRipple) *
                    Math.max(wReac, 0.35);
                spikeFactor = Math.min(
                    1.0,
                    (sharpSpike + bassPulse) / 0.32,
                );
            }

            // Giới hạn an toàn tuyệt đối (clamp) để bán kính không bao giờ vượt quá vùng nhìn của camera (1.68 < 2.0)
            const totalOffset =
                (idleBreathing + bloomOffset + procWave + audioDisplacement) *
                motionScale;
            const currentRadius = Math.min(1.68, p.baseRadius + totalOffset);

            positions[pIdx] = p.nx * currentRadius;
            positions[pIdx + 1] = p.ny * currentRadius;
            positions[pIdx + 2] = p.nz * currentRadius;

            // Nội suy màu sắc: Xanh ngọc (Core) -> Hổ phách (Mid) -> Đỏ son (Spike / Processing)
            const procColorMix =
                wProc *
                (0.35 + 0.35 * Math.sin(p.theta * 2.0 - time * 4.5));
            const redMix = Math.min(
                1.0,
                spikeFactor + procColorMix + this.completionBloom * 0.4,
            );

            workColor
                .copy(this.colCore)
                .lerp(baseTopColor, (p.ny + 1) * 0.5)
                .lerp(this.colSpike, redMix);

            colors[pIdx] = workColor.r;
            colors[pIdx + 1] = workColor.g;
            colors[pIdx + 2] = workColor.b;
        }

        this.sphereParticles.geometry.attributes.position.needsUpdate = true;
        this.sphereParticles.geometry.attributes.color.needsUpdate = true;
    }

    // -------------------------------------------------------------------------
    // 7. GIẢI PHÓNG TÀI NGUYÊN (UNMOUNT & DISPOSE)
    // -------------------------------------------------------------------------
    unmount() {
        this.isMounted = false;
        if (this.animId) {
            cancelAnimationFrame(this.animId);
            this.animId = null;
        }
        this.disconnect();

        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        if (this.intersectionObserver) {
            this.intersectionObserver.disconnect();
            this.intersectionObserver = null;
        }
        if (this._boundVisibilityChange) {
            document.removeEventListener(
                "visibilitychange",
                this._boundVisibilityChange,
            );
            this._boundVisibilityChange = null;
        }
        if (
            this._reducedMotionQuery &&
            this._boundReducedMotionChange &&
            this._reducedMotionQuery.removeEventListener
        ) {
            this._reducedMotionQuery.removeEventListener(
                "change",
                this._boundReducedMotionChange,
            );
            this._boundReducedMotionChange = null;
        }

        if (this.innerCoreMesh) {
            if (this.innerCoreMesh.geometry)
                this.innerCoreMesh.geometry.dispose();
            if (this.innerCoreMesh.material)
                this.innerCoreMesh.material.dispose();
            this.innerCoreMesh = null;
        }
        if (this.sphereParticles) {
            if (this.sphereParticles.geometry)
                this.sphereParticles.geometry.dispose();
            if (this.sphereParticles.material)
                this.sphereParticles.material.dispose();
            this.sphereParticles = null;
        }
        if (this.particleTexture) {
            this.particleTexture.dispose();
            this.particleTexture = null;
        }
        if (this.renderer) {
            this.renderer.dispose();
            if (
                this.renderer.domElement &&
                this.renderer.domElement.parentNode
            ) {
                this.renderer.domElement.parentNode.removeChild(
                    this.renderer.domElement,
                );
            }
            this.renderer = null;
        }

        this.scene = null;
        this.camera = null;
        this.sphereGroup = null;
        this.sphereParticleMeta = [];
    }

    dispose() {
        this.unmount();
    }
}

// Global instance helper
window.AudioWaveRibbon = AudioWaveRibbon;

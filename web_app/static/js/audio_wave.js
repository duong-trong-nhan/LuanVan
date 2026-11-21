/**
 * HERA - Minimalist 2D Analog Acoustic Spectrum Strip (Web Audio API FFT)
 * ------------------------------------------------------------------------
 * Thay thế quả cầu 3D nền (vốn gây nhiễu đọc văn bản & nặng WebGL) bằng
 * dải phổ âm học 2D siêu mảnh, tĩnh lặng, chỉ phản hồi theo tín hiệu âm thanh
 * thực tế từ Audio Player hoặc Live Microphone.
 */

class AudioWaveRibbon {
    constructor(canvasIds = ["audio-spectrum-canvas", "mic-spectrum-canvas"]) {
        this.canvasIds = Array.isArray(canvasIds) ? canvasIds : [canvasIds];
        this.canvases = [];
        this.state = "idle";
        this.accentColor = "#C84B31";

        // Web Audio API (chỉ khởi tạo khi phát âm thanh hoặc thu âm thực tế)
        this.audioCtx = null;
        this.analyser = null;
        this.sourceMap = new WeakMap();
        this.streamSource = null;
        this.freqData = null;
        this.barCount = 52;
        this.smoothedBars = new Float32Array(this.barCount);

        this.isActive = false;
        this.animId = null;

        this._discoverCanvases();
        this._drawStatic();
        window.addEventListener("resize", () => this._drawStatic());
    }

    _discoverCanvases() {
        this.canvases = this.canvasIds
            .map((id) => document.getElementById(id))
            .filter((el) => el && el.tagName === "CANVAS");
    }

    _ensureAudioContext() {
        if (!this.audioCtx) {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) return null;
            this.audioCtx = new Ctx();
        }
        if (this.audioCtx.state === "suspended") {
            this.audioCtx.resume().catch(() => {});
        }
        if (!this.analyser) {
            this.analyser = this.audioCtx.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.78;
            this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
        }
        return this.audioCtx;
    }

    connectMediaElement(mediaEl) {
        if (!mediaEl) return;
        // Chỉ chạy spectrum 2D cho audio player (video đã có hình ảnh trực quan)
        if (mediaEl.tagName === "VIDEO") return;
        this._discoverCanvases();
        try {
            const ctx = this._ensureAudioContext();
            if (!ctx || !this.analyser) return;

            let src = this.sourceMap.get(mediaEl);
            if (!src) {
                src = ctx.createMediaElementSource(mediaEl);
                src.connect(this.analyser);
                this.analyser.connect(ctx.destination);
                this.sourceMap.set(mediaEl, src);
            }
            this.isActive = true;
            this.state = "audio-reactive";
            this._startLoop();
        } catch (e) {
            // Bỏ qua an toàn nếu trình duyệt hạn chế MediaElementSource trên cùng phần tử
        }
    }

    connectStream(mediaStream) {
        if (!mediaStream) return;
        this._discoverCanvases();
        try {
            const ctx = this._ensureAudioContext();
            if (!ctx || !this.analyser) return;
            if (this.streamSource) {
                try {
                    this.streamSource.disconnect();
                } catch (_) {}
            }
            this.streamSource = ctx.createMediaStreamSource(mediaStream);
            this.streamSource.connect(this.analyser);
            this.isActive = true;
            this.state = "audio-reactive";
            this._startLoop();
        } catch (e) {
            // Bỏ qua an toàn nếu không tạo được analyser cho stream
        }
    }

    disconnect() {
        this.isActive = false;
        this.state = "idle";
        if (this.streamSource) {
            try {
                this.streamSource.disconnect();
            } catch (_) {}
            this.streamSource = null;
        }
        if (this.animId) {
            cancelAnimationFrame(this.animId);
            this.animId = null;
        }
        this.smoothedBars.fill(0);
        this._drawStatic();
    }

    setState(nextState) {
        this.state = nextState || "idle";
        if (this.state === "idle") {
            this._drawStatic();
        }
    }

    setProcessingState(isProcessing) {
        this.state = isProcessing ? "processing" : "idle";
    }

    setCompletedState(hexColor) {
        if (hexColor) this.accentColor = hexColor;
        this.state = "completed";
        this._drawStatic();
    }

    _startLoop() {
        if (this.animId) return;
        const tick = () => {
            if (!this.isActive) {
                this.animId = null;
                this._drawStatic();
                return;
            }
            this._drawLiveFrame();
            this.animId = requestAnimationFrame(tick);
        };
        this.animId = requestAnimationFrame(tick);
    }

    _renderToCanvas(canvas, barHeights) {
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        if (rect.width < 10 || rect.height < 10) return;

        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const targetW = Math.round(rect.width * dpr);
        const targetH = Math.round(rect.height * dpr);
        if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
        }

        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.save();
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, rect.width, rect.height);

        const midY = rect.height / 2;

        // Đường trục ngang mảnh kiểu bản ghi âm học lưu trữ
        ctx.strokeStyle = "#E5E0D5";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(rect.width, midY);
        ctx.stroke();

        const count = barHeights.length;
        const padX = 6;
        const usableW = Math.max(20, rect.width - padX * 2);
        const step = usableW / Math.max(1, count - 1);
        const maxHalfH = (rect.height - 8) / 2;

        for (let i = 0; i < count; i++) {
            const amp = barHeights[i];
            const x = padX + i * step;
            const halfH = Math.max(1.5, amp * maxHalfH);
            const isPeak = amp > 0.42;

            ctx.fillStyle = isPeak
                ? this.accentColor
                : amp > 0.08
                  ? "#2A2825"
                  : "#CFC9BC";
            ctx.fillRect(x - 1, midY - halfH, 2, halfH * 2);
        }

        ctx.restore();
    }

    _drawStatic() {
        this._discoverCanvases();
        const idleBars = new Float32Array(this.barCount);
        for (let i = 0; i < this.barCount; i++) {
            // Nhịp vạch tĩnh cực mảnh, thanh lịch khi chưa phát
            idleBars[i] = 0.06 + 0.04 * Math.sin(i * 0.45);
        }
        this.canvases.forEach((c) => this._renderToCanvas(c, idleBars));
    }

    _drawLiveFrame() {
        if (!this.analyser || !this.freqData) return;
        this.analyser.getByteFrequencyData(this.freqData);

        const half = (this.barCount - 1) / 2;
        const maxBin = Math.max(8, Math.floor(this.freqData.length * 0.55));
        for (let i = 0; i < this.barCount; i++) {
            // Đối xứng từ tâm ra hai mép giúp dải sóng giọng nói cân đối và nhạy theo biên độ thực
            const distNorm = Math.abs(i - half) / Math.max(1, half);
            const binIdx = Math.min(this.freqData.length - 1, Math.floor(distNorm * maxBin));
            const raw = (this.freqData[binIdx] || 0) / 255.0;
            // Đường cong cảm nhận nhẹ giúp giọng nói/hát thông thường nảy rõ nhưng khoảng lặng vẫn phẳng
            const shaped = raw < 0.03 ? 0 : Math.min(1, Math.pow(raw, 0.78) * 1.25);
            this.smoothedBars[i] = this.smoothedBars[i] * 0.62 + shaped * 0.38;
        }

        this.canvases.forEach((c) => this._renderToCanvas(c, this.smoothedBars));
    }
}

window.AudioWaveRibbon = AudioWaveRibbon;

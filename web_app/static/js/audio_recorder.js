/**
 * Heritage XAI - Live Microphone Audio Recorder & Stream Bridge
 * -------------------------------------------------------------
 * Thu âm trực tiếp qua navigator.mediaDevices.getUserMedia,
 * kết nối luồng MediaStream sang Dải sóng hạt 3D (Three.js),
 * đếm thời gian thực (yêu cầu tối thiểu 5s, khuyến nghị 6-15s),
 * chuyển đổi buffer âm thanh thành định dạng .wav chuẩn (16kHz, Mono, 16-bit PCM).
 */

class LiveAudioRecorder {
  constructor(options = {}) {
    this.timerEl = document.getElementById(options.timerId || "mic-timer");
    this.ringEl = document.getElementById(options.ringId || "mic-timer-ring");
    this.statusTextEl = document.getElementById(options.statusId || "mic-status-text");

    this.mediaStream = null;
    this.audioContext = null;
    this.scriptProcessor = null;
    this.sourceNode = null;

    this.isRecording = false;
    this.startTime = 0;
    this.timerInterval = null;
    this.recordedSamples = [];
    this.totalSamplesCount = 0;

    this.targetSampleRate = 16000;
  }

  async startRecording(onStreamConnected = null) {
    if (this.isRecording) return;

    try {
      // Cấu hình chất lượng cao chuẩn phòng thu cho âm nhạc & thanh nhạc di sản:
      // Tắt các bộ lọc hội thoại (echoCancellation, noiseSuppression, autoGainControl)
      // để bảo toàn nguyên vẹn âm sắc nhạc cụ dân tộc (phách, đàn đáy, nhị, trống) và luyến láy.
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioContextClass();
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }
      this.actualSampleRate = this.audioContext.sampleRate || 44100;

      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

      // Kết nối luồng mic sang Quả cầu 3D Three.js nếu được cung cấp callback
      if (onStreamConnected) {
        onStreamConnected(this.mediaStream);
      }

      // Thu thập dữ liệu PCM raw qua ScriptProcessorNode
      const bufferSize = 4096;
      this.scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

      this.recordedSamples = [];
      this.totalSamplesCount = 0;

      this.scriptProcessor.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const inputData = e.inputBuffer.getChannelData(0);
        // Lưu trữ bản sao Float32Array
        this.recordedSamples.push(new Float32Array(inputData));
        this.totalSamplesCount += inputData.length;
      };

      this.sourceNode.connect(this.scriptProcessor);

      // Triệt tiêu hoàn toàn vòng lặp dội âm phản hồi (Acoustic Feedback Loop) ra loa máy tính
      // bằng GainNode gain = 0, ngăn tiếng vang rè kim loại khi người dùng không đeo tai nghe.
      this.muteGain = this.audioContext.createGain();
      this.muteGain.gain.value = 0;
      this.scriptProcessor.connect(this.muteGain);
      this.muteGain.connect(this.audioContext.destination);

      this.isRecording = true;
      this.startTime = Date.now();

      if (this.ringEl) this.ringEl.classList.add("recording");
      if (this.statusTextEl) this.statusTextEl.innerText = "Đang thu âm chuẩn phòng thu... (Nói/hát vào mic)";

      // Cập nhật đồng hồ đếm giây (00:00)
      this.timerInterval = setInterval(() => {
        const elapsedSec = (Date.now() - this.startTime) / 1000;
        const mins = Math.floor(elapsedSec / 60);
        const secs = Math.floor(elapsedSec % 60);
        if (this.timerEl) {
          this.timerEl.innerText = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
        }
      }, 100);

      return true;
    } catch (err) {
      console.error("[LiveAudioRecorder] Lỗi truy cập Microphone:", err);
      alert("Không thể truy cập Microphone. Vui lòng cấp quyền Microphone trong trình duyệt của bạn.");
      return false;
    }
  }

  stopRecording() {
    if (!this.isRecording) return null;

    this.isRecording = false;
    clearInterval(this.timerInterval);

    const elapsedSeconds = (Date.now() - this.startTime) / 1000;

    if (this.ringEl) this.ringEl.classList.remove("recording");
    if (this.statusTextEl) this.statusTextEl.innerText = `Đã ghi ${elapsedSeconds.toFixed(1)}s (Chất lượng cao)`;

    // Ngắt kết nối audio nodes
    if (this.scriptProcessor) {
      try { this.scriptProcessor.disconnect(); } catch (e) {}
    }
    if (this.muteGain) {
      try { this.muteGain.disconnect(); } catch (e) {}
    }
    if (this.sourceNode) {
      try { this.sourceNode.disconnect(); } catch (e) {}
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
    }

    // Ghép tất cả các chunk Float32 thành 1 buffer duy nhất
    const mergedBuffer = new Float32Array(this.totalSamplesCount);
    let offset = 0;
    for (const chunk of this.recordedSamples) {
      mergedBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    // Bảo toàn tần số lấy mẫu gốc (44.1kHz hoặc 48kHz) chuẩn phòng thu,
    // loại bỏ thuật toán hạ mẫu tuyến tính (Linear Interpolation) gây méo gấp phổ (Aliasing).
    const sampleRate = this.actualSampleRate || 44100;

    // Đóng gói thành Blob WAV chuẩn 16-bit PCM Mono
    const wavBlob = this.encodeWAV(mergedBuffer, sampleRate);
    const audioFile = new File([wavBlob], `mic_recording_${Date.now()}.wav`, { type: "audio/wav" });

    return {
      file: audioFile,
      blob: wavBlob,
      duration: elapsedSeconds
    };
  }

  /**
   * Resample tuyến tính đơn giản từ origRate -> targetRate
   */
  resampleAudio(audioBuffer, origRate, targetRate) {
    if (origRate === targetRate) return audioBuffer;

    const ratio = origRate / targetRate;
    const newLength = Math.round(audioBuffer.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const origIndex = i * ratio;
      const lower = Math.floor(origIndex);
      const upper = Math.min(lower + 1, audioBuffer.length - 1);
      const frac = origIndex - lower;

      result[i] = audioBuffer[lower] * (1 - frac) + audioBuffer[upper] * frac;
    }

    return result;
  }

  /**
   * Đóng gói Float32Array thành định dạng chuẩn RIFF WAV 16-bit PCM Mono
   */
  encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    this.writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    this.writeString(view, 8, "WAVE");

    // FMT sub-chunk
    this.writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);          // SubChunk1Size (16 for PCM)
    view.setUint16(20, 1, true);           // AudioFormat (1 for PCM)
    view.setUint16(22, 1, true);           // NumChannels (1: Mono)
    view.setUint32(24, sampleRate, true);  // SampleRate
    view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    view.setUint16(32, 2, true);           // BlockAlign (NumChannels * BitsPerSample/8)
    view.setUint16(34, 16, true);          // BitsPerSample (16 bits)

    // Data sub-chunk
    this.writeString(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);

    // Ghi dữ liệu 16-bit PCM (chuyển đổi từ [-1.0, 1.0] sang [-32768, 32767])
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      const val = s < 0 ? s * 0x8000 : s * 0x7FFF;
      view.setInt16(offset, val, true);
      offset += 2;
    }

    return new Blob([view], { type: "audio/wav" });
  }

  writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
}

// Global instance helper
window.LiveAudioRecorder = LiveAudioRecorder;

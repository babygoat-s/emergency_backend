// Standalone browser integration example; no UI or storage is included.
// The React/Vite app has its own client in emergency_frontend/src/audio/realtime.js.
// Microphone access requires localhost or HTTPS.
export async function startRealtimeAnalysis({
  url = "ws://127.0.0.1:8000/api/v1/analysis/realtime",
  onEvent = () => {},
  onError = console.error,
} = {}) {
  const ws = new WebSocket(url);
  let context, stream, source, node;
  let stopped = false;
  let finished;
  const finishedPromise = new Promise((resolve) => { finished = resolve; });

  const cleanup = () => {
    stopped = true;
    stream?.getTracks().forEach((track) => track.stop());
    source?.disconnect();
    node?.disconnect();
    if (context && context.state !== "closed") context.close().catch(() => {});
  };

  try {
    await new Promise((resolve, reject) => {
      ws.onopen = () => ws.send(JSON.stringify({
        type: "start", encoding: "pcm_s16le", sample_rate: 16000, channels: 1, language: "ko",
      }));
      ws.onerror = () => reject(new Error("WebSocket 연결 실패"));
      ws.onclose = () => { cleanup(); finished(); reject(new Error("연결 종료")); };
      ws.onmessage = ({ data }) => {
        const event = JSON.parse(data);
        onEvent(event);
        if (event.type === "ready") resolve();
        if (event.type === "finished") finished();
        if (event.type === "error") {
          const error = new Error(event.error.message);
          onError(error);
          reject(error);
        }
      };
    });

    context = new AudioContext({ sampleRate: 16000 });
    if (context.sampleRate !== 16000) throw new Error("16 kHz AudioContext가 필요합니다.");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true },
    });
    if (stopped || ws.readyState !== WebSocket.OPEN) throw new Error("연결이 종료되었습니다.");
    const workletSource = `
      class PCMProcessor extends AudioWorkletProcessor {
        constructor() {
          super();
          this.reset();
          this.stopped = false;
          this.port.onmessage = ({ data }) => {
            if (data === "flush") {
              if (this.offset) this.emit();
              this.stopped = true;
              this.port.postMessage("flushed");
            }
          };
        }
        reset() { this.buffer = new ArrayBuffer(3200); this.view = new DataView(this.buffer); this.offset = 0; }
        emit() {
          const data = this.buffer.slice(0, this.offset * 2);
          this.port.postMessage(data, [data]);
          this.reset();
        }
        process(inputs) {
          if (this.stopped) return false;
          const channels = inputs[0];
          if (!channels?.length) return true;
          for (let i = 0; i < channels[0].length; i++) {
            let sample = 0;
            for (const channel of channels) sample += channel[i] / channels.length;
            sample = Math.max(-1, Math.min(1, sample));
            this.view.setInt16(this.offset++ * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
            if (this.offset === 1600) this.emit();
          }
          return true;
        }
      }
      registerProcessor("pcm-processor", PCMProcessor);
    `;
    const moduleUrl = URL.createObjectURL(new Blob([workletSource], { type: "text/javascript" }));
    try { await context.audioWorklet.addModule(moduleUrl); }
    finally { URL.revokeObjectURL(moduleUrl); }
    if (stopped || ws.readyState !== WebSocket.OPEN) throw new Error("연결이 종료되었습니다.");
    node = new AudioWorkletNode(context, "pcm-processor");
    let flushed;
    const flushedPromise = new Promise((resolve) => { flushed = resolve; });
    node.port.onmessage = ({ data }) => {
      if (data === "flushed") { flushed(); return; }
      if (ws.readyState !== WebSocket.OPEN) return;
      if (ws.bufferedAmount > 320000) {
        onError(new Error("음성 전송이 지연되어 녹음을 중단했습니다."));
        cleanup();
        ws.close();
        return;
      }
      ws.send(data); // 100 ms raw PCM frames. Do not wait for analysis results.
    };
    source = context.createMediaStreamSource(stream);
    source.connect(node);
    // Processor output is silent. Connecting keeps the worklet running without mic feedback.
    node.connect(context.destination);
    await context.resume();

    return {
      async stop() {
        if (!stopped && ws.readyState === WebSocket.OPEN) {
          node.port.postMessage("flush");
          await flushedPromise;
          cleanup();
          ws.send(JSON.stringify({ type: "finish" }));
        }
        await finishedPromise;
      },
      cancel() { cleanup(); ws.close(); },
    };
  } catch (error) {
    cleanup();
    ws.close();
    throw error;
  }
}

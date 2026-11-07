'use strict';

// Page-only values. The server owns everything about detection, and clamps the delay itself.
const BEEP_HZ = 880;
const BEEP_ON_MS = 300;
const BEEP_OFF_MS = 200;
const BEEP_PEAK_GAIN = 0.3;
const OVERLAY_LINE_FRAC = 0.003;
const OVERLAY_LETTER_FRAC = 0.045;
const BOX_COLOUR = '#22c55e';
const ALERT_BOX_COLOUR = '#ef4444';
const FRAME_WIDTH = 640;
const FRAME_HEIGHT = 480;
const JPEG_QUALITY = 0.8;
const FRAME_GAP_MS = 1000 / 15;
const WARNING_MS = 5000;
const PAUSED_NOTE_MS = 3000;

const LOADING_MESSAGE = 'Loading...';
const SOUND_ERROR_MESSAGE = 'The alarm sound could not be started. Reload the page to try again.';
const PAUSED_MESSAGE = 'Detection paused while this tab was hidden';
const LOST_MESSAGE = 'Connection to the app was lost. Check it is running, then reload.';
const BUSY_MESSAGE = 'The app is already open in another tab.';
const OVERFLOW_MESSAGE = 'Showing the 4 largest faces';
const CAPTURE_WARNING = 'Skipped a camera frame that could not be captured.';
const REPLY_WARNING = 'The app sent a reply that could not be read.';
const DEBUG_LOG_ERROR_MESSAGE = 'The debug log could not be saved. Please try again.';

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const startButton = document.getElementById('start');
const message = document.getElementById('message');
const strip = document.getElementById('strip');
const debugPanel = document.getElementById('debug-panel');
const debug = document.getElementById('debug');
const debugDownloadButton = document.getElementById('debug-download');
const logList = document.getElementById('log');
const muteButton = document.getElementById('mute');
const volumeSlider = document.getElementById('volume');
const volumeValue = document.getElementById('volume-value');
const delaySlider = document.getElementById('delay');
const delayValue = document.getElementById('delay-value');
const snapshotButton = document.getElementById('snapshot');

const debugOn = new URLSearchParams(window.location.search).get('debug') === '1';
debugPanel.hidden = !debugOn;

// Off-screen, so the frames sent to the server are the unflipped video.
const capture = document.createElement('canvas');
capture.width = FRAME_WIDTH;
capture.height = FRAME_HEIGHT;

class Beeper {
  constructor() {
    this.ctx = new AudioContext();
    this.gain = this.ctx.createGain();
    this.gain.gain.value = 0;
    this.gain.connect(this.ctx.destination);
    this.osc = null;
    this.timer = null;
    this.sounding = false;
    this.volume = 1;
    this.ctx.resume().catch(() => {});
  }

  setVolume(v) {
    this.volume = Math.min(1, Math.max(0, v));
    if (this.osc && this.timer !== null && this.gain.gain.value > 0) {
      this.gain.gain.value = this.volume * BEEP_PEAK_GAIN;
    }
  }

  setSounding(on) {
    if (on === this.sounding) return;
    this.sounding = on;
    if (on) this.start();
    else this.stop();
  }

  start() {
    try {
      this.ctx.resume().catch(() => {});
      const osc = this.ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.value = BEEP_HZ;
      osc.connect(this.gain);
      osc.start();
      this.osc = osc;
      this.beepOn();
    } catch (err) {
      console.error('Alarm sound failed', err);
      this.sounding = false;
    }
  }

  beepOn() {
    this.gain.gain.value = this.volume * BEEP_PEAK_GAIN;
    this.timer = window.setTimeout(() => this.beepOff(), BEEP_ON_MS);
  }

  beepOff() {
    this.gain.gain.value = 0;
    this.timer = window.setTimeout(() => this.beepOn(), BEEP_OFF_MS);
  }

  stop() {
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    this.gain.gain.value = 0;
    if (this.osc) {
      try {
        this.osc.stop();
      } catch {
        // already stopped
      }
      this.osc.disconnect();
      this.osc = null;
    }
  }
}

let beeper = null;
let socket = null;
let stream = null;
let running = false; // the camera is on and frames may be sent
let sessionOver = false;
let frameInFlight = false; // captured or sent, with no reply yet
let frameTimer = null;
let lastSentMs = -Infinity;
let alarmOn = false;
let muted = false;
let lastError = null; // an error with no frame since: if the socket closes now, this is why
let shownLog = '';
let messageTimer = null;

function showMessage(text) {
  window.clearTimeout(messageTimer);
  message.textContent = text ?? '';
  message.hidden = text === null;
}

function showBriefly(text, ms) {
  showMessage(text);
  messageTimer = window.setTimeout(() => showMessage(null), ms);
}

function cameraErrorMessage(err) {
  switch (err?.name) {
    case 'NotAllowedError':
      return 'Camera access was blocked. Please allow camera access in the browser and reload the page.';
    case 'NotFoundError':
      return 'No camera found. Connect a camera and reload the page.';
    case 'NotReadableError':
      return 'The camera is in use by another app. Close it and reload the page.';
    default:
      return 'The camera could not be started. Check it is connected and reload the page.';
  }
}

function showVolume() {
  volumeValue.textContent = `${Math.round(volumeSlider.valueAsNumber * 100)}%`;
}

function showDelay() {
  delayValue.textContent = `${delaySlider.valueAsNumber}s`;
}

function updateSound() {
  beeper?.setSounding(alarmOn && !muted && !document.hidden);
}

/** Draws only a thin box and a small letter per face. Faces hidden this frame get nothing. */
function drawOverlay(people) {
  if (overlay.width !== video.videoWidth) overlay.width = video.videoWidth;
  if (overlay.height !== video.videoHeight) overlay.height = video.videoHeight;
  const ctx = overlay.getContext('2d');
  if (!ctx) return;
  const { width, height } = overlay;
  ctx.clearRect(0, 0, width, height);

  const lineWidth = height * OVERLAY_LINE_FRAC;
  const letterSize = height * OVERLAY_LETTER_FRAC;
  ctx.lineWidth = lineWidth;
  ctx.font = `600 ${letterSize}px 'Segoe UI', system-ui, sans-serif`;
  ctx.textAlign = 'left';

  for (const person of people) {
    if (!person.visible) continue;
    const { box } = person;
    const colour = person.red ? ALERT_BOX_COLOUR : BOX_COLOUR;
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;

    // The video is mirrored by CSS and this canvas is not, so flip the position here.
    const x = (1 - box.x - box.width) * width;
    const y = box.y * height;
    ctx.strokeRect(x, y, box.width * width, box.height * height);

    const fitsAbove = y >= letterSize + lineWidth;
    ctx.textBaseline = fitsAbove ? 'bottom' : 'top';
    // A face still being identified shows ?, the same as its snap.
    ctx.fillText(person.letter ?? '?', x, fitsAbove ? y - lineWidth : y + lineWidth);
  }
}

function renderStrip(reply) {
  const count = document.createElement('div');
  count.className = 'strip-count';
  count.textContent = reply.count_text;

  const lines = document.createElement('div');
  lines.className = 'strip-people';
  reply.lines.forEach((text, i) => {
    const line = document.createElement('div');
    line.className = reply.people[i]?.red ? 'strip-person alert' : 'strip-person';
    line.textContent = text;
    lines.append(line);
  });
  strip.replaceChildren(count, lines);

  if (reply.overflow) {
    const note = document.createElement('div');
    note.className = 'strip-overflow';
    note.textContent = OVERFLOW_MESSAGE;
    strip.append(note);
  }
}

/** The server sends the whole panel each frame. Only touch the list when it changed. */
function renderLog(lines) {
  const joined = lines.join('\n');
  if (joined === shownLog) return;
  shownLog = joined;
  logList.replaceChildren(
    ...lines.map((text) => {
      const item = document.createElement('li');
      item.textContent = text;
      return item;
    }),
  );
}

function send(data) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(data));
}

function scheduleFrame() {
  if (!running || document.hidden || frameInFlight || frameTimer !== null) return;
  const wait = Math.max(0, lastSentMs + FRAME_GAP_MS - performance.now());
  frameTimer = window.setTimeout(sendFrame, wait);
}

function sendFrame() {
  frameTimer = null;
  if (!running || document.hidden) return;
  lastSentMs = performance.now();
  if (video.readyState < video.HAVE_CURRENT_DATA) {
    scheduleFrame();
    return;
  }
  frameInFlight = true;
  try {
    capture.getContext('2d').drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    capture.toBlob(onFrameCaptured, 'image/jpeg', JPEG_QUALITY);
  } catch (err) {
    console.error('Frame capture failed', err);
    showBriefly(CAPTURE_WARNING, WARNING_MS);
    frameInFlight = false;
    scheduleFrame();
  }
}

function onFrameCaptured(blob) {
  if (blob && running && socket?.readyState === WebSocket.OPEN) {
    socket.send(blob);
    return;
  }
  frameInFlight = false;
  scheduleFrame();
}

function onFrameReply(reply) {
  lastError = null;
  if (!running) return;
  drawOverlay(reply.people);
  renderStrip(reply);
  renderLog(reply.log);
  if (debugOn) debug.textContent = (reply.debug ?? []).join('\n');
  if (reply.warning) showBriefly(reply.warning, WARNING_MS);
  alarmOn = reply.alarm;
  updateSound();
}

function saveDebugLog(samples) {
  const savedAt = new Date();
  const json = JSON.stringify({ savedAt: savedAt.toISOString(), samples }, null, 1);
  const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
  try {
    const link = document.createElement('a');
    link.href = url;
    link.download = `debug-log-${savedAt.toISOString().replace(/[:.]/g, '-')}.json`;
    link.click();
  } finally {
    // The download has its own hold on the data, so the URL can go on the next turn.
    window.setTimeout(() => URL.revokeObjectURL(url));
  }
}

function stopCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
}

/** Ends the page's side of the session and leaves one lasting message on screen. */
function endSession(text) {
  sessionOver = true;
  running = false;
  window.clearTimeout(frameTimer);
  frameTimer = null;
  alarmOn = false;
  updateSound();
  stopCamera();
  snapshotButton.disabled = true;
  debugDownloadButton.disabled = true;
  overlay.getContext('2d')?.clearRect(0, 0, overlay.width, overlay.height);
  strip.replaceChildren();
  showMessage(text);
  socket?.close();
}

async function startCamera() {
  try {
    const opened = await navigator.mediaDevices.getUserMedia({
      video: { width: FRAME_WIDTH, height: FRAME_HEIGHT },
      audio: false,
    });
    if (sessionOver) {
      opened.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = opened;
    for (const track of stream.getVideoTracks()) {
      track.addEventListener('ended', () => {
        if (!sessionOver) endSession(cameraErrorMessage(new Error('Camera ended')));
      });
    }
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    console.error('Camera failed to start', err);
    if (!sessionOver) endSession(cameraErrorMessage(err));
    return;
  }
  if (sessionOver) return; // ended while starting; its message is already showing

  showMessage(null);
  running = true;
  snapshotButton.disabled = false;
  debugDownloadButton.disabled = false;
  scheduleFrame();
}

function onSocketMessage(event) {
  let reply;
  try {
    reply = JSON.parse(event.data);
    switch (reply.type) {
      case 'ready':
        send({ type: 'settings', delaySec: delaySlider.valueAsNumber });
        void startCamera();
        break;
      case 'frame':
        frameInFlight = false;
        onFrameReply(reply);
        scheduleFrame();
        break;
      case 'busy':
        endSession(BUSY_MESSAGE);
        break;
      case 'error':
        lastError = reply.message;
        showBriefly(reply.message, WARNING_MS);
        break;
      case 'debug_dump':
        saveDebugLog(reply.samples);
        break;
      default:
        console.error('Unknown reply from the app', reply.type);
    }
  } catch (err) {
    console.error('Reply failed', err);
    showBriefly(reply?.type === 'debug_dump' ? DEBUG_LOG_ERROR_MESSAGE : REPLY_WARNING, WARNING_MS);
    if (reply?.type !== 'debug_dump') {
      frameInFlight = false; // the only reply that releases the next frame may be this one
      scheduleFrame();
    }
  }
}

function openSocket() {
  try {
    socket = new WebSocket(`ws://${window.location.host}/ws`);
  } catch (err) {
    console.error('Socket failed to open', err);
    endSession(LOST_MESSAGE);
    return;
  }
  socket.addEventListener('open', () => send({ type: 'start', debug: debugOn }));
  socket.addEventListener('message', onSocketMessage);
  socket.addEventListener('close', () => {
    if (!sessionOver) endSession(lastError ?? LOST_MESSAGE);
  });
}

function start() {
  startButton.hidden = true;
  try {
    beeper = new Beeper();
    beeper.setVolume(volumeSlider.valueAsNumber);
  } catch (err) {
    console.error('Alarm sound failed', err);
    showMessage(SOUND_ERROR_MESSAGE);
    return;
  }
  showMessage(LOADING_MESSAGE);
  openSocket();
}

/**
 * Frames stop while the tab is hidden but the beeper runs on timers, so the alarm
 * would stick on. Frames after return turn it back on if it is still needed.
 */
function onVisibilityChange() {
  if (document.hidden) {
    alarmOn = false;
    updateSound();
    return;
  }
  if (!running) return;
  showBriefly(PAUSED_MESSAGE, PAUSED_NOTE_MS);
  scheduleFrame();
}

startButton.addEventListener('click', start);
document.addEventListener('visibilitychange', onVisibilityChange);

muteButton.addEventListener('click', () => {
  muted = !muted;
  muteButton.textContent = muted ? 'Unmute' : 'Mute';
  muteButton.setAttribute('aria-pressed', String(muted));
  updateSound();
});

volumeSlider.addEventListener('input', () => {
  beeper?.setVolume(volumeSlider.valueAsNumber);
  showVolume();
});

delaySlider.addEventListener('input', () => {
  showDelay();
  send({ type: 'settings', delaySec: delaySlider.valueAsNumber });
});

snapshotButton.addEventListener('click', () => send({ type: 'snapshot' }));
debugDownloadButton.addEventListener('click', () => send({ type: 'debug_dump' }));

showVolume();
showDelay();

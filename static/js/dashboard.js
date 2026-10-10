/**
 * Dashboard Client Controller & Web Audio Alert Synthesizer
 * Vision-Based Distracted Driver Detection System
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // 1. STATE & GLOBAL VARIABLES
    // -------------------------------------------------------------
    let socket = null;
    let audioEnabled = true;
    let audioContext = null;
    let lastAudioTriggerTime = 0;
    const AUDIO_COOLDOWN_MS = 1200; // 1.2 second minimum interval between chimes

    let alertChart = null;
    let prevDriverStatus = 'SAFE';
    let prevAlertMsg = '';

    // Continuous / Repeating Alarm state
    let repeatingAlarmTimer = null;
    let currentAlarmSeverity = null;
    let currentAlarmMsg = '';

    // Polling fallback state
    let isPolling = false;
    let pollingIntervalId = null;

    // -------------------------------------------------------------
    // 2. INITIALIZATION
    // -------------------------------------------------------------
    initClock();
    initSocketIO();
    initChart();
    fetchAlertHistory();
    fetchStatistics();
    bindUIEvents();

    // Always poll API statistics & history every 5 seconds
    setInterval(fetchAlertHistory, 5000);
    setInterval(fetchStatistics, 5000);

    // Initial check for REST API telemetry if socket takes time
    setTimeout(() => {
        if (!socket || !socket.connected) {
            startPollingTelemetry();
        }
    }, 1500);

    // -------------------------------------------------------------
    // 3. LIVE CLOCK
    // -------------------------------------------------------------
    function initClock() {
        const clockEl = document.getElementById('live-clock');
        if (!clockEl) return;
        function updateTime() {
            const now = new Date();
            clockEl.textContent = now.toLocaleTimeString();
        }
        updateTime();
        setInterval(updateTime, 1000);
    }

    // -------------------------------------------------------------
    // 4. WEB AUDIO API SYNTHESIZER & REPEATING SPEECH ALERTS
    // -------------------------------------------------------------
    window.unlockAudioContextBanner = function() {
        unlockAudioContext();
        const banner = document.getElementById('audio-unlock-banner');
        if (banner) {
            banner.classList.add('d-none');
            banner.classList.remove('d-flex');
        }
        logEvent('Audio safety siren & voice alerts activated.', 'success');
    };

    function unlockAudioContext() {
        if (!audioContext) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                audioContext = new AudioCtx();
            }
        }
        if (audioContext) {
            if (audioContext.state === 'suspended') {
                audioContext.resume().then(() => {
                    const banner = document.getElementById('audio-unlock-banner');
                    if (banner) {
                        banner.classList.add('d-none');
                        banner.classList.remove('d-flex');
                    }
                }).catch(() => {});
            } else {
                const banner = document.getElementById('audio-unlock-banner');
                if (banner) {
                    banner.classList.add('d-none');
                    banner.classList.remove('d-flex');
                }
            }
        }
        return audioContext;
    }

    // Auto-unlock audio on any user interaction with page
    document.addEventListener('click', unlockAudioContext, { once: false });
    document.addEventListener('keydown', unlockAudioContext, { once: false });
    document.addEventListener('touchstart', unlockAudioContext, { once: false });

    function speakVoiceAlert(text) {
        if (!audioEnabled || !('speechSynthesis' in window)) return;
        try {
            window.speechSynthesis.cancel(); // Stop prior speech
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.15;
            utterance.pitch = 1.1;
            utterance.volume = 1.0;
            window.speechSynthesis.speak(utterance);
        } catch (e) {
            console.warn("Speech synthesis error:", e);
        }
    }

    function playAudioAlertOnce(severity, alertMessage = '') {
        if (!audioEnabled) return;
        
        const now = Date.now();
        if (now - lastAudioTriggerTime < AUDIO_COOLDOWN_MS) {
            return;
        }
        lastAudioTriggerTime = now;

        const ctx = unlockAudioContext();
        if (ctx && ctx.state === 'running') {
            if (severity === 'WARNING') {
                playWarningBeep(ctx);
            } else if (severity === 'CRITICAL') {
                playCriticalAlarm(ctx);
            }
        } else {
            // Show audio unlock banner if blocked by browser policy
            const banner = document.getElementById('audio-unlock-banner');
            if (banner) {
                banner.classList.remove('d-none');
                banner.classList.add('d-flex');
            }
        }

        // Voice alert
        if (alertMessage) {
            if (severity === 'CRITICAL') {
                speakVoiceAlert("Warning! " + alertMessage);
            } else if (severity === 'WARNING') {
                speakVoiceAlert(alertMessage);
            }
        }
    }

    function startRepeatingAlarm(severity, alertMessage) {
        stopRepeatingAlarm();
        currentAlarmSeverity = severity;
        currentAlarmMsg = alertMessage;

        // Play first burst immediately
        playAudioAlertOnce(severity, alertMessage);

        // Repeat periodically so driver is continuously warned
        const intervalMs = severity === 'CRITICAL' ? 1500 : 2500;
        repeatingAlarmTimer = setInterval(() => {
            if (audioEnabled && currentAlarmSeverity) {
                playAudioAlertOnce(currentAlarmSeverity, currentAlarmMsg);
            }
        }, intervalMs);
    }

    function stopRepeatingAlarm() {
        if (repeatingAlarmTimer) {
            clearInterval(repeatingAlarmTimer);
            repeatingAlarmTimer = null;
        }
        currentAlarmSeverity = null;
        currentAlarmMsg = '';
    }

    function playWarningBeep(ctx) {
        try {
            // High-visibility dual warning chime (900Hz -> 1250Hz)
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            
            const startTime = ctx.currentTime;
            osc.frequency.setValueAtTime(900, startTime);
            osc.frequency.setValueAtTime(1250, startTime + 0.15);

            gain.gain.setValueAtTime(0.01, startTime);
            gain.gain.linearRampToValueAtTime(0.5, startTime + 0.05);
            gain.gain.linearRampToValueAtTime(0.01, startTime + 0.35);

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start(startTime);
            osc.stop(startTime + 0.36);
        } catch (e) {
            console.warn("Audio warning synth failed:", e);
        }
    }

    function playCriticalAlarm(ctx) {
        try {
            // Loud piercing automotive siren alarm (4 rapid sweeping bursts)
            const bursts = 4;
            const startTime = ctx.currentTime;

            for (let i = 0; i < bursts; i++) {
                const burstStart = startTime + (i * 0.16);
                
                // Primary piercing siren
                const osc1 = ctx.createOscillator();
                const gain1 = ctx.createGain();
                osc1.type = 'sawtooth';
                osc1.frequency.setValueAtTime(800, burstStart);
                osc1.frequency.linearRampToValueAtTime(1350, burstStart + 0.12);

                gain1.gain.setValueAtTime(0.01, burstStart);
                gain1.gain.linearRampToValueAtTime(0.55, burstStart + 0.03);
                gain1.gain.linearRampToValueAtTime(0.01, burstStart + 0.13);

                osc1.connect(gain1);
                gain1.connect(ctx.destination);

                osc1.start(burstStart);
                osc1.stop(burstStart + 0.14);

                // Sub-tone for maximum acoustic impact
                const osc2 = ctx.createOscillator();
                const gain2 = ctx.createGain();
                osc2.type = 'square';
                osc2.frequency.setValueAtTime(450, burstStart);
                gain2.gain.setValueAtTime(0.2, burstStart);
                gain2.gain.linearRampToValueAtTime(0.01, burstStart + 0.13);

                osc2.connect(gain2);
                gain2.connect(ctx.destination);

                osc2.start(burstStart);
                osc2.stop(burstStart + 0.14);
            }
        } catch (e) {
            console.warn("Audio critical synth failed:", e);
        }
    }

    window.testAlarmSound = function() {
        const ctx = unlockAudioContext();
        if (ctx && ctx.state === 'running') {
            playCriticalAlarm(ctx);
        } else {
            alert("Audio context suspended. Please click anywhere on the page first!");
        }
        speakVoiceAlert("Alarm test successful. Live driver monitoring active.");
        logEvent("Test Alarm Triggered (Audio & Speech Synthesis OK)", "critical");
    };

    // -------------------------------------------------------------
    // 5. SOCKET.IO & REST API FALLBACK TELEMETRY POLLING
    // -------------------------------------------------------------
    function startPollingTelemetry() {
        if (isPolling) return;
        isPolling = true;
        console.log('[TELEMETRY] Active REST API polling fallback enabled.');
        pollingIntervalId = setInterval(async () => {
            try {
                const res = await fetch('/api/telemetry');
                if (res.ok) {
                    const telemetry = await res.json();
                    if (telemetry && Object.keys(telemetry).length > 0) {
                        updateDashboard(telemetry);
                    }
                }
            } catch (e) {
                // Silently ignore temporary fetch errors
            }
        }, 250);
    }

    function stopPollingTelemetry() {
        if (pollingIntervalId) {
            clearInterval(pollingIntervalId);
            pollingIntervalId = null;
        }
        isPolling = false;
    }

    function initSocketIO() {
        if (typeof io === 'undefined') {
            console.warn("[SOCKET] Socket.IO CDN library not loaded. Falling back to HTTP REST polling.");
            startPollingTelemetry();
            return;
        }

        try {
            socket = io({ transports: ['websocket', 'polling'], timeout: 3000, reconnectionDelay: 1000 });

            socket.on('connect', () => {
                logEvent('Connected to real-time WebSocket telemetry stream.', 'success');
                const statusEl = document.getElementById('system-status-text');
                if (statusEl) statusEl.textContent = 'SYSTEM ONLINE';
                stopPollingTelemetry();
            });

            socket.on('disconnect', () => {
                logEvent('WebSocket telemetry disconnected. Using REST polling fallback...', 'warning');
                const statusEl = document.getElementById('system-status-text');
                if (statusEl) statusEl.textContent = 'REST POLLING';
                startPollingTelemetry();
            });

            socket.on('connect_error', () => {
                startPollingTelemetry();
            });

            socket.on('telemetry_update', (telemetry) => {
                stopPollingTelemetry();
                updateDashboard(telemetry);
            });

            socket.on('processed_frame_response', (data) => {
                isSendingFrame = false;
                if (data && data.image) {
                    const imgEl = document.getElementById('live-video-stream');
                    if (imgEl) {
                        imgEl.src = data.image;
                        const errOverlay = document.getElementById('camera-error-overlay');
                        if (errOverlay) errOverlay.classList.add('d-none');
                        const camLabel = document.getElementById('cam-status-label');
                        if (camLabel) camLabel.innerHTML = '<i class="fa-solid fa-link text-success me-1"></i>CAM OK';
                    }
                }
                if (data && data.telemetry) {
                    updateDashboard(data.telemetry);
                }
                setTimeout(sendNextBrowserFrame, 25);
            });

            socket.on('alert_triggered', (telemetry) => {
                logEvent(`ALERT TRIGGERED: ${telemetry.alert} (${telemetry.severity})`, (telemetry.severity || 'info').toLowerCase());
                fetchAlertHistory();
                fetchStatistics();
            });
        } catch (e) {
            console.warn("[SOCKET] Socket.IO init error:", e);
            startPollingTelemetry();
        }

        // Auto-start browser webcam for cloud deployments (e.g. Render)
        setTimeout(() => {
            initBrowserWebcam();
        }, 1500);
    }

    // -------------------------------------------------------------
    // BROWSER WEBCAM CLIENT STREAMER (FOR CLOUD DEPLOYMENTS)
    // -------------------------------------------------------------
    let clientMediaStream = null;
    let clientWebcamActive = false;
    let isSendingFrame = false;

    async function initBrowserWebcam() {
        if (clientWebcamActive) return;
        try {
            let video = document.getElementById('client-webcam-element');
            if (!video) {
                video = document.createElement('video');
                video.id = 'client-webcam-element';
                video.setAttribute('autoplay', '');
                video.setAttribute('playsinline', '');
                video.setAttribute('muted', '');
                video.style.display = 'none';
                document.body.appendChild(video);
            }

            let canvas = document.getElementById('client-canvas-element');
            if (!canvas) {
                canvas = document.createElement('canvas');
                canvas.id = 'client-canvas-element';
                canvas.style.display = 'none';
                document.body.appendChild(canvas);
            }

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.warn('Browser webcam API not supported.');
                return;
            }

            clientMediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { max: 30 } }
            });
            video.srcObject = clientMediaStream;
            await video.play();
            clientWebcamActive = true;
            logEvent('Browser webcam active. Streaming live feed to AI engine...', 'success');
            
            const errOverlay = document.getElementById('camera-error-overlay');
            if (errOverlay) errOverlay.classList.add('d-none');

            sendNextBrowserFrame();
        } catch (err) {
            console.warn('Browser webcam access failed or denied:', err);
        }
    }

    let lastFrameSentTime = 0;

    function sendNextBrowserFrame() {
        const now = Date.now();
        if (isSendingFrame && now - lastFrameSentTime > 1500) {
            // Watchdog: reset stuck frame flag if network stalled
            isSendingFrame = false;
        }

        if (!clientWebcamActive || !socket || !socket.connected || isSendingFrame) {
            setTimeout(sendNextBrowserFrame, 60);
            return;
        }

        const video = document.getElementById('client-webcam-element');
        const canvas = document.getElementById('client-canvas-element');
        if (!video || !canvas || video.readyState !== 4) {
            setTimeout(sendNextBrowserFrame, 60);
            return;
        }

        isSendingFrame = true;
        lastFrameSentTime = now;
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 640, 480);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.50);

        socket.emit('process_browser_frame', dataUrl);
    }

    // -------------------------------------------------------------
    // 6. DASHBOARD DOM UPDATER
    // -------------------------------------------------------------
    function updateDashboard(t) {
        if (!t) return;

        // 1. FPS Badge
        const fpsBadge = document.getElementById('fps-badge');
        if (fpsBadge && t.fps !== undefined) {
            fpsBadge.textContent = `FPS: ${Math.round(t.fps)}`;
        }

        // 2. Driver Status Badge & Video Container Border Glow
        const statusBadge = document.getElementById('driver-status-badge');
        const statusDesc = document.getElementById('driver-status-desc');
        const videoContainer = document.getElementById('video-card-container');
        const hudBadge = document.getElementById('hud-badge-status');

        if (statusBadge) statusBadge.textContent = t.driver_status || 'SAFE';
        if (hudBadge) hudBadge.textContent = t.driver_status || 'SAFE';

        if (statusBadge && videoContainer && hudBadge) {
            statusBadge.classList.remove('status-badge-safe', 'status-badge-warning', 'status-badge-critical');
            videoContainer.classList.remove('border-state-safe', 'border-state-warning', 'border-state-critical');
            hudBadge.classList.remove('bg-success', 'bg-warning', 'bg-danger');

            if (t.driver_status === 'SAFE') {
                statusBadge.classList.add('status-badge-safe');
                videoContainer.classList.add('border-state-safe');
                hudBadge.classList.add('bg-success');
                if (statusDesc) statusDesc.textContent = 'Driver is attentive and alert';
            } else if (t.driver_status === 'WARNING') {
                statusBadge.classList.add('status-badge-warning');
                videoContainer.classList.add('border-state-warning');
                hudBadge.classList.add('bg-warning');
                if (statusDesc) statusDesc.textContent = 'Potential distraction / fatigue indicator detected';
            } else {
                statusBadge.classList.add('status-badge-critical');
                videoContainer.classList.add('border-state-critical');
                hudBadge.classList.add('bg-danger');
                if (statusDesc) statusDesc.textContent = 'CRITICAL: Driver distraction threshold exceeded!';
            }
        }

        // 3. Audio Alarm & Voice Siren Control
        if (t.severity === 'CRITICAL' || t.severity === 'WARNING') {
            if (t.driver_status !== prevDriverStatus || t.alert !== prevAlertMsg || !repeatingAlarmTimer) {
                startRepeatingAlarm(t.severity, t.alert);
            }
        } else {
            stopRepeatingAlarm();
        }

        // 4. Distraction Score (0-100)
        const scoreText = document.getElementById('distraction-score-text');
        const scoreBar = document.getElementById('score-progress-bar');
        const scoreBadge = document.getElementById('score-level-badge');

        const score = t.distraction_score || 0;
        if (scoreText) scoreText.innerHTML = `${score}<span class="fs-6 text-muted">/100</span>`;
        if (scoreBar) scoreBar.style.width = `${score}%`;
        if (scoreBadge) scoreBadge.textContent = t.distraction_level || 'Normal';

        if (scoreText && scoreBar) {
            scoreText.classList.remove('text-success', 'text-warning', 'text-danger');
            scoreBar.classList.remove('bg-success', 'bg-warning', 'bg-danger');

            if (score <= 40) {
                scoreText.classList.add('text-success');
                scoreBar.classList.add('bg-success');
            } else if (score <= 70) {
                scoreText.classList.add('text-warning');
                scoreBar.classList.add('bg-warning');
            } else {
                scoreText.classList.add('text-danger');
                scoreBar.classList.add('bg-danger');
            }
        }

        // 5. Eye Monitoring Card
        const valEyeState = document.getElementById('val-eye-state');
        const valEar = document.getElementById('val-ear');
        const valBlinks = document.getElementById('val-total-blinks');
        const valEyeTimer = document.getElementById('val-eye-timer');
        const barEyeTimer = document.getElementById('bar-eye-timer');

        if (valEyeState) valEyeState.textContent = t.eye_status || 'OPEN';
        if (valEar) valEar.textContent = t.avg_ear !== undefined ? Number(t.avg_ear).toFixed(2) : '0.00';
        if (valBlinks) valBlinks.textContent = t.total_blinks || 0;

        const eyeTime = t.eye_closure_time || 0;
        const eyeThresh = t.eye_closure_threshold || 2.0;
        const eyePct = Math.min(100, (eyeTime / eyeThresh) * 100);
        if (valEyeTimer) valEyeTimer.textContent = `${eyeTime.toFixed(2)} / ${eyeThresh.toFixed(2)} s`;
        if (barEyeTimer) {
            barEyeTimer.style.width = `${eyePct}%`;
            barEyeTimer.className = `progress-bar ${eyePct > 80 ? 'bg-danger' : eyePct > 40 ? 'bg-warning' : 'bg-success'}`;
        }

        // 6. Head Direction Card
        const valHeadDir = document.getElementById('val-head-dir');
        const valHeadYaw = document.getElementById('val-head-yaw');
        const valHeadPose = document.getElementById('val-head-pose');
        const valHeadTimer = document.getElementById('val-head-timer');
        const barHeadTimer = document.getElementById('bar-head-timer');

        if (valHeadDir) valHeadDir.textContent = t.head_direction || 'CENTER';
        if (valHeadYaw) valHeadYaw.textContent = `${t.yaw !== undefined ? Number(t.yaw).toFixed(1) : 0.0}°`;
        if (valHeadPose) valHeadPose.textContent = t.head_direction === 'CENTER' ? 'Aligned' : `Turned ${t.head_direction}`;

        const headTime = t.head_turn_time || 0;
        const headThresh = t.head_turn_threshold || 3.0;
        const headPct = Math.min(100, (headTime / headThresh) * 100);
        if (valHeadTimer) valHeadTimer.textContent = `${headTime.toFixed(2)} / ${headThresh.toFixed(2)} s`;
        if (barHeadTimer) {
            barHeadTimer.style.width = `${headPct}%`;
            barHeadTimer.className = `progress-bar ${headPct > 80 ? 'bg-danger' : headPct > 40 ? 'bg-warning' : 'bg-success'}`;
        }

        // 7. Hand Interaction Card
        const valHandsState = document.getElementById('val-hands-state');
        const valHandsCount = document.getElementById('val-hands-count');
        const valSteering = document.getElementById('val-steering-status');
        const valHandsTimer = document.getElementById('val-hands-timer');
        const barHandsTimer = document.getElementById('bar-hands-timer');

        if (valHandsState) valHandsState.textContent = t.hands || 'TWO HANDS';
        if (valHandsCount) valHandsCount.textContent = `${t.num_hands || 0} Visible`;
        if (valSteering) valSteering.textContent = t.hands === 'TWO HANDS' ? 'Engaged' : t.hands === 'ONE HAND' ? 'Single Hand' : 'Hands Off!';

        const handTime = t.hand_time || 0;
        const handThresh = t.hand_threshold || 2.0;
        const handPct = t.hands === 'TWO HANDS' ? 0 : Math.min(100, (handTime / handThresh) * 100);
        if (valHandsTimer) valHandsTimer.textContent = `${handTime.toFixed(2)} / ${handThresh.toFixed(2)} s`;
        if (barHandsTimer) {
            barHandsTimer.style.width = `${handPct}%`;
            barHandsTimer.className = `progress-bar ${handPct > 80 ? 'bg-danger' : handPct > 40 ? 'bg-warning' : 'bg-success'}`;
        }

        // 8. Yawn / Fatigue Card
        const valYawnState = document.getElementById('val-yawn-state');
        const valMar = document.getElementById('val-mar');
        const valYawns = document.getElementById('val-total-yawns');
        const valYawnTimer = document.getElementById('val-yawn-timer');
        const barYawnTimer = document.getElementById('bar-yawn-timer');

        if (valYawnState) valYawnState.textContent = t.yawning ? 'DETECTED' : 'NO';
        if (valMar) valMar.textContent = t.mar !== undefined ? Number(t.mar).toFixed(2) : '0.00';
        if (valYawns) valYawns.textContent = t.total_yawns || 0;

        const yawnTime = t.yawn_time || 0;
        const yawnThresh = t.yawn_threshold || 1.2;
        const yawnPct = Math.min(100, (yawnTime / yawnThresh) * 100);
        if (valYawnTimer) valYawnTimer.textContent = `${yawnTime.toFixed(2)} / ${yawnThresh.toFixed(2)} s`;
        if (barYawnTimer) {
            barYawnTimer.style.width = `${yawnPct}%`;
            barYawnTimer.className = `progress-bar ${yawnPct > 80 ? 'bg-danger' : yawnPct > 40 ? 'bg-warning' : 'bg-success'}`;
        }

        // 9. Active Alert Card & HUD text
        const activeMsg = document.getElementById('active-alert-msg');
        const hudAlertText = document.getElementById('hud-alert-text');
        const sevBadge = document.getElementById('alert-severity-badge');

        if (activeMsg) activeMsg.textContent = t.alert || 'Driver monitoring active';
        if (hudAlertText) hudAlertText.textContent = t.alert || 'Safe';

        if (sevBadge) {
            sevBadge.textContent = t.severity || 'INFO';
            sevBadge.className = `badge ${t.severity === 'CRITICAL' ? 'bg-danger' : t.severity === 'WARNING' ? 'bg-warning' : 'bg-success'}`;
        }

        // 10. Video HUD Timers Breakdown
        const hudTimers = document.getElementById('hud-timers-breakdown');
        if (hudTimers) {
            let timerParts = [];
            if (eyeTime > 0) timerParts.push(`Eyes: ${eyeTime.toFixed(1)}s`);
            if (headTime > 0) timerParts.push(`Head: ${headTime.toFixed(1)}s`);
            if (handTime > 0 && t.hands !== 'TWO HANDS') timerParts.push(`Hands: ${handTime.toFixed(1)}s`);
            hudTimers.textContent = timerParts.join(' | ');
        }

        // 11. Emergency Escalation Banner
        const emBanner = document.getElementById('emergency-banner');
        if (emBanner) {
            if (t.emergency_simulation) {
                emBanner.classList.remove('d-none');
                const emTime = document.getElementById('em-time');
                const emEvent = document.getElementById('em-event');
                if (emTime) emTime.textContent = new Date().toLocaleTimeString();
                if (emEvent) emEvent.textContent = t.alert;
            } else {
                emBanner.classList.add('d-none');
            }
        }

        // 12. State transition logging
        if (t.driver_status !== prevDriverStatus) {
            if (t.driver_status === 'SAFE') {
                logEvent('Driver returned to SAFE state.', 'success');
            } else if (t.driver_status === 'WARNING') {
                logEvent(`Driver entered WARNING state: ${t.alert}`, 'warning');
            } else if (t.driver_status === 'CRITICAL') {
                logEvent(`CRITICAL ALERT: ${t.alert}`, 'critical');
            }
        }

        prevDriverStatus = t.driver_status;
        prevAlertMsg = t.alert;
    }

    // -------------------------------------------------------------
    // 7. REAL-TIME EVENT CONSOLE
    // -------------------------------------------------------------
    window.logEvent = function(message, type = 'info') {
        const consoleBox = document.getElementById('event-log-console');
        if (!consoleBox) return;

        const now = new Date();
        const timeStr = now.toTimeString().split(' ')[0];

        const entry = document.createElement('div');
        entry.className = `log-entry log-entry-${type}`;
        entry.textContent = `[${timeStr}] ${message}`;

        consoleBox.insertBefore(entry, consoleBox.firstChild);

        while (consoleBox.children.length > 50) {
            consoleBox.removeChild(consoleBox.lastChild);
        }
    };

    window.clearEventConsole = function() {
        const consoleBox = document.getElementById('event-log-console');
        if (consoleBox) consoleBox.innerHTML = '';
        logEvent('Event console cleared.', 'info');
    };

    // -------------------------------------------------------------
    // 8. ALERT HISTORY TABLE (SQLite API)
    // -------------------------------------------------------------
    async function fetchAlertHistory() {
        try {
            const res = await fetch('/api/alerts?limit=25');
            if (!res.ok) return;
            const alerts = await res.json();
            renderAlertTable(alerts);
        } catch (e) {
            console.warn('Failed to fetch alert history:', e);
        }
    }

    function renderAlertTable(alerts) {
        const tbody = document.getElementById('alert-history-tbody');
        if (!tbody) return;

        if (!alerts || alerts.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted py-3">No alerts recorded yet.</td></tr>`;
            return;
        }

        tbody.innerHTML = alerts.map(a => {
            const sevBadge = a.severity === 'CRITICAL' 
                ? '<span class="badge bg-danger">CRITICAL</span>' 
                : a.severity === 'WARNING' 
                ? '<span class="badge bg-warning text-dark">WARNING</span>' 
                : '<span class="badge bg-info">INFO</span>';

            const timeFormatted = a.timestamp.includes(' ') ? a.timestamp.split(' ')[1] : a.timestamp;

            return `
                <tr>
                    <td class="font-monospace text-light">${timeFormatted}</td>
                    <td class="fw-semibold">${a.alert_type}</td>
                    <td>${sevBadge}</td>
                    <td class="font-monospace text-cyan">${a.duration}s</td>
                    <td><span class="text-success"><i class="fa-solid fa-check me-1"></i>${a.status}</span></td>
                </tr>
            `;
        }).join('');
    }

    window.clearAlertHistory = async function() {
        if (!confirm('Are you sure you want to clear all alert history?')) return;
        try {
            const res = await fetch('/api/alerts', { method: 'DELETE' });
            if (res.ok) {
                logEvent('Alert history cleared.', 'info');
                fetchAlertHistory();
                fetchStatistics();
            }
        } catch (e) {
            alert('Failed to clear alert history');
        }
    };

    // -------------------------------------------------------------
    // 9. STATISTICS & CHART.JS
    // -------------------------------------------------------------
    function initChart() {
        if (typeof Chart === 'undefined') {
            console.warn('[CHART] Chart.js library not loaded. Skipping chart initialization.');
            return;
        }
        try {
            const ctx = document.getElementById('alertDistributionChart');
            if (!ctx) return;

            alertChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Eye Closure', 'Looking Left/Right', 'Single Hand', 'Yawning', 'No Hands'],
                    datasets: [{
                        label: 'Alert Count',
                        data: [0, 0, 0, 0, 0],
                        backgroundColor: [
                            'rgba(239, 68, 68, 0.7)',
                            'rgba(245, 158, 11, 0.7)',
                            'rgba(6, 182, 212, 0.7)',
                            'rgba(59, 130, 246, 0.7)',
                            'rgba(168, 85, 247, 0.7)'
                        ],
                        borderColor: [
                            '#ef4444',
                            '#f59e0b',
                            '#06b6d4',
                            '#3b82f6',
                            '#a855f7'
                        ],
                        borderWidth: 1.5,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: '#94a3b8', stepSize: 1 },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' }
                        },
                        x: {
                            ticks: { color: '#94a3b8', font: { size: 10 } },
                            grid: { display: false }
                        }
                    }
                }
            });
        } catch (e) {
            console.warn('[CHART] Init exception:', e);
        }
    }

    async function fetchStatistics() {
        try {
            const res = await fetch('/api/statistics');
            if (!res.ok) return;
            const stats = await res.json();
            
            const totalEl = document.getElementById('total-alerts-count');
            if (totalEl) totalEl.textContent = `Total: ${stats.total_alerts || 0}`;

            if (alertChart && stats.type_distribution) {
                const dist = stats.type_distribution;
                alertChart.data.datasets[0].data = [
                    dist['Eye Closure Drowsiness'] || dist['Eye Closure'] || 0,
                    (dist['Looking Left'] || 0) + (dist['Looking Right'] || 0) + (dist['Looking Left/Right'] || 0),
                    dist['Single-Hand Driving'] || dist['Single Hand'] || 0,
                    dist['Yawning / Fatigue'] || dist['Yawning'] || 0,
                    dist['Hands Not Detected'] || 0
                ];
                alertChart.update();
            }
        } catch (e) {
            console.warn('Failed to fetch statistics:', e);
        }
    }

    // -------------------------------------------------------------
    // 10. SIMULATION & CAMERA CONTROLS
    // -------------------------------------------------------------
    window.triggerSimulation = async function(scenario) {
        try {
            const res = await fetch('/api/demo/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenario: scenario, duration: 8.0 })
            });
            const data = await res.json();
            logEvent(`Scenario Simulation Triggered: [${scenario.toUpperCase()}]`, 'warning');
        } catch (e) {
            console.error('Simulation trigger failed:', e);
        }
    };

    window.restartCamera = async function() {
        await fetch('/api/camera/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'start' })
        });
        const errOverlay = document.getElementById('camera-error-overlay');
        if (errOverlay) errOverlay.classList.add('d-none');
    };

    function bindUIEvents() {
        // Audio Toggle (FIXED: replaced getAudioContext with unlockAudioContext)
        const soundBtn = document.getElementById('sound-toggle-btn');
        if (soundBtn) {
            soundBtn.addEventListener('click', () => {
                audioEnabled = !audioEnabled;
                unlockAudioContext();
                if (audioEnabled) {
                    soundBtn.className = 'btn btn-sm btn-outline-info';
                    soundBtn.innerHTML = '<i class="fa-solid fa-volume-high" id="sound-icon"></i> Audio ON';
                    logEvent('Audio alerts enabled.', 'info');
                } else {
                    stopRepeatingAlarm();
                    soundBtn.className = 'btn btn-sm btn-outline-secondary';
                    soundBtn.innerHTML = '<i class="fa-solid fa-volume-xmark" id="sound-icon"></i> Audio OFF';
                    logEvent('Audio alerts muted.', 'info');
                }
            });
        }


        // Camera Control Buttons
        const btnStart = document.getElementById('btn-camera-start');
        if (btnStart) {
            btnStart.addEventListener('click', async () => {
                initBrowserWebcam();
                await fetch('/api/camera/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'start' }) });
                logEvent('Camera started.', 'info');
            });
        }

        const btnPause = document.getElementById('btn-camera-pause');
        if (btnPause) {
            btnPause.addEventListener('click', async () => {
                await fetch('/api/camera/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'pause' }) });
                logEvent('Detection paused.', 'warning');
            });
        }

        const btnResume = document.getElementById('btn-camera-resume');
        if (btnResume) {
            btnResume.addEventListener('click', async () => {
                await fetch('/api/camera/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'resume' }) });
                logEvent('Detection resumed.', 'info');
            });
        }

        const btnStop = document.getElementById('btn-camera-stop');
        if (btnStop) {
            btnStop.addEventListener('click', async () => {
                await fetch('/api/camera/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'stop' }) });
                logEvent('Camera stopped.', 'warning');
            });
        }

        // Video error & load listener
        const videoImg = document.getElementById('live-video-stream');
        if (videoImg) {
            videoImg.onerror = () => {
                const errOverlay = document.getElementById('camera-error-overlay');
                const camLabel = document.getElementById('cam-status-label');
                if (errOverlay) errOverlay.classList.remove('d-none');
                if (camLabel) camLabel.innerHTML = '<i class="fa-solid fa-triangle-exclamation text-danger me-1"></i>CAM ERR';
            };
            videoImg.onload = () => {
                const errOverlay = document.getElementById('camera-error-overlay');
                const camLabel = document.getElementById('cam-status-label');
                if (errOverlay) errOverlay.classList.add('d-none');
                if (camLabel) camLabel.innerHTML = '<i class="fa-solid fa-link text-success me-1"></i>CAM OK';
            };
        }
    }
});

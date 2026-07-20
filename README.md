# Proctoring App

A browser-based, webcam-driven exam proctoring application. It watches a candidate through their webcam during an exam session, automatically detects suspicious behavior in real time, records short video clips of each incident, and reports both events and clips to a backend API for later review.

## Features

- **Real-time webcam monitoring** — uses the candidate's webcam feed for continuous, on-device analysis for the duration of the exam session. No footage is analyzed server-side; detection runs entirely in the browser.
- **Automatic violation detection** — a detection loop continuously checks the live video for the violation types listed below and flags them the moment they occur.
- **Automatic incident recording** — the moment a violation is detected, a ~10 second video clip is recorded and attached to that incident. Recording only happens around violations; there is no continuous full-session recording.
- **Violation cooldown & episode handling** — a violation that persists (e.g. a phone left in frame) is treated as one continuous "episode" rather than re-triggering repeatedly; a new incident (new clip, new event) is only created once the same violation has cleared and re-occurs. This avoids duplicate clips/events for a single ongoing incident.
- **On-screen warning toasts** — a short-lived warning notification appears on screen each time a new violation episode starts, showing the violation type and how many times it has occurred so far in the session.
- **Live activity log** — a running, timestamped log of detected violations is kept and shown during the session.
- **Recorded clip list** — every violation clip is listed with its status (recording in progress / ready / failed) and can be downloaded individually. A live progress indicator shows while a clip is still being captured.
- **Violation event reporting** — each violation is reported to the backend as an event (type + timestamp), independent of the video clip upload.
- **Violation clip upload** — each recorded clip is uploaded to the backend, tagged with the violation type and the time window it covers.
- **Identity verification against KYC photo (wrong-face / face mismatch check)** — the live face in the webcam feed is continuously compared against a reference selfie photo retrieved from the candidate's KYC (Know Your Customer) verification record, to confirm the same person who was verified is the one taking the exam.
- **Detection overlay toggles** — optional on-screen overlays can be switched on to visualize what the detectors are seeing: person/face detection boxes, phone detection boxes, and eye/gaze tracking — useful for demonstration or troubleshooting.
- **Session identification** — the exam session is tied to a session ID (and user/ID number) passed in the URL, which is used to associate all events, clips, and identity checks with the correct candidate.

## Violations Detected

| Violation | Description |
|---|---|
| **Cell Phone / Possible Phone Usage** | A mobile phone is detected in view of the webcam, suggesting the candidate may be using it during the exam. |
| **Face Not Visible** | No face can be detected in the webcam feed (e.g. candidate has stepped away, is out of frame, or the camera is obstructed). |
| **Multiple Faces Detected** | More than one face is detected in the frame, suggesting another person may be present with the candidate. |
| **Tab Switch Detected** | The candidate has switched away from the exam browser tab/window. |
| **Face Mismatch (Wrong Face) Detected** | The face currently on webcam does not match the candidate's verified KYC selfie, suggesting the exam may be taken by someone else. |
| **Possible Looking Away Violation (Eyes Off Screen)** | The candidate's gaze/head pose indicates they are looking away from the screen for a sustained period, rather than briefly glancing away. |

Each violation type is tracked independently, with its own on-screen warning, log entry, recorded clip, and reported event.

## Detection Models

Detection runs entirely client-side in [src/App.tsx](src/App.tsx), using three independently loaded
MediaPipe Tasks Vision models plus two checks that don't require a model. Each violation type uses
its own model — swapping one does not affect the others.

| Violation | Detection method | Model |
|---|---|---|
| Cell Phone | MediaPipe `ObjectDetector` — object must be classified as a phone category (`cell phone` / `phone` / `mobile phone` / `smartphone`) with confidence `>= 0.42`, sustained for 2 consecutive frames, with an 800ms debounce between repeat triggers | **EfficientDet-Lite2** (float16, GPU delegate) |
| Multiple Faces / Face Not Visible | MediaPipe `FaceDetector`, minimum detection confidence 0.6 | BlazeFace short-range |
| Eyes Off Screen | MediaPipe `FaceLandmarker` — head pose (yaw/pitch from the facial transformation matrix) + gaze blendshapes | Face Landmarker (blendshapes + facial transformation matrix) |
| Tab Switch | Browser visibility/blur events | n/a — no ML model |
| Face Mismatch (Wrong Face) | face-api.js, in [src/proctoring/useFaceProctoring.ts](src/proctoring/useFaceProctoring.ts) | face-api.js models |

There is no rule engine layered on top of the object detector for phone detection — no hand-overlap,
gaze-toward-object, or geometry gates. A single object-detector confidence pass above threshold is
what confirms a Cell Phone violation.

**Cell-phone model choice:** the object detector uses EfficientDet-Lite2, the highest-accuracy
variant MediaPipe publishes for the Object Detector task (Lite0 is the only other option Google
hosts; Lite3+ is not published). Lite2 replaced the previously-used Lite0 after under-detection was
observed on lower-quality webcams/lighting, where Lite0's confidence scores fell under the 0.42
threshold for real phones.

## Event Reporting

Every violation is reported to two independent destinations, both driven from the same `eventType`
value (`getEventTypeFromDetectionType` in [src/App.tsx](src/App.tsx)) so they always agree:

- **Backend API** — `POST https://proctor-x-api.appricode.net/api/proctor/event` with
  `{ sessionId, timestamp, eventType }`.
- **Host application** — `window.parent.postMessage({ type: 'violation', eventType, message }, '*')`,
  sent from [src/lib/parentMessenger.ts](src/lib/parentMessenger.ts). The host listens with
  `window.addEventListener('message', ...)` and reads `event.data.eventType` /
  `event.data.message`.

| Violation (`DetectionType`) | `eventType` (sent to both) | `message` in postMessage |
|---|---|---|
| Cell Phone (`potential_prohibited_object`) | `potential-prohibited-object` | `Potential prohibited object detected` |
| Face Not Visible (`face_not_visible`) | `face-not-visible` | `Face Not Visible` |
| Multiple Faces Detected (`multiple_faces`) | `multiple-faces` | `Multiple Faces` |
| Tab Switch Detected (`tab_switch`) | `tab-switch` | `Tab Switch` |
| Face Mismatch (`wrong_face`) | `face-mismatch` | `Face Mismatch` |
| Eyes Off Screen (`eyes_off_screen`) | `eyes-off-screen` | `Looking Away` |

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

# Optional video review

The evidence auditor can ask Gemini 3.8 Flash questions about a local video:

```bash
python -m gym_anything_robotics.creation_audit.video_review \
  --video /absolute/path/evidence.mp4 \
  --question 'Describe what moves, with timestamps. Does the supposedly stationary object drift?' \
  --output /absolute/path/new_assessment.json
```

Use the Python runtime containing this package and its video-audit extra.
The helper takes GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment. Do not
put credentials in prompts or reports. It uploads this selected video, waits for
processing, saves the question and assessment, and deletes its remote upload.
It never deletes or edits the local video. Provider or processing failure is an
error, not an assessment. It does not replace the coding auditor's judgment.

Source: https://ai.google.dev/gemini-api/docs/video-understanding

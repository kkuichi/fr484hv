# XAI Moderation

XAI Moderation is a prototype web application for
explainable detection of toxic and antisocial content
in digital spaces.

The project was created as part of a bachelor thesis.

## What the project does

The application can analyze:

- text
- images
- audio files
- video files

The system returns:

- moderation result
- confidence score
- highlighted words that influenced toxic classification

For text analysis, the system uses Toxic-BERT and
Integrated Gradients.

## Project structure

Main parts of the project:

- `gateway` - web interface and API Gateway
- `text-service` - text toxicity detection and explanation
- `image-service` - OCR processing for images
- `audio-service` - speech-to-text transcription
- `video-service` - audio extraction and OCR from video frames
- `infra` - Docker Compose configuration
- `.github/workflows` - GitHub Actions workflow

## Running locally

Go to the infrastructure directory:

```bash
cd infra
```

Build and start all containers:

```bash
docker compose up -d --build
```

Check running containers:

```bash
docker compose ps
```

Open the application in browser:

```text
http://localhost:8080
```

## Environment variables

The project uses a `.env` file for configuration.

Important variables:

- `HF_TOKEN`
- `OCR_SPACE_API_KEY`
- `SIGHTENGINE_API_USER`
- `SIGHTENGINE_API_KEY`

Sensitive values should not be committed to Git.

## CI

GitHub Actions workflow is located in:

```text
.github/workflows/ci-test.yml
```

It builds and publishes Docker images for the main services
when changes are pushed to the `test` branch.

## Author

Fylyp Redkin © 2026
# Deploy AfriHate API to **Google Cloud Run** (project `zero-shot-494819`)

This guide deploys **three** services so your UI can compare:

| Cloud Run service (example name) | Weights in GCS (you choose bucket + prefix) | Env `NLP_SERVE_VARIANT` |
|----------------------------------|-----------------------------------------------|-------------------------|
| `afrihate-e4-zero` | `…/models/e4/` (contents of `Final_Source_Model/`) | `e4_zero` |
| `afrihate-fewshot-twi-5` | `…/models/fewshot_twi_5/` | `fewshot_twi_5` |
| `afrihate-fewshot-pcm-5` | `…/models/fewshot_pcm_5/` | `fewshot_pcm_5` |

Use **k=5** few-shot adapters because your results showed the clearest Pidgin gain at k=5; Twi adapters are still useful for UI contrast even if macro-F1 is similar to zero-shot.

> **Account:** use the Google identity that has **Owner** or **Editor** on project `zero-shot-494819` (e.g. your team Gmail). **Do not** commit passwords or JSON keys to Git.

> **Limitation:** this assistant **cannot** run `gcloud` against your account. Run every command below on your machine after `gcloud auth login` and `gcloud config set project zero-shot-494819`.

---

## Automated one-shot deploy (recommended)

From **repository root** (`FinalProject/`), after weights and few-shot folders exist:

```bash
gcloud auth login
gcloud auth application-default login
./deploy/gcp/deploy_all.sh
```

This script: enables APIs, creates Artifact Registry + bucket (if missing), **rsync**s `Final_Source_Model/` and `Phase2_Outputs/fewshot_{twi,pcm}_5/` to GCS, runs **Cloud Build** from `deploy/gcp/cloudbuild.yaml`, grants IAM, and deploys **three** Cloud Run services.

Options:

```bash
./deploy/gcp/deploy_all.sh --skip-upload   # weights already in GCS
./deploy/gcp/deploy_all.sh --skip-build    # image already pushed; only (re)deploy Run
./deploy/gcp/deploy_all.sh --help
```

Override defaults with env vars, e.g. `GCS_MODEL_BUCKET=my-unique-bucket-name`.

### If deploy fails: billing (`UREQ_PROJECT_BILLING_NOT_FOUND`)

Cloud Run, Cloud Build, and Artifact Registry **require a billing account** on the target project.

1. Open **[Billing → link a billing account](https://console.cloud.google.com/billing/linkedaccount?project=zero-shot-494819)** for project `zero-shot-494819` (use a project Owner / Billing Admin account).
2. Wait ~1 minute, then run `./deploy/gcp/deploy_all.sh` again.

Without billing, `gcloud services enable` fails with *Billing account for project … is not found*.

### If you see a quota-project warning (`zephyrmobile-492023` vs `zero-shot-494819`)

After `gcloud auth application-default login`, your ADC file may still point quota at another project. The deploy script runs:

`gcloud auth application-default set-quota-project zero-shot-494819`

You can run that manually anytime so client libraries and warnings match the deploy project.

---

## 0. One-time: APIs, Artifact Registry, bucket

```bash
export PROJECT_ID=zero-shot-494819
export REGION=us-central1
export AR_REPO=afrihate
export BUCKET=${PROJECT_ID}-model-weights   # must be globally unique — pick another name if taken

gcloud config set project "${PROJECT_ID}"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com

gcloud artifacts repositories create "${AR_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="AfriHate inference images" \
  2>/dev/null || true

gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" 2>/dev/null || true
```

---

## 1. Upload model folders to Cloud Storage

From your laptop (paths relative to repo root `FinalProject/`):

```bash
export BUCKET=YOUR_BUCKET_NAME_HERE

# E4 zero-shot (must contain config.json, tokenizer*, model.safetensors)
gcloud storage rsync --recursive Final_Source_Model/ "gs://${BUCKET}/models/e4/"

# Few-shot adapters (after Experiment 2 has created these folders)
gcloud storage rsync --recursive Phase2_Outputs/fewshot_twi_5/ "gs://${BUCKET}/models/fewshot_twi_5/"
gcloud storage rsync --recursive Phase2_Outputs/fewshot_pcm_5/ "gs://${BUCKET}/models/fewshot_pcm_5/"
```

---

## 2. Build and push the Docker image

From **repository root**:

```bash
export PROJECT_ID=zero-shot-494819
export REGION=us-central1
export AR_REPO=afrihate
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/afrihate-api:latest"

gcloud auth configure-docker "${REGION}-docker.pkg.dev"

docker build -f deploy/gcp/Dockerfile -t "${IMAGE}" .
docker push "${IMAGE}"
```

(Alternatively: `gcloud builds submit --tag "${IMAGE}"` from the same directory; requires Cloud Build API and Dockerfile path support — use `cloudbuild.yaml` if you prefer remote builds.)

---

## 3. Grant Cloud Run access to read the bucket

Cloud Run’s default compute service account needs **object read** on your bucket:

```bash
export PROJECT_ID=zero-shot-494819
export BUCKET=YOUR_BUCKET_NAME_HERE
export PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

---

## 4. Deploy three Cloud Run services

Replace `YOUR_BUCKET` and adjust memory if deploy fails (2 GB weights + PyTorch needs headroom).

```bash
export PROJECT_ID=zero-shot-494819
export REGION=us-central1
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/afrihate/afrihate-api:latest"
export BUCKET=YOUR_BUCKET_NAME_HERE

# --- E4 zero-shot ---
gcloud run deploy afrihate-e4-zero \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --allow-unauthenticated \
  --memory=8Gi \
  --cpu=2 \
  --timeout=900 \
  --set-env-vars="GCS_MODEL_URI=gs://${BUCKET}/models/e4,NLP_SERVE_VARIANT=e4_zero"

# --- Few-shot Twi k=5 ---
gcloud run deploy afrihate-fewshot-twi-5 \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --allow-unauthenticated \
  --memory=8Gi \
  --cpu=2 \
  --timeout=900 \
  --set-env-vars="GCS_MODEL_URI=gs://${BUCKET}/models/fewshot_twi_5,NLP_SERVE_VARIANT=fewshot_twi_5"

# --- Few-shot Pidgin k=5 ---
gcloud run deploy afrihate-fewshot-pcm-5 \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --allow-unauthenticated \
  --memory=8Gi \
  --cpu=2 \
  --timeout=900 \
  --set-env-vars="GCS_MODEL_URI=gs://${BUCKET}/models/fewshot_pcm_5,NLP_SERVE_VARIANT=fewshot_pcm_5"
```

Each command prints a **HTTPS URL**. Point your web app at:

- `https://…/predict` on each host (three different bases).

**Cold starts:** the first request after idle may download ~2 GB from GCS (can take **several minutes**). For a demo, set **`--min-instances=1`** on each service (adds cost) or warm up once before presenting.

**Security:** `--allow-unauthenticated` is for class demos only. For production, remove it and use **IAM / Identity-Aware Proxy / your own API gateway**.

---

## 5. UI wiring

Example body for all three services (same JSON):

```json
{"text":"…"}
```

Read `model_scope.variant` / `model_scope.variant_name` in the response to label columns in your UI (“E4 zero-shot” vs “Few-shot Twi k=5” vs “Few-shot Pidgin k=5”).

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `403` / download fails | Re-check bucket IAM for the **compute** service account (§3). |
| Container OOM | Increase `--memory` (try `16Gi`). |
| Slow first request | Expected: model download; use `--min-instances=1` or bake weights into the image in a private build. |
| `fewshot_*` folder missing | Run **Experiment 2** locally first, then re-upload §1. |

---

## Optional: same layout locally (Docker)

```bash
docker build -f deploy/gcp/Dockerfile -t afrihate-api:local .
docker run --rm -p 8080:8080 \
  -e GCS_MODEL_URI="gs://YOUR_BUCKET/models/e4" \
  -e NLP_SERVE_VARIANT=e4_zero \
  -v "${HOME}/.config/gcloud:/root/.config/gcloud:ro" \
  afrihate-api:local
```

(Uses your gcloud ADC to read the bucket; simpler for debugging than Cloud Run.)

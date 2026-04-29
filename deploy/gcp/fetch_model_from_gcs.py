#!/usr/bin/env python3
"""Download a Hugging Face model folder from GCS into a local directory (for Cloud Run cold start).

Usage:
  python fetch_model_from_gcs.py gs://BUCKET/PREFIX/path/ /dest/dir

All objects under PREFIX are written preserving relative paths. PREFIX may omit trailing slash.
Requires Application Default Credentials (Cloud Run service account) or local gcloud auth.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_GS = re.compile(r"^gs://([^/]+)/?(.*)$")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: fetch_model_from_gcs.py gs://bucket/prefix/ /local/dest", file=sys.stderr)
        sys.exit(2)
    uri, dest_s = sys.argv[1], sys.argv[2]
    m = _GS.match(uri.strip())
    if not m:
        print("URI must start with gs://", file=sys.stderr)
        sys.exit(2)
    bucket_name, prefix = m.group(1), m.group(2).strip("/")
    if not prefix:
        print("URI must include a path prefix, e.g. gs://my-bucket/models/e4/", file=sys.stderr)
        sys.exit(2)
    prefix = prefix + "/"
    dest = Path(dest_s)
    dest.mkdir(parents=True, exist_ok=True)

    from google.cloud import storage

    client = storage.Client()
    bkt = client.bucket(bucket_name)
    found = 0
    for blob in client.list_blobs(bucket_name, prefix=prefix or None):
        name = blob.name
        if name.endswith("/"):
            continue
        rel = name[len(prefix) :] if prefix else name
        rel = rel.lstrip("/")
        if not rel:
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(out))
        found += 1
    if found == 0:
        print(f"No objects under gs://{bucket_name}/{prefix}", file=sys.stderr)
        sys.exit(1)
    print(f"Downloaded {found} files to {dest}")


if __name__ == "__main__":
    main()

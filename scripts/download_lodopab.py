"""Resume downloads of the files needed by the LoDoPaB staging command."""
import argparse
import http.client
import time
import urllib.error
import urllib.request
from pathlib import Path

RELEASE = "https://zenodo.org/records/3384092/files"
CHUNK_BYTES = 64 * 1024 * 1024
RETRY_LIMIT = 4
SIZES = {
    "ground_truth_test.zip": 1582139537,
    "observation_test.zip": 2996574366,
    "ground_truth_validation.zip": 1567086812,
    "observation_validation.zip": 2944573582,
    "patient_ids_rand_test.csv": 14212,
    "patient_ids_rand_train.csv": 137022,
    "patient_ids_rand_validation.csv": 14088,
}


def _download_range(path: Path, size: int, offset: int, stop: int, release: str) -> None:
    request = urllib.request.Request(
        f"{release}/{path.name}?download=1&offset={offset}&end={stop - 1}",
        headers={"Range": f"bytes={offset}-{stop - 1}", "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        expected_range = f"bytes {offset}-{stop - 1}/{size}"
        if response.status == 206:
            if response.headers.get("Content-Range") != expected_range:
                raise ValueError(f"Unexpected Content-Range for {path.name}")
        elif offset != 0 or stop != size or response.status != 200:
            raise ValueError(f"Server did not honor the resume offset for {path.name}")
        length = response.headers.get("Content-Length")
        if length is not None and length != str(stop - offset):
            raise ValueError(f"Unexpected Content-Length for {path.name}")

        remaining = stop - offset
        with path.open("ab") as target:
            while remaining:
                chunk = response.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError(f"Incomplete download: {path.name}")
                target.write(chunk)
                remaining -= len(chunk)


def download_file(path: Path, size: int, release: str = RELEASE) -> None:
    """Download bounded ranges and resume from bytes saved before a transient failure."""
    failures = 0
    offset = path.stat().st_size if path.exists() else 0
    if offset > size:
        raise ValueError(f"{path.name} is larger than the official file")
    while offset < size:
        stop = min(size, offset + CHUNK_BYTES)
        try:
            _download_range(path, size, offset, stop, release)
            failures = 0
        except (OSError, http.client.HTTPException) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (
                408, 500, 502, 503, 504
            ):
                raise
            failures += 1
            if failures == RETRY_LIMIT:
                raise
            time.sleep(2 ** (failures - 1))
        offset = path.stat().st_size if path.exists() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--split", choices=("validation", "test"),
                           help="Download one split (default: validation)")
    selection.add_argument("--file", choices=SIZES, help="Download only this file")
    args = parser.parse_args()
    args.downloads.mkdir(parents=True, exist_ok=True)
    split = args.split or "validation"
    files = [args.file] if args.file else [
        name for name in SIZES if name.endswith(".csv") or name.endswith(f"_{split}.zip")
    ]
    try:
        for name in files:
            print(f"Downloading or reusing {name}", flush=True)
            download_file(args.downloads / name, SIZES[name])
    except (OSError, ValueError, http.client.HTTPException) as error:
        parser.exit(1, f"{error}\nRepeat the same command to resume a partial download.\n")
    print("Transfer sizes complete. Run stage_lodopab.py to verify release checksums.")


if __name__ == "__main__":
    main()

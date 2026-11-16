"""Resume downloads of the files needed by the LoDoPaB staging command."""
import argparse
import http.client
import urllib.request
from pathlib import Path

RELEASE = "https://zenodo.org/records/3384092/files"
SIZES = {
    "ground_truth_validation.zip": 1567086812,
    "observation_validation.zip": 2944573582,
    "patient_ids_rand_test.csv": 14212,
    "patient_ids_rand_train.csv": 137022,
    "patient_ids_rand_validation.csv": 14088,
}


def download_file(path: Path, size: int, release: str = RELEASE) -> None:
    """Keep partial bytes and append only a response matching the requested range."""
    offset = path.stat().st_size if path.exists() else 0
    if offset == size:
        return
    if offset > size:
        raise ValueError(f"{path.name} is larger than the official file")
    request = urllib.request.Request(
        f"{release}/{path.name}?download=1&offset={offset}",
        headers={"Range": f"bytes={offset}-{size - 1}", "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        expected_range = f"bytes {offset}-{size - 1}/{size}"
        if response.status == 206:
            if response.headers.get("Content-Range") != expected_range:
                raise ValueError(f"Unexpected Content-Range for {path.name}")
        elif offset != 0 or response.status != 200:
            raise ValueError(f"Server did not honor the resume offset for {path.name}")
        length = response.headers.get("Content-Length")
        if length is not None and length != str(size - offset):
            raise ValueError(f"Unexpected Content-Length for {path.name}")

        remaining = size - offset
        with path.open("ab") as target:
            while remaining:
                chunk = response.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError(f"Incomplete download: {path.name}")
                target.write(chunk)
                remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument(
        "--file", choices=SIZES,
        help="Download only this file; default downloads all validation inputs",
    )
    args = parser.parse_args()
    args.downloads.mkdir(parents=True, exist_ok=True)
    try:
        for name in [args.file] if args.file else SIZES:
            print(f"Downloading or reusing {name}", flush=True)
            download_file(args.downloads / name, SIZES[name])
    except (OSError, ValueError, http.client.HTTPException) as error:
        parser.exit(1, f"{error}\nRepeat the same command to resume a partial download.\n")
    print("Transfer sizes complete. Run stage_lodopab.py to verify release checksums.")


if __name__ == "__main__":
    main()

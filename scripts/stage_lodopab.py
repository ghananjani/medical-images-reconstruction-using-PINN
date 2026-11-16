"""Verify and unpack official LoDoPaB validation or test archives."""
import argparse
import filecmp
import hashlib
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

# Release 1.0.0 checksums from https://zenodo.org/api/records/3384092
CHECKSUMS = {
    "ground_truth_test.zip": "ecc655767fbe3d40908ca823921f4c7f",
    "ground_truth_validation.zip": "666c36f403734842f14ca8811a63b8f7",
    "observation_test.zip": "9ae6b053bb1faa94d573311af8ec67b2",
    "observation_validation.zip": "3cff406e09c59774912655eb7a72cfcf",
    "patient_ids_rand_test.csv": "e86068312ad8a039e03f7f929352f7fd",
    "patient_ids_rand_train.csv": "4fee01b076920f85fc92e0de774dc277",
    "patient_ids_rand_validation.csv": "a387e619074f49573ae376c60a948db4",
}


def archive_members(bundle, stem):
    members = {}
    for entry in bundle.infolist():
        if entry.is_dir():
            continue
        name = Path(entry.filename).name
        if not re.fullmatch(rf"{re.escape(stem)}_\d{{3}}\.hdf5", name) or name in members:
            raise ValueError(f"Unexpected or duplicate archive member: {entry.filename}")
        members[name] = entry
    expected = {f"{stem}_{i:03d}.hdf5" for i in range(28)}
    if members.keys() != expected:
        raise ValueError(f"Unexpected shard listing for {stem}")
    return members


def stage(downloads, output, split):
    mappings = [f"patient_ids_rand_{part}.csv" for part in ("train", "validation", "test")]
    archives = [f"{kind}_{split}.zip" for kind in ("observation", "ground_truth")]
    errors = []
    for name in [*mappings, *archives]:
        path = downloads / name
        try:
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, lambda: hashlib.md5(usedforsecurity=False))
            if digest.hexdigest() != CHECKSUMS[name]:
                errors.append(f"Checksum mismatch: {path}")
        except OSError as error:
            errors.append(str(error))
    if errors:
        raise ValueError("\n".join(errors))

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temp:
        staging = Path(temp)
        for name in archives:
            with zipfile.ZipFile(downloads / name) as bundle:
                members = archive_members(bundle, Path(name).stem)
                for filename, member in members.items():
                    with bundle.open(member) as source, (staging / filename).open("wb") as target:
                        shutil.copyfileobj(source, target)
        for name in mappings:
            shutil.copyfile(downloads / name, staging / name)
        for path in staging.iterdir():
            target = output / path.name
            if target.exists() and not filecmp.cmp(path, target, shallow=False):
                raise ValueError(f"Existing file differs from the official release: {target}")
        for path in staging.iterdir():
            target = output / path.name
            if not target.exists():
                path.replace(target)
    print(f"Verified and staged the {split} split in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    try:
        stage(args.downloads, args.output, args.split)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

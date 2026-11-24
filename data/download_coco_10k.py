#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Download COCO images referenced by data/4_coco_30k_10k.csv.')
    parser.add_argument('--csv', default='data/4_coco_30k_10k.csv', help='CSV containing image_id and coco_url.')
    parser.add_argument('--output-dir', default='data/coco_10k', help='Directory to store downloaded images.')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel download workers.')
    parser.add_argument('--timeout', type=float, default=60.0, help='Per-request timeout in seconds.')
    parser.add_argument('--retries', type=int, default=3, help='Retries per image before marking it failed.')
    return parser


def extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix or '.jpg'


def normalize_download_url(url: str) -> str:
    if url.startswith('http://images.cocodataset.org/'):
        return 'https://' + url[len('http://'):]
    return url


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def load_rows(csv_path: Path) -> list[tuple[int, str, Path]]:
    rows: list[tuple[int, str, Path]] = []
    with csv_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(itertools.dropwhile(lambda line: line.startswith('#'), handle))
        for row in reader:
            image_id = int(row['image_id'])
            url = row['coco_url'].strip()
            ext = extension_from_url(url)
            rows.append((image_id, url, Path(f'{image_id}{ext}')))
    return rows


def download_one(image_id: int, url: str, dest_path: Path, timeout: float, retries: int) -> tuple[str, int, str]:
    if dest_path.is_file() and dest_path.stat().st_size > 0 and is_valid_image(dest_path):
        return ('skipped', image_id, dest_path.name)
    if dest_path.exists():
        dest_path.unlink()

    tmp_path = dest_path.with_suffix(dest_path.suffix + '.part')
    headers = {'User-Agent': 'Mozilla/5.0'}
    normalized_url = normalize_download_url(url)
    parsed = urlparse(normalized_url)
    context = None
    if parsed.scheme == 'https' and parsed.hostname == 'images.cocodataset.org':
        context = ssl._create_unverified_context()

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(normalized_url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response, tmp_path.open('wb') as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if not is_valid_image(tmp_path):
                raise ValueError(f'not a valid image: {normalized_url}')
            os.replace(tmp_path, dest_path)
            return ('downloaded', image_id, dest_path.name)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                return ('failed', image_id, f'{normalized_url} :: {exc}')
            time.sleep(min(2 ** (attempt - 1), 5))
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            return ('failed', image_id, f'{normalized_url} :: {exc}')
    return ('failed', image_id, normalized_url)


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.workers <= 0:
        raise SystemExit('--workers must be positive')
    if not csv_path.is_file():
        raise SystemExit(f'CSV not found: {csv_path}')

    rows = load_rows(csv_path)
    total = len(rows)
    print(f'CSV: {csv_path}')
    print(f'Output: {output_dir}')
    print(f'Images: {total}')
    print(f'Workers: {args.workers}')

    downloaded = 0
    skipped = 0
    failed: list[tuple[int, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, image_id, url, output_dir / rel_path, args.timeout, args.retries)
            for image_id, url, rel_path in rows
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            status, image_id, info = future.result()
            if status == 'downloaded':
                downloaded += 1
            elif status == 'skipped':
                skipped += 1
            else:
                failed.append((image_id, info))

            if index == 1 or index == total or index % 100 == 0:
                print(
                    f'Progress: {index}/{total} '
                    f'(downloaded={downloaded}, skipped={skipped}, failed={len(failed)})',
                    flush=True,
                )

    if failed:
        failed_path = output_dir / 'failed_downloads.csv'
        with failed_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['image_id', 'error'])
            writer.writerows(failed)
        print(f'Wrote failures: {failed_path}')

    print(f'Finished: downloaded={downloaded}, skipped={skipped}, failed={len(failed)}')
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Offline OCR for image-only PDFs; retain originals and rasterize one page at a time."""

import os
import re
import subprocess
import tempfile
from pathlib import Path


def ocr_pdf(path: str) -> str:
    source = Path(path)
    if source.suffix.lower() != ".pdf":
        raise ValueError("Local OCR accepts PDF documents only")
    env = {**os.environ, "OMP_THREAD_LIMIT": "1", "LC_ALL": "C"}

    def run(args):
        try:
            return subprocess.run(
                args, check=True, capture_output=True, timeout=120, env=env
            ).stdout.decode("utf-8", errors="replace")
        except (OSError, subprocess.SubprocessError):
            raise ValueError(
                "Local PDF OCR failed; verify the PDF and OCR runtime"
            ) from None

    info = run(["pdfinfo", str(source)])
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
    pages = int(match.group(1)) if match else 0
    if not 1 <= pages <= 200:
        raise ValueError("Local PDF OCR requires between 1 and 200 pages")
    result = []
    with tempfile.TemporaryDirectory(prefix="open-notebook-ocr-") as directory:
        prefix = Path(directory) / "page"
        for page in range(1, pages + 1):
            run(
                [
                    "pdftoppm",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-r",
                    "200",
                    "-singlefile",
                    "-png",
                    str(source),
                    str(prefix),
                ]
            )
            text = run(
                [
                    "tesseract",
                    str(prefix) + ".png",
                    "stdout",
                    "-l",
                    "rus+eng",
                    "--psm",
                    "3",
                ]
            )
            result.append(f"[OCR page {page}]\n{text.strip()}")
            Path(str(prefix) + ".png").unlink(missing_ok=True)
    text = "\n\n".join(result)
    if not any(
        re.search(r"[A-Za-zА-Яа-я]{3}", page.split("\n", 1)[1]) for page in result
    ):
        raise ValueError("Local PDF OCR returned no usable text")
    return text

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from open_notebook.utils.local_pdf_ocr import ocr_pdf


class LocalOCRTest(unittest.TestCase):
    def test_pages_languages_and_no_original_write(self):
        calls = []

        def run(args, **kwargs):
            calls.append(args)
            self.assertEqual(kwargs["env"]["OMP_THREAD_LIMIT"], "1")
            if args[0] == "pdfinfo":
                return subprocess.CompletedProcess(args, 0, b"Pages: 2\n")
            if args[0] == "tesseract":
                return subprocess.CompletedProcess(args, 0, "Текст Jira".encode())
            Path(args[-1] + ".png").write_bytes(b"synthetic")
            return subprocess.CompletedProcess(args, 0, b"")

        with patch("subprocess.run", side_effect=run):
            text = ocr_pdf("/uploads/original.pdf")
        self.assertIn("[OCR page 2]", text)
        self.assertEqual(len([c for c in calls if c[0] == "tesseract"]), 2)
        self.assertTrue(all("rus+eng" in c for c in calls if c[0] == "tesseract"))
        self.assertFalse(Path("/uploads/original.pdf").exists())

    def test_invalid_count_and_subprocess_failure_are_not_partial_success(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, b"Pages: 201\n"),
        ):
            with self.assertRaises(ValueError):
                ocr_pdf("/uploads/original.pdf")
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("tesseract", 120)
        ):
            with self.assertRaisesRegex(ValueError, "Local PDF OCR failed"):
                ocr_pdf("/uploads/original.pdf")

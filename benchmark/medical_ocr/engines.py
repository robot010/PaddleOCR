"""OCR engine wrappers providing a unified interface for benchmarking.

Each engine implements:
    - name: str
    - recognize(image_path: str) -> str
    - recognize_batch(image_paths: list[str]) -> list[str]
"""

from __future__ import annotations

import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image


def _ensure_rgb_jpeg(image_path: str | Path) -> str:
    """Ensure image is a valid RGB JPEG. Returns path (original or converted temp file).

    Some datasets contain misnamed files (e.g., GIF saved as .jpg) or palette-mode
    images that OCR engines can't handle. This converts them to RGB JPEG.
    """
    img = Image.open(image_path)
    if img.format == "JPEG" and img.mode == "RGB":
        return str(Path(image_path).resolve())
    # Convert to RGB JPEG in a temp file
    img = img.convert("RGB")
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG")
    return tmp.name


class OCREngine(ABC):
    """Base class for OCR engine wrappers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""

    @abstractmethod
    def recognize(self, image_path: str | Path) -> str:
        """Recognize text from a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            Recognized text string.
        """

    def recognize_batch(self, image_paths: list[str | Path]) -> list[str]:
        """Recognize text from a batch of images. Default: sequential."""
        return [self.recognize(p) for p in image_paths]

    def recognize_timed(self, image_path: str | Path) -> tuple[str, float]:
        """Recognize text and return (text, elapsed_ms)."""
        start = time.perf_counter()
        text = self.recognize(image_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return text, elapsed_ms


class TesseractEngine(OCREngine):
    """Wrapper for Tesseract OCR via pytesseract."""

    def __init__(self, lang: str = "eng", config: str = "") -> None:
        import pytesseract  # noqa: F401 — verify import

        self._lang = lang
        self._config = config

    @property
    def name(self) -> str:
        return f"Tesseract(lang={self._lang})"

    def recognize(self, image_path: str | Path) -> str:
        import pytesseract

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=self._lang, config=self._config)
        return text.strip()


class EasyOCREngine(OCREngine):
    """Wrapper for EasyOCR."""

    def __init__(self, lang_list: list[str] | None = None, gpu: bool = True) -> None:
        import easyocr

        self._lang_list = lang_list or ["en"]
        self._reader = easyocr.Reader(self._lang_list, gpu=gpu)

    @property
    def name(self) -> str:
        langs = "+".join(self._lang_list)
        return f"EasyOCR(lang={langs})"

    def recognize(self, image_path: str | Path) -> str:
        results = self._reader.readtext(str(image_path))
        # EasyOCR returns list of (bbox, text, confidence)
        texts = [r[1] for r in results]
        return " ".join(texts).strip()


class PaddleOCRv5Engine(OCREngine):
    """Wrapper for PaddleOCR PP-OCRv5 pipeline (detection + recognition)."""

    def __init__(self, lang: str = "en", use_gpu: bool = True) -> None:
        from paddleocr import PaddleOCR

        self._lang = lang
        self._ocr = PaddleOCR(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False)

    @property
    def name(self) -> str:
        return f"PP-OCRv5(lang={self._lang})"

    def recognize(self, image_path: str | Path) -> str:
        safe_path = _ensure_rgb_jpeg(image_path)
        result = self._ocr.predict(safe_path)
        # result is a list of prediction results
        texts = []
        for res in result:
            if hasattr(res, "rec_texts"):
                texts.extend(res.rec_texts)
            elif isinstance(res, dict) and "rec_texts" in res:
                texts.extend(res["rec_texts"])
        return " ".join(texts).strip()


class PaddleOCRVLEngine(OCREngine):
    """Wrapper for PaddleOCR-VL (Vision-Language model)."""

    def __init__(self) -> None:
        from paddleocr import PaddleOCRVL

        self._ocr = PaddleOCRVL()

    @property
    def name(self) -> str:
        return "PaddleOCR-VL-1.5"

    def recognize(self, image_path: str | Path) -> str:
        safe_path = _ensure_rgb_jpeg(image_path)
        result = self._ocr.predict(safe_path)
        texts = []
        for res in result:
            if hasattr(res, "text"):
                texts.append(res.text)
            elif isinstance(res, dict) and "text" in res:
                texts.append(res["text"])
        # PaddleOCR-VL may return markdown; extract plain text
        full_text = "\n".join(texts).strip()
        return full_text


class DocTREngine(OCREngine):
    """Wrapper for docTR (Mindee Document Text Recognition)."""

    def __init__(self) -> None:
        from doctr.models import ocr_predictor

        self._predictor = ocr_predictor(
            det_arch="db_mobilenet_v3_large", reco_arch="crnn_mobilenet_v3_large", pretrained=True
        )

    @property
    def name(self) -> str:
        return "docTR"

    def recognize(self, image_path: str | Path) -> str:
        from doctr.io import DocumentFile

        doc = DocumentFile.from_images(str(Path(image_path).resolve()))
        result = self._predictor(doc)

        texts = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join(w.value for w in line.words)
                    texts.append(line_text)
        return " ".join(texts).strip()


class GLMOCREngine(OCREngine):
    """Wrapper for GLM-OCR (0.9B VLM from Zhipu AI)."""

    def __init__(self) -> None:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_path = "zai-org/GLM-OCR"
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto",
        )

    @property
    def name(self) -> str:
        return "GLM-OCR(0.9B)"

    def recognize(self, image_path: str | Path) -> str:
        safe_path = _ensure_rgb_jpeg(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": safe_path},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        inputs.pop("token_type_ids", None)
        generated_ids = self._model.generate(**inputs, max_new_tokens=256)
        output = self._processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return output.strip()


class MinerUEngine(OCREngine):
    """Wrapper for MinerU 2.5 (1.2B VLM from OpenDataLab)."""

    def __init__(self) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from mineru_vl_utils import MinerUClient

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "opendatalab/MinerU2.5-2509-1.2B",
            dtype="auto",
            device_map="auto",
        )
        # Patch missing attribute that mineru-vl-utils expects
        if not hasattr(model.config, "max_position_embeddings"):
            model.config.max_position_embeddings = 32768
        processor = AutoProcessor.from_pretrained(
            "opendatalab/MinerU2.5-2509-1.2B",
        )
        self._client = MinerUClient(
            backend="transformers",
            model=model,
            processor=processor,
        )

    @property
    def name(self) -> str:
        return "MinerU2.5(1.2B)"

    def recognize(self, image_path: str | Path) -> str:
        img = Image.open(image_path).convert("RGB")
        blocks = self._client.two_step_extract(img)
        texts = []
        for block in blocks:
            if hasattr(block, "content") and block.content:
                texts.append(block.content)
            elif isinstance(block, dict) and block.get("content"):
                texts.append(block["content"])
        return " ".join(texts).strip()


def get_engine(engine_name: str, **kwargs: Any) -> OCREngine:
    """Factory function to create an OCR engine by name.

    Args:
        engine_name: One of 'tesseract', 'easyocr', 'ppocr_v5', 'paddleocr_vl',
                     'doctr', 'glm_ocr', 'mineru'.
        **kwargs: Engine-specific keyword arguments.

    Returns:
        An OCREngine instance.
    """
    engines = {
        "tesseract": TesseractEngine,
        "easyocr": EasyOCREngine,
        "ppocr_v5": PaddleOCRv5Engine,
        "paddleocr_vl": PaddleOCRVLEngine,
        "doctr": DocTREngine,
        "glm_ocr": GLMOCREngine,
        "mineru": MinerUEngine,
    }
    if engine_name not in engines:
        raise ValueError(
            f"Unknown engine '{engine_name}'. Available: {list(engines.keys())}"
        )
    return engines[engine_name](**kwargs)

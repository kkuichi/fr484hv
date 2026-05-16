from fastapi import FastAPI, UploadFile, File, Request
import os
import requests
from io import BytesIO
from PIL import Image, ImageOps, UnidentifiedImageError

app = FastAPI()

OCR_KEY = os.getenv("OCR_SPACE_API_KEY")

OCR_MAX_BYTES = 1_400_000


def compress_image_for_ocr(
    image_bytes: bytes,
    filename: str
):
    original_size = len(image_bytes)

    if original_size <= OCR_MAX_BYTES:
        return {
            "bytes": image_bytes,
            "filename": filename,
            "content_type": "image/png",
            "compressed": False,
            "original_size": original_size,
            "final_size": original_size
        }

    try:
        img = Image.open(BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)

        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new(
                "RGB",
                img.size,
                (255, 255, 255)
            )

            if img.mode == "P":
                img = img.convert("RGBA")

            background.paste(
                img,
                mask=img.split()[-1]
                if img.mode in ("RGBA", "LA")
                else None
            )

            img = background
        else:
            img = img.convert("RGB")

    except UnidentifiedImageError:
        return {
            "bytes": image_bytes,
            "filename": filename,
            "content_type": "application/octet-stream",
            "compressed": False,
            "original_size": original_size,
            "final_size": original_size,
            "warning": "File is not a valid image"
        }

    quality_values = [85, 75, 65, 55, 45, 35]

    for quality in quality_values:
        output = BytesIO()

        img.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=True
        )

        compressed = output.getvalue()

        if len(compressed) <= OCR_MAX_BYTES:
            return {
                "bytes": compressed,
                "filename": "compressed_ocr.jpg",
                "content_type": "image/jpeg",
                "compressed": True,
                "original_size": original_size,
                "final_size": len(compressed),
                "quality": quality
            }

    while True:
        width, height = img.size

        new_width = int(width * 0.85)
        new_height = int(height * 0.85)

        if new_width < 600 or new_height < 300:
            break

        img = img.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        for quality in quality_values:
            output = BytesIO()

            img.save(
                output,
                format="JPEG",
                quality=quality,
                optimize=True
            )

            compressed = output.getvalue()

            if len(compressed) <= OCR_MAX_BYTES:
                return {
                    "bytes": compressed,
                    "filename": "compressed_ocr.jpg",
                    "content_type": "image/jpeg",
                    "compressed": True,
                    "original_size": original_size,
                    "final_size": len(compressed),
                    "quality": quality,
                    "width": new_width,
                    "height": new_height
                }

    return {
        "bytes": image_bytes,
        "filename": filename,
        "content_type": "application/octet-stream",
        "compressed": False,
        "original_size": original_size,
        "final_size": original_size,
        "warning": "Could not compress image below OCR limit"
    }


@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(None)
):

    headers = {
        "apikey": OCR_KEY
    }

    compression_info = None

    if file:
        raw_file = await file.read()

        prepared = compress_image_for_ocr(
            raw_file,
            file.filename
        )

        compression_info = {
            "compressed": prepared.get("compressed"),
            "original_size": prepared.get("original_size"),
            "final_size": prepared.get("final_size"),
            "quality": prepared.get("quality"),
            "width": prepared.get("width"),
            "height": prepared.get("height"),
            "warning": prepared.get("warning")
        }

        files = {
            "file": (
                prepared["filename"],
                prepared["bytes"],
                prepared["content_type"]
            )
        }

        data = {
            "language": "eng",
            "OCREngine": "2",
            "scale": "true",
            "detectOrientation": "true",
            "isOverlayRequired": "false"
        }

        response = requests.post(
            "https://api.ocr.space/parse/image",
            headers=headers,
            files=files,
            data=data,
            timeout=60
        )

    else:
        body = await request.json()
        url = body.get("url")

        if not url:
            return {
                "status": "error",
                "error": "No URL provided"
            }

        data = {
            "url": url,
            "language": "eng",
            "OCREngine": "2",
            "scale": "true",
            "detectOrientation": "true",
            "isOverlayRequired": "false"
        }

        response = requests.post(
            "https://api.ocr.space/parse/image",
            headers=headers,
            data=data,
            timeout=60
        )

    if response.status_code != 200:
        return {
            "status": "error",
            "type": "ocr_api_error",
            "message": "OCR API request failed.",
            "details": response.text,
            "compression": compression_info
        }

    result = response.json()

    ocr_error = result.get("error", "")

    if ocr_error:
        if "E556" in ocr_error:
            return {
                "status": "error",
                "type": "ocr_file_too_large",
                "message": (
                    "Image is too large for OCR. "
                    "Please upload an image no larger than 1.5 MB."
                ),
                "details": result,
                "sizeBytes": result.get("sizeBytes"),
                "compression": compression_info
            }

        return {
            "status": "error",
            "type": "ocr_error",
            "message": ocr_error,
            "details": result,
            "compression": compression_info
        }

    if result.get("IsErroredOnProcessing"):
        return {
            "status": "error",
            "type": "ocr_processing_failed",
            "message": "OCR processing failed.",
            "details": result,
            "compression": compression_info
        }

    parsed_results = result.get("ParsedResults", [])

    if not parsed_results:
        return {
            "status": "ok",
            "text_found": False,
            "text": "",
            "message": "Text was not found on this image.",
            "compression": compression_info
        }

    parsed = parsed_results[0]
    text = parsed.get("ParsedText", "").strip()

    if not text:
        return {
            "status": "ok",
            "text_found": False,
            "text": "",
            "message": "Text was not found on this image.",
            "compression": compression_info
        }

    return {
        "status": "ok",
        "text_found": True,
        "text": text,
        "compression": compression_info
    }
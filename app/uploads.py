from pathlib import PurePosixPath

from fastapi import UploadFile

from app.config import Settings
from app.errors import ServiceError

SUPPORTED_EXTENSIONS = ("wav", "mp3", "m4a", "webm", "ogg", "mp4")


def check_filename(filename: str | None):
    extension = PurePosixPath((filename or "").replace("\\", "/")).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_EXTENSIONS:
        raise ServiceError(415, "UNSUPPORTED_AUDIO", "지원 형식: wav, mp3, m4a, webm, ogg, mp4")


async def read_audio(file: UploadFile, settings: Settings) -> bytes:
    try:
        check_filename(file.filename)
        # Do not trust Content-Length, MIME type or filename as proof of valid audio.
        content = bytearray()
        while chunk := await file.read(1024 * 1024):
            content.extend(chunk)
            if len(content) > settings.max_upload_bytes:
                raise ServiceError(413, "FILE_TOO_LARGE", "허용된 파일 크기를 초과했습니다.")
        if not content:
            raise ServiceError(422, "EMPTY_FILE", "빈 파일은 업로드할 수 없습니다.")
        return bytes(content)
    finally:
        await file.close()

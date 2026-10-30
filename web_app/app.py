# -*- coding: utf-8 -*-
"""
Heritage XAI: FastAPI Application Server
========================================
Server cung cấp API phân tích Đa phương thức và giao diện Web Editorial Light Mode
cho Di sản Văn hóa Phi vật thể Việt Nam (Ca trù, Chèo, Cải lương, Chầu văn, Quan họ, Hát xẩm).
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from web_app.inference_engine import inference_engine, GENRES, GENRE_COLORS

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
TEMP_DIR = BASE_DIR / "temp"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Heritage Multimodal XAI",
    description="Hệ thống Phân lớp & Giải thích Đa phương thức Di sản Văn hóa Phi vật thể Việt Nam",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class InterveneRequest(BaseModel):
    base_probs: Optional[Dict[str, float]] = None
    current_probs: Dict[str, float]
    concept_id: str
    is_enabled: bool
    all_concepts: Optional[List[Dict[str, Any]]] = None


class SampleAnalyzeRequest(BaseModel):
    sample_path: str
    mode: str = "video"


class UrlAnalyzeRequest(BaseModel):
    url: str
    mode: str = "video"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Giao diện index.html chưa được khởi tạo.")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "genres": GENRES,
        "engine": "Neuro-Symbolic CBM + Fuzzy Rules + Audio-Reactive 3D"
    }


@app.get("/api/sample_files")
async def list_sample_files():
    """
    Trả về danh sách 6 mẫu đại diện tiêu biểu (1 mẫu / 1 thể loại)
    có sẵn trong thư mục dataset/video/ để người dùng trải nghiệm tức thì.
    """
    samples = [
        {
            "genre": "Ca trù",
            "title": "Tây Hồ Hoài Cổ (Đào nương, Phách & Đàn đáy)",
            "file_path": "dataset/video/catru/catru_tayhohoaico_03.mp4",
            "mode": "video",
            "duration": "01:24",
            "size": "4.0 MB",
            "color": "#C84B31"
        },
        {
            "genre": "Cải lương",
            "title": "Đời Cô Lựu (Vọng cổ & Đàn guitar phím lõm)",
            "file_path": "dataset/video/cailuong/cailuong_doicoluu_06.mp4",
            "mode": "video",
            "duration": "01:12",
            "size": "5.9 MB",
            "color": "#C97A1B"
        },
        {
            "genre": "Chèo",
            "title": "Lưu Bình Dương Lễ (Hề chèo, Trống & Yếm đào)",
            "file_path": "dataset/video/cheo/cheo_luubinhduongle_08.mp4",
            "mode": "video",
            "duration": "00:45",
            "size": "2.3 MB",
            "color": "#2E5A88"
        },
        {
            "genre": "Chầu văn",
            "title": "Văn Quán Quê Tôi (Hát Văn Hầu Đồng, Khăn chầu)",
            "file_path": "dataset/video/chauvan/chauvan_vanquanquetoi.mp4",
            "mode": "video",
            "duration": "08:26",
            "size": "42.5 MB",
            "color": "#8C3B88"
        },
        {
            "genre": "Quan họ",
            "title": "Cây Trúc Xinh (Nón quai thao & Áo tứ thân)",
            "file_path": "dataset/video/quanho/quanho_caytrucxinh.mp4",
            "mode": "video",
            "duration": "03:22",
            "size": "17.8 MB",
            "color": "#1E6B5B"
        },
        {
            "genre": "Hát xẩm",
            "title": "Anh Đừng Yêu Em (Đàn nhị, Chiếu xẩm phố cổ)",
            "file_path": "dataset/video/xam/xam_anhdungyeu.mp4",
            "mode": "video",
            "duration": "07:24",
            "size": "26.5 MB",
            "color": "#7A6248"
        }
    ]
    return {"status": "success", "samples": samples}


@app.post("/api/analyze")
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    mode: str = Form("audio")
):
    """
    API tiếp nhận tệp tải lên (Video, Audio hoặc Bản ghi âm từ Microphone).
    Thực hiện quy chuẩn Chunking 3-5s, trích xuất 7 trục MPEG-7, đối chiếu luật mờ và sinh XAI.
    """
    # Làm sạch tên file và lưu vào thư mục temp
    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    if not safe_name:
        safe_name = f"recorded_audio_{os.urandom(4).hex()}.wav"

    save_path = TEMP_DIR / safe_name
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Tự động điều chỉnh mode nếu phát hiện đuôi video
        ext = save_path.suffix.lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            mode = "video"
        elif ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            if mode not in ["mic", "audio"]:
                mode = "audio"

        # Phân tích qua Inference Engine
        result = inference_engine.analyze(save_path, mode=mode)

        # Cung cấp URL phát media cho trình duyệt
        result["media_url"] = f"/api/media/temp/{safe_name}"
        result["file_name"] = safe_name

        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích đa phương thức: {str(e)}")


@app.post("/api/analyze_sample")
async def analyze_sample_file(req: SampleAnalyzeRequest):
    """
    Phân tích trực tiếp một mẫu có sẵn trong hệ thống mà không cần người dùng tải lên.
    """
    rel_path = req.sample_path.strip().replace("\\", "/")
    full_path = ROOT_DIR / rel_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy mẫu tại: {rel_path}")

    try:
        result = inference_engine.analyze(full_path, mode=req.mode)
        # Tạo đường dẫn stream media
        result["media_url"] = f"/api/media_stream?path={rel_path}"
        result["file_name"] = full_path.name
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích mẫu: {str(e)}")


@app.post("/api/intervene")
async def intervene_concept(req: InterveneRequest):
    """
    API Can thiệp Khái niệm CBM (Interactive Concept Bottleneck Intervention).
    Tái cân bằng xác suất 6 thể loại theo thời gian thực khi người dùng bật/tắt Concept.
    """
    try:
        res = inference_engine.intervene(
            current_probs=req.current_probs,
            toggled_concept_id=req.concept_id,
            is_enabled=req.is_enabled,
            all_concepts=req.all_concepts or [],
            base_probs=req.base_probs
        )
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi can thiệp CBM: {str(e)}")


@app.api_route("/api/media/temp/{filename}", methods=["GET", "HEAD"])
async def get_temp_media(filename: str, request: Request):
    """
    Phát file media tạm hỗ trợ HTTP Range requests để tua (seek) mượt mà trên Safari/Chrome.
    """
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp media")

    return _stream_file_with_range(file_path, request)


@app.api_route("/api/media_stream", methods=["GET", "HEAD"])
async def stream_media_file(path: str, request: Request):
    """
    Stream file media từ kho dữ liệu dataset/video/ hỗ trợ Range requests.
    """
    clean_path = path.strip().replace("\\", "/")
    full_path = ROOT_DIR / clean_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file media")

    return _stream_file_with_range(full_path, request)


@app.post("/api/analyze_url")
async def analyze_url_video(req: UrlAnalyzeRequest):
    """
    Tải và phân tích video từ YouTube hoặc liên kết URL trực tiếp bằng yt-dlp.
    """
    raw_url = req.url.strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp URL video hợp lệ")

    import yt_dlp
    import uuid
    import re

    # Làm sạch URL YouTube: Loại bỏ các tham số playlist thừa (&list=..., &index=..., &start_radio=...)
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', raw_url)
    clean_url = f"https://www.youtube.com/watch?v={match.group(1)}" if match else raw_url

    file_id = f"yt_{uuid.uuid4().hex[:8]}"
    out_template = str(TEMP_DIR / f"{file_id}.%(ext)s")

    ydl_opts = {
        'format': 'bestvideo[height<=480][vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[height<=480]+bestaudio/best[height<=480][ext=mp4]/best',
        'outtmpl': out_template,
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'merge_output_format': 'mp4',
        'max_filesize': 150 * 1024 * 1024,
    }

    try:
        title = "Video từ URL"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=True)
            if info:
                title = info.get('title', 'Video từ URL')
            filename = ydl.prepare_filename(info) if info else None
            saved_path = Path(filename) if filename else None
            if not saved_path or not saved_path.exists():
                candidates = [p for p in TEMP_DIR.glob(f"{file_id}.*") if not p.name.endswith(".part")]
                candidates.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
                if candidates:
                    saved_path = candidates[0]

        if not saved_path or not saved_path.exists():
            raise FileNotFoundError("Không thể lưu tệp video từ URL")

        # Phân tích qua engine
        result = inference_engine.analyze(saved_path, mode="video")
        result["media_url"] = f"/api/media/temp/{saved_path.name}"
        result["file_name"] = title

        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi tải hoặc phân tích video URL: {str(e)}")


def _stream_file_with_range(file_path: Path, request: Request):
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp media")

    stat = file_path.stat()
    file_size = stat.st_size
    range_header = request.headers.get("range")

    ext = file_path.suffix.lower()
    content_type = "video/mp4" if ext in [".mp4", ".mov", ".avi", ".mkv"] else "audio/wav" if ext == ".wav" else "audio/mpeg"

    # Hỗ trợ yêu cầu HEAD từ trình duyệt để nạp metadata
    if request.method == "HEAD":
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        }
        return Response(content=b"", status_code=200, headers=headers)

    if range_header:
        try:
            h = range_header.replace("bytes=", "").strip()
            parts = h.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            if end >= file_size:
                end = file_size - 1
            length = end - start + 1

            with open(file_path, "rb") as f:
                f.seek(start)
                data = f.read(length)

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(data)),
                "Content-Type": content_type,
            }
            return Response(content=data, status_code=206, headers=headers)
        except Exception:
            return FileResponse(file_path, media_type=content_type, headers={"Accept-Ranges": "bytes"})
    else:
        return FileResponse(file_path, media_type=content_type, headers={"Accept-Ranges": "bytes"})


if __name__ == "__main__":
    uvicorn.run("web_app.app:app", host="127.0.0.1", port=8000, reload=True)

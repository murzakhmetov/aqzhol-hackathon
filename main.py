import os
import io
import json
import uuid
import base64
import threading
import traceback
from collections import deque
from datetime import datetime
from typing import List

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from util.inference_utils import inference, create_model
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")

SUPABASE_STORAGE_ACCESS_KEY_ID = os.getenv("SUPABASE_STORAGE_ACCESS_KEY_ID", "")
SUPABASE_STORAGE_SECRET_ACCESS_KEY = os.getenv("SUPABASE_STORAGE_SECRET_ACCESS_KEY", "")



class Parameters(BaseModel):
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    netG: str = 'resnet_9blocks'
    norm: str = 'instance'
    init_type: str = 'xavier'
    init_gain: float = 0.02
    display_sides: int = 1
    num_classes: int = 1
    gpu_ids: List[int] = []
    isTrain: bool = False



app = FastAPI(title="Aqzhol — AI Road Crack Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="web-deepcrack"), name="static")



print(" Loading DeepCrack model...")
parameters = Parameters()
model = create_model(parameters)
print(" DeepCrack model loaded!")

# DeepCrackModel keeps per-call state (set_input -> test -> get_current_visuals),
# so inference calls running in the threadpool must not overlap.
model_lock = threading.Lock()


def run_inference(bytesImg: bytes, dim: tuple, unit: str):
    with model_lock:
        return inference(model, bytesImg, dim, unit)



# Records include base64 images, so keep only the most recent analyses in memory.
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "50"))
analysis_history: deque = deque(maxlen=MAX_HISTORY)



def image_to_base64(img_array: np.ndarray) -> str:
   
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return encoded


def sanitize_for_json(obj):
    
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def check_cracks_detected(visuals: dict) -> tuple:
    
    fused = visuals['fused']
    if len(fused.shape) == 3:
        gray = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
    else:
        gray = fused
    crack_pixels = np.sum(gray > 90)
    total_pixels = gray.shape[0] * gray.shape[1]
    crack_ratio = crack_pixels / total_pixels
    detected = bool(crack_ratio > 0.005)
    confidence = float(crack_ratio * 100)
    return detected, confidence


async def analyze_with_groq(image_base64: str, crack_detected: bool) -> dict:
    
    import httpx
    if not crack_detected:
        return {
            "status": "Трещины не обнаружены",
            "crack_type": "Нет",
            "severity": "Нет повреждений",
            "severity_level": 0,
            "description": "На данном участке дороги трещины не обнаружены. Дорожное покрытие в удовлетворительном состоянии.",
            "repair_recommendation": "Ремонт не требуется",
            "repair_timeline": "Не применимо",
            "estimated_cost": "0 ₸",
            "risk_assessment": "Низкий — дорога в нормальном состоянии"
        }
    prompt = """Ты — эксперт-дорожный инженер. Проанализируй это изображение дорожного покрытия с обнаруженными трещинами.

Ответь СТРОГО в формате JSON (без markdown, без ```):
{
    "status": "Трещины обнаружены",
    "crack_type": "тип трещины (продольная/поперечная/сетка/аллигаторная/блочная)",
    "severity": "степень повреждения (легкая/средняя/тяжелая/критическая)",
    "severity_level": число от 1 до 10,
    "description": "подробное описание состояния дорожного покрытия на 2-3 предложения",
    "repair_recommendation": "конкретная рекомендация по ремонту",
    "repair_timeline": "рекомендуемый срок ремонта",
    "estimated_cost": "ориентировочная стоимость ремонта в тенге за м²",
    "risk_assessment": "оценка риска при бездействии"
}"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024
                }
            )
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]
                content = content.strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {
                        "status": "Трещины обнаружены",
                        "crack_type": "Требует ручной классификации",
                        "severity": "Средняя",
                        "severity_level": 5,
                        "description": content[:500],
                        "repair_recommendation": "Требуется осмотр специалиста",
                        "repair_timeline": "В течение 30 дней",
                        "estimated_cost": "Требует оценки",
                        "risk_assessment": "Средний риск"
                    }
            else:
                print(f"Groq API error: {response.status_code} — {response.text}")
                return {
                    "status": "Трещины обнаружены",
                    "crack_type": "Автоматический анализ недоступен",
                    "severity": "Требует ручной оценки",
                    "severity_level": 5,
                    "description": "Groq API временно недоступен. Трещины обнаружены моделью DeepCrack.",
                    "repair_recommendation": "Рекомендуется ручной осмотр",
                    "repair_timeline": "Как можно скорее",
                    "estimated_cost": "Требует оценки",
                    "risk_assessment": "Требует оценки специалиста"
                }
    except Exception as e:
        print(f"Groq API exception: {e}")
        return {
            "status": "Трещины обнаружены",
            "crack_type": "Ошибка анализа",
            "severity": "Неизвестно",
            "severity_level": 5,
            "description": f"Ошибка при обращении к Groq API: {str(e)}",
            "repair_recommendation": "Рекомендуется ручной осмотр",
            "repair_timeline": "Как можно скорее",
            "estimated_cost": "Требует оценки",
            "risk_assessment": "Требует оценки"
        }




@app.get("/")
def read_root():
    return {"status": "Aqzhol API is running", "version": "1.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True, "timestamp": datetime.now().isoformat()}


@app.post("/analyze")
async def analyze_road(
    file: UploadFile = File(...),
    width: float = Form(default=1.0),
    height: float = Form(default=1.0),
    unit: str = Form(default="m"),
    location: str = Form(default="Не указано"),
    notes: str = Form(default="")
):

    try:
        bytesImg = await file.read()
        # inference is CPU/GPU-bound and synchronous; keep it off the event loop.
        contour_img, visuals = await run_in_threadpool(run_inference, bytesImg, (width, height), unit)
        cracks_detected, crack_ratio = check_cracks_detected(visuals)
        original_b64 = base64.b64encode(bytesImg).decode('ascii')
        result_b64 = image_to_base64(contour_img)
        fused_b64 = image_to_base64(visuals['fused'])
        sides = {}
        for k in ['side1', 'side2', 'side3', 'side4', 'side5']:
            if k in visuals:
                sides[k] = image_to_base64(visuals[k])
        groq_analysis = await analyze_with_groq(result_b64, cracks_detected)
        analysis_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        record = {
            "id": analysis_id,
            "timestamp": timestamp,
            "filename": file.filename,
            "location": location,
            "notes": notes,
            "dimensions": f"{width}x{height} {unit}",
            "cracks_detected": bool(cracks_detected),
            "crack_coverage_percent": float(round(crack_ratio, 2)),
            "analysis": groq_analysis,
            "images": {
                "original": f"data:image/png;base64,{original_b64}",
                "result": f"data:image/png;base64,{result_b64}",
                "fused": f"data:image/png;base64,{fused_b64}",
                **{k: f"data:image/png;base64,{v}" for k, v in sides.items()}
            }
        }
        analysis_history.append(record)
        safe_record = sanitize_for_json(record)
        return Response(
            content=json.dumps(safe_record, ensure_ascii=False),
            media_type="application/json"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


@app.get("/history")
def get_history():
    
    # Everything except the (large) base64 images.
    return [{k: v for k, v in r.items() if k != "images"} for r in reversed(analysis_history)]


@app.get("/history/{analysis_id}")
def get_analysis(analysis_id: str):
  
    for r in analysis_history:
        if r["id"] == analysis_id:
            return r
    raise HTTPException(status_code=404, detail="Анализ не найден")


@app.get("/history/export/text", response_class=PlainTextResponse)
def export_history_text():
 
    if not analysis_history:
        return "История анализов пуста.\n"
    lines = []
    lines.append("=" * 80)
    lines.append("AQZHOL — ОТЧЁТ ПО АНАЛИЗУ ДОРОЖНОГО ПОКРЫТИЯ")
    lines.append(f"Дата выгрузки: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    lines.append(f"Всего анализов: {len(analysis_history)}")
    lines.append("=" * 80)
    lines.append("")
    for i, r in enumerate(reversed(analysis_history), 1):
        a = r["analysis"]
        lines.append(f" Анализ #{i} ({'ID: ' + r['id']}) ")
        lines.append(f"  Дата/время:        {r['timestamp']}")
        lines.append(f"  Файл:              {r['filename']}")
        lines.append(f"  Локация:           {r['location']}")
        lines.append(f"  Размеры:           {r['dimensions']}")
        if r['notes']:
            lines.append(f"  Примечания:        {r['notes']}")
        lines.append(f"  Трещины:           {'Обнаружены' if r['cracks_detected'] else 'Не обнаружены'}")
        lines.append(f"  Покрытие трещин:   {r['crack_coverage_percent']}%")
        lines.append(f"  Статус:            {a.get('status', 'N/A')}")
        lines.append(f"  Тип трещины:       {a.get('crack_type', 'N/A')}")
        lines.append(f"  Степень:           {a.get('severity', 'N/A')} ({a.get('severity_level', 'N/A')}/10)")
        lines.append(f"  Описание:          {a.get('description', 'N/A')}")
        lines.append(f"  Рекомендация:      {a.get('repair_recommendation', 'N/A')}")
        lines.append(f"  Срок ремонта:      {a.get('repair_timeline', 'N/A')}")
        lines.append(f"  Ориент. стоимость: {a.get('estimated_cost', 'N/A')}")
        lines.append(f"  Оценка риска:      {a.get('risk_assessment', 'N/A')}")
        lines.append("")
    lines.append("=" * 80)
    lines.append("Конец отчёта")
    lines.append("=" * 80)
    return "\n".join(lines)
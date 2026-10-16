from pathlib import Path
import json
import os
import threading
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen
import webbrowser

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
import uvicorn


BASE_DIR = Path(__file__).resolve().parent
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

app = FastAPI(
    title="Smart Agriculture Advisor",
    description="AI crop recommendations from a local Ollama model. No database required.",
)
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class CropQuery(BaseModel):
    soil_type: Literal["Loamy", "Clay", "Sandy", "Silt"]
    temperature: float = Field(ge=-20, le=60, description="Average temperature in °C")
    rainfall: float = Field(ge=0, le=10000, description="Annual rainfall in mm")


class CropAdvice(BaseModel):
    recommended_crops: list[str] = Field(min_length=1, max_length=3)
    reasoning: str
    advisory_tips: list[str] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)
    caution: str


class CropRecommendation(CropQuery):
    model: str
    recommended_crops: list[str]
    reasoning: str
    advisory_tips: list[str]
    next_steps: list[str]
    caution: str


def _request_ollama(query: CropQuery) -> CropAdvice:
    prompt = (
        "You are a careful agricultural advisor. Use the farmer's conditions to "
        "suggest suitable crops and practical, locally adaptable advice. Do not "
        "invent soil-test results, guarantee yields, or prescribe exact chemical "
        "or pesticide doses. Mention that local climate, variety, and soil testing "
        "can change the recommendation. Return only a JSON object with these keys: "
        '"recommended_crops" (array of 1 to 3 crop names), "reasoning" (short '
        "explanation), \"advisory_tips\" (array of practical tips), "
        '"next_steps" (array of useful actions), and "caution" (one concise '
        "qualification).\n\n"
        f"Field conditions: soil={query.soil_type}, "
        f"average_temperature_c={query.temperature}, "
        f"annual_rainfall_mm={query.rainfall}."
    )
    request_body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3},
        }
    ).encode("utf-8")
    request = UrlRequest(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=90) as response:
            ollama_result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {error.code}. Check the local Ollama service.",
        ) from error
    except (URLError, TimeoutError, ConnectionError) as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot connect to Ollama. Start Ollama and make sure the "
                f"'{OLLAMA_MODEL}' model is installed."
            ),
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an unreadable response. Try again or use another model.",
        ) from error

    if not isinstance(ollama_result, dict) or not isinstance(
        ollama_result.get("response"), str
    ):
        raise HTTPException(
            status_code=502,
            detail="Ollama's response did not include generated advice.",
        )

    try:
        return CropAdvice.model_validate_json(ollama_result["response"])
    except ValidationError as error:
        raise HTTPException(
            status_code=502,
            detail="The AI response did not match the expected advice format. Please try again.",
        ) from error


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/recommend")
async def recommend_crop(query: CropQuery) -> CropRecommendation:
    advice = await run_in_threadpool(_request_ollama, query)
    return CropRecommendation(
        **query.model_dump(),
        model=OLLAMA_MODEL,
        **advice.model_dump(),
    )


if __name__ == "__main__":
    threading.Timer(1, webbrowser.open, args=("http://127.0.0.1:8000/",)).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)

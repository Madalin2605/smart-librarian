import os, re, base64
from io import BytesIO
from typing import Optional, List
from PIL import Image
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_DIR = "outputs/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TITLE_HINT = re.compile(r"Recomandare:\s*(.+)", re.IGNORECASE | re.UNICODE)

def extract_chosen_title(reply_text: str) -> Optional[str]:
    if not reply_text:
        return None
    m = TITLE_HINT.search(reply_text)
    if m:
        return m.group(1).strip(" '\"\t")
    for line in reply_text.splitlines():
        ln = line.strip(" '\"\t")
        if ln:
            return ln
    return None

def _slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")

def _build_prompt(title: str, themes: Optional[List[str]] = None, lang: str = "ro") -> str:
    t = ", ".join(themes or ["prietenie", "aventură"])
    if lang == "ro":
        return (
            f"Ilustrație originală, tip poster, inspirată de cartea „{title}”. "
            f"Nu reproduce sau imita coperți existente sau materiale protejate de drepturi de autor. "
            f"Evidențiază vizual temele: {t}. "
            "Stil cinematic, compoziție clară, detalii bogate, luminozitate echilibrată."
        )
    return (
        f"Original poster-style illustration inspired by the book '{title}'. "
        f"Do not recreate or imitate existing covers or copyrighted material. "
        f"Highlight themes: {t}. Cinematic, clear composition, rich details, balanced lighting."
    )

def generate_book_image(title: str, themes: Optional[List[str]] = None, size: str = "1024x1024", lang: str = "ro") -> str:
    if not title or not title.strip():
        raise ValueError("Nu am un titlu valid pentru generarea imaginii.")
    prompt = _build_prompt(title, themes, lang)

    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        n=1,
        response_format="b64_json"
    )

    if not result or not getattr(result, "data", None) or not result.data:
        raise RuntimeError("API nu a returnat niciun rezultat de imagine.")
    b64 = getattr(result.data[0], "b64_json", None)
    if not b64:
        rp = getattr(result.data[0], "revised_prompt", None)
        raise RuntimeError(f"Nu am primit payload de imagine. (revised_prompt={rp!r})")

    img_bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    filename = f"{_slugify(title)}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path, format="PNG")
    return path

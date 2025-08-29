import os, re, base64
from io import BytesIO
from typing import Optional, List
from PIL import Image
from openai import OpenAI


# Initialize OpenAI client using API key from local environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_DIR = "outputs/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TITLE_HINT = re.compile(r"Recomandare:\s*(.+)", re.IGNORECASE | re.UNICODE)


def extract_chosen_title(reply_text: str) -> Optional[str]:
    """
    Extract a book title from an assistant's reply.

    The function first tries to match the pattern "Recomandare: <titlu>".
    If that fails, it falls back to returning the first non-empty line.

    Args:
        reply_text (str): The assistant reply text.

    Returns:
        Optional[str]: The detected book title, or None if nothing is found.
    """
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
    """
    Convert a string into a safe filename.

    Keeps only alphanumeric characters, replaces others with dashes,
    converts to lowercase, and trims extra dashes at the ends.

    Args:
        name (str): Input string (e.g., book title).

    Returns:
        str: A slugified string safe for filenames.
    """
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")


def _build_prompt(title: str, themes: Optional[List[str]] = None, lang: str = "ro") -> str:
    """
    Build an illustration prompt for DALL·E image generation.

    Args:
        title (str): Book title.
        themes (Optional[List[str]]): Themes to emphasize visually (default: ["prietenie", "aventură"]).
        lang (str): Language of the prompt ("ro" for Romanian, otherwise English).

    Returns:
        str: The formatted prompt string for image generation.
    """
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
    """
    Generate an AI illustration for a book using OpenAI's DALL·E 3 model.

    - Builds a descriptive prompt from the given title and themes.
    - Calls the OpenAI Images API to generate one image.
    - Decodes the base64 response into an image.
    - Saves the image as PNG inside `outputs/images/`.
    - Returns the path to the saved file.

    Args:
        title (str): The exact book title.
        themes (Optional[List[str]]): List of themes to emphasize (default: ["prietenie", "aventură"]).
        size (str): Output image size (default: "1024x1024").
        lang (str): Prompt language ("ro" for Romanian, otherwise English).

    Returns:
        str: Path to the saved PNG file.

    Raises:
        ValueError: If no valid title is provided.
        RuntimeError: If the API call fails or no image payload is returned.
    """
    if not title or not title.strip():
        raise ValueError("Nu am un titlu valid pentru generarea imaginii.")
    
    # Step 1: Build prompt text
    prompt = _build_prompt(title, themes, lang)

    # Step 2: Call OpenAI Images API
    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        n=1,
        response_format="b64_json"
    )

    # Step 3: Defensive checks
    if not result or not getattr(result, "data", None) or not result.data:
        raise RuntimeError("API nu a returnat niciun rezultat de imagine.")
    b64 = getattr(result.data[0], "b64_json", None)
    if not b64:
        rp = getattr(result.data[0], "revised_prompt", None)
        raise RuntimeError(f"Nu am primit payload de imagine. (revised_prompt={rp!r})")

    # Step 4: Decode base64, save as PNG
    img_bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    filename = f"{_slugify(title)}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path, format="PNG")
    
    return path


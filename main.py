import os
import time
import base64
import httpx
from io import BytesIO
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import replicate
from PIL import Image

# --- CONFIGURATION ---
# PASTE YOUR REPLICATE API KEY BETWEEN THE QUOTES BELOW!
os.environ["REPLICATE_API_TOKEN"] = os.environ.get("REPLICATE_API_TOKEN", "")

MODEL_ID = "google/nano-banana-pro"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_with_retry(instruction, image_file, retries=3):
    for attempt in range(retries):
        try:
            image_file.seek(0)
            output = replicate.run(
                MODEL_ID,
                input={
                    "prompt": instruction,
                    "image_input": [image_file],
                    "prompt_strength": 0.70,
                    "resolution": "2K",
                    "output_format": "png",
                    "safety_filter_level": "block_only_high"
                }
            )
            return output
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Attempt {attempt+1} failed: {error_msg}")
            if "429" in error_msg or "503" in error_msg or "unavailable" in error_msg:
                if attempt < retries - 1:
                    print("⏳ Waiting 10 seconds for Replicate to cool down...")
                    time.sleep(10)
                    continue
            raise e
    return None

@app.post("/fix-my-tattoo")
async def fix_tattoo(
    file: UploadFile = File(...),
    tattoo_description: str = Form("a tattoo"),
    mode: str = Form("Restore"),
    custom_request: str = Form("")
):
    print(f"Processing. Mode: {mode}")

    file_bytes = await file.read()
    input_img = Image.open(BytesIO(file_bytes)).convert("RGB")
    input_img.thumbnail((1024, 1024))

    buffered = BytesIO()
    input_img.save(buffered, format="JPEG")
    buffered.seek(0)

    if mode == "Restore":
        instruction = f"A photorealistic close-up photograph of human skin with a freshly inked black and grey realistic tattoo of {tattoo_description}. Fine needle details, smooth shading, professional tattoo artist work. Natural skin texture and lighting. Do NOT add a background."
    else: 
        request = custom_request if custom_request else "fix this tattoo"
        instruction = f"A photorealistic close-up photograph of a professional tattoo on human skin: {request}. Fine needle details, smooth shading, natural skin texture and lighting."

    results = []

    print("Generating photorealistic render...")

    try:
        output = run_with_retry(instruction, buffered)

        if output:
            ai_url = str(output[0]) if isinstance(output, list) else str(output)

            img_response = httpx.get(ai_url)
            final_img_base64 = base64.b64encode(img_response.content).decode("utf-8")
            final_data_uri = f"data:image/jpeg;base64,{final_img_base64}"

            results.append({
                "title": "Realistic Restoration",
                "description": "Photorealistic shading and skin blending.",
                "image_url": final_data_uri
            })
        else:
            print(f"❌ Skipped after max retries.")

    except Exception as e:
        print(f"❌ Error: {e}")

    return {"stages": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

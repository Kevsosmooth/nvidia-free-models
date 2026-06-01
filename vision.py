#!/usr/bin/env python3
"""Test NVIDIA vision (image-understanding) models with a local image.

Usage:
    python vision.py test_image.png "What's in this image?"
    python vision.py -m microsoft/phi-3-vision-128k-instruct photo.jpg "Read the text"

Sends the image as a base64 data URI in OpenAI structured-content format,
which the Llama-3.2 / Phi vision NIMs accept.
"""
import argparse
import base64
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"


def main() -> None:
    p = argparse.ArgumentParser(description="Test NVIDIA vision models on a local image.")
    p.add_argument("image", help="path to a local image (png/jpg)")
    p.add_argument("prompt", nargs="*", default=["Describe this image in detail."],
                   help="question about the image")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help="vision model id (default: %(default)s)")
    args = p.parse_args()

    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        sys.exit("NVIDIA_API_KEY not set (see .env).")
    if not os.path.exists(args.image):
        sys.exit(f"Image not found: {args.image}")

    ext = os.path.splitext(args.image)[1].lstrip(".").lower() or "png"
    if ext == "jpg":
        ext = "jpeg"
    with open(args.image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = " ".join(args.prompt) if args.prompt else "Describe this image in detail."

    client = OpenAI(base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                    api_key=key)

    print(f"--- {args.model} ---")
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
            ],
        }],
        max_tokens=512,
        temperature=0.2,
        stream=False,
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()

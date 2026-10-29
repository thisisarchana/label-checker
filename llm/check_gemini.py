"""Checks the Gemini API key, lists the models it can use, and makes one small test call."""
import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is not set. Put it in a file named .env in the project folder:")
        print("  GEMINI_API_KEY=your-key-here")
        return 1

    client = genai.Client(api_key=key)
    try:
        models = [m for m in client.models.list()
                  if "generateContent" in (getattr(m, "supported_actions", None) or [])]
    except Exception as error:   # wrong key, no network, quota
        print(f"Could not list models: {error}")
        return 1

    print(f"Models your key can call with generateContent ({len(models)}):")
    for model in models:
        print(" ", model.name.removeprefix("models/"))

    chosen = os.environ.get("GEMINI_MODEL")
    if not chosen:
        print("\nPick one of the names above and add it to .env, for example:")
        print("  GEMINI_MODEL=<model name>")
        print("A 'flash' or 'flash-lite' model is the right choice for the free tier.")
        return 0

    try:
        reply = client.models.generate_content(model=chosen, contents="Reply with the single word: ready")
        print(f"\nTest call to {chosen}: {reply.text.strip()}")
    except Exception as error:
        print(f"\nTest call to {chosen} failed: {error}")
        print("A 429 error means the free-tier limit for that model was hit; try a lighter model or wait.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
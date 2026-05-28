import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

openai_api_port = os.getenv("OPENAI_API_PORT", "8000")
api_key = os.getenv("API_KEY", "changeme-local-token")

client = OpenAI(
    base_url=f"http://localhost:{openai_api_port}/v1",
    api_key=api_key,
)

result = client.images.generate(
    model="z-image-turbo-aio-fp8",
    prompt='A photorealistic storefront with a sign reading "LINKSPIX" in large readable letters, clean daylight, professional ad photo.',
    size="1024x1024",
    n=1,
    response_format="b64_json",
    extra_body={
        "seed": 123,
        "steps": 9,
        "guidance_scale": 1.0,
        "sampler_name": "res_multistep",
        "scheduler": "simple",
    },
)
print(result.data[0].b64_json[:120] + "...")

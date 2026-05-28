from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="changeme-local-token",
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

from openai import OpenAI

NAME_TO_URL = {
    "ali": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "llama": "https://api.llama-api.com",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": None
}

def access_client(api_key, backend):
    base_url = NAME_TO_URL[backend]
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model_name = "gemini-3.6-flash"

llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key,
    thinking_level="minimal",
    max_output_tokens=2048,
    timeout=60,
    max_retries=1
)

print("Testing model:", model_name)

start = time.perf_counter()

try:
    response = llm.invoke(
        "What is Java? Explain in 3 simple sentences."
    )

    end = time.perf_counter()

    print("\nModel used:",
          response.response_metadata.get("model_name"))

    print("\nAnswer:")

    content = response.content

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                print(item.get("text", ""))
    else:
        print(content)

    print("\nFinish reason:",
          response.response_metadata.get("finish_reason"))

    print(f"\nGeneration Time: {end - start:.2f} seconds")

except Exception as error:
    print("\nGemini Error:")
    print(error)
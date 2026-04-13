"""Gemini API client wrapper."""

import os
from google import genai
from google.genai import types


class GeminiClient:
    """Thin wrapper around the Google Gen AI SDK."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.model_name = model
        self._client = genai.Client(api_key=self.api_key)

    def generate_video_prompt(self, topic: str, style: str = "cinematic") -> str:
        """Use Gemini to create a detailed video generation prompt from a topic."""
        system_instruction = (
            "You are a creative director specializing in short-form video scripts. "
            "Given a topic and visual style, produce a concise but vivid video prompt "
            "suitable for an AI video generation model. "
            "Respond with the prompt text only — no extra commentary."
        )
        user_message = f"Topic: {topic}\nStyle: {style}"
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[system_instruction, user_message],
            config=types.GenerateContentConfig(
                temperature=0.9,
                max_output_tokens=256,
            ),
        )
        return response.text.strip()

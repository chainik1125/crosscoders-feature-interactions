from typing import Dict, List
import math
from functools import lru_cache
from pathlib import Path
import yaml
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

from sleepers.autointerp.util.autointerp_prompts import SYSTEM_EXPLAINER, SYSTEM_EVALUATOR, SYSTEM_INTERACTION_EVALUATOR

# Load environment variables from .env file
load_dotenv()

endpoint = "https://ft-test-sweden.openai.azure.com/"
deployment = "gpt-4o-N2"

api_version = "2024-12-01-preview"

client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

class azureAutointerp:
    def __init__(self):
        self.client = client

    def format_explainer_prompt(self, examples: str) -> str:

        formatted_examples = []
        for example_type in examples:
            for i in range(len(examples[example_type])):
                formatted_examples.append(f'{example_type}, Value: {round(examples[example_type][i]["activation"], 3)}, Sequence: {examples[example_type][i]["context"]} \n')
        formatted_examples = "".join(formatted_examples)
        message = [
            {"role": "system", "content": SYSTEM_EXPLAINER},
            {"role": "user", "content": formatted_examples}
        ]

        return message

    def format_evaluator_prompt(self, explanation: str, sequences: List[str]) -> str:
        message = [
            {"role": "system", "content": SYSTEM_EVALUATOR},
            {"role": "user", "content": f"EXPLANATION: {explanation}\nSEQUENCES: {sequences}"}
        ]
        return message

    def format_interaction_evaluator_prompt(self, cluster_explanations: List[str], test_explanations: str) -> str:
        cluster_explanations = "\n".join([f'{explanation}\n' for i, explanation in enumerate(cluster_explanations)])
        test_explanations = "\n".join([f'{i}: {explanation}\n' for i, explanation in enumerate(test_explanations)])
        message = [
            {"role": "system", "content": SYSTEM_INTERACTION_EVALUATOR},
            {"role": "user", "content": f"CLUSTER EXPLANATIONS: {cluster_explanations}\nTEST EXPLANATIONS: {test_explanations}"}
        ]
        return message
                

    def generate_autointerp(self, prompt: list) -> str:
        response = self.client.chat.completions.create(
            model=deployment,
            messages=prompt,
            temperature=0.7,
            max_tokens=1500,
            top_p=0.95
        )

        if response.choices and \
           response.choices[0].message and \
           response.choices[0].message.content is not None:
            return response.choices[0].message.content.strip()
        else:
            # Log the issue and the full response for debugging
            finish_reason = "Unknown"
            if response.choices and response.choices[0] and hasattr(response.choices[0], 'finish_reason'):
                finish_reason = response.choices[0].finish_reason
            
            print(f"Warning: API returned no content for prompt. Finish reason: {finish_reason}. Full response: {response}")
            # On failure, return empty string so downstream code can handle missing explanation
            return ""
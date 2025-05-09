from dotenv import load_dotenv
import os
import json
from pathlib import Path
from openai import OpenAI
import openai
from logger import get_logger
logger = get_logger(__name__)



def load_api(file_name):
    
    load_dotenv()
    GEMINI_API_KEY = os.getenv(file_name)
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set in .env")
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    logger.info("GEMINI_API_KEY loaded successfully.")
    return GEMINI_API_KEY


def load_prompt(path):
    path = Path(path)
    if not path.exists():
        logger.error(f"{path} does not exist.")
        raise FileNotFoundError(f"{path} does not exist.")

    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    logger.info(f"Loaded prompts from {path}")
    return prompts

def code_generation(api_key,prompt):
   try:
        logger.info("Initializing OpenAI client.")
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        logger.info("Sending request to OpenAI.")
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "You are a code generation expert. When given a prompt, you will output correct, complete Python code without commentary."},
                {"role": "user", "content": prompt}
            ]
        )

        logger.info("Response received from OpenAI.")
        return response.choices[0].message.content
   
   except openai.APIError as e:
        logger.exception("OpenAI API call failed.")
        raise
   
   except Exception as e:
        logger.exception("Unexpected error during OpenAI API call.")
        raise

def save_generated_output(response, branch_name: str, output_dir: str = "outputs/gemini"):
    try:
        generated = response
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        

        outfile = output_dir_path / f"{branch_name}.txt"
        outfile.write_text(generated, encoding="utf-8")
        
        logger.info(f"Output saved to {outfile}")
        
    except Exception as e:
        logger.error(f"An error occurred while saving the output: {e}")


def main():
    Env = "GEMINI_API_KEY"
    prompt_path = 'prompts/prompt.json'
    api_key = load_api(Env)
    prompts = load_prompt(prompt_path)

    for branch, prompt_text in prompts.items():
        logger.info(f"Processing branch: {branch}")
        response = code_generation(api_key, prompt_text)
        save_generated_output(response, branch)
        logger.info(f"Completed processing for branch: {branch}")


if __name__ == "__main__":
    main()

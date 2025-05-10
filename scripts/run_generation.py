from dotenv import load_dotenv
import os
import json
from pathlib import Path
from openai import OpenAI
import openai
from logger import get_logger
logger = get_logger('run_generation')
from ast import parse as ast_parse

def clean_wrappers(source: str) -> str:
    """Remove markdown code fences"""
    lines = source.splitlines()
    while lines and lines[0].strip().startswith("```"):
        lines.pop(0)
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines)

def validate_python_code(code: str) -> bool:
    """Check if code is syntactically valid"""
    try:
        ast_parse(code)
        return True
    except SyntaxError as e:
        logger.error(f"Invalid Python code: {str(e)}")
        return False

# Modified save function
def save_generated_output(response, branch_name: str, output_dir: str = "outputs/gemini"):
    try:
        cleaned_code = clean_wrappers(response)
        
        if not validate_python_code(cleaned_code):
            logger.error(f"Invalid Python code for {branch_name}")
            return False

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        outfile = output_dir_path / f"{branch_name.replace(' ', '_')}.py"
        
        if outfile.exists():
            logger.warning(f"Overwriting existing file: {outfile}")
            
        outfile.write_text(cleaned_code, encoding="utf-8")
        return True
        
    except Exception as e:
        logger.error(f"Save failed for {branch_name}: {str(e)}")
        return False



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
        # Ensure the output directory exists
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Construct filename with .py extension
        outfile = output_dir_path / f"{branch_name.replace(' ', '_')}.py"

        # Write the code string directly to a .py file
        outfile.write_text(response, encoding="utf-8")

        logger.info(f"Output saved to {outfile}")

    except Exception as e:
        logger.error(f"An error occurred while saving the code: {e}")


def main():
    logger.info("Starting the code generation process.")
    logger.info("Loading environment variables and prompts.")
    Env = "GEMINI_API_KEY"
    prompt_path = 'prompts/prompt.json'
    api_key = load_api(Env)
    prompts = load_prompt(prompt_path)

    logger.info("API key and prompts loaded successfully.")
    logger.info("Starting code generation for each branch.")

    success_count = 0
    for branch, prompt_text in prompts.items():
        try:
            logger.info(f"Processing branch: {branch}")
            response = code_generation(api_key, prompt_text)
            if save_generated_output(response, branch):
                success_count += 1
            logger.info(f"Completed processing for branch: {branch}")
        except Exception as e:
            logger.error(f"Failed to process {branch}: {str(e)}")
            continue
    
    logger.info(f"Successfully processed {success_count}/{len(prompts)} branches")


if __name__ == "__main__":
    main()

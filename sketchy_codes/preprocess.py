import os
import json
import uuid
import pandas as pd
import openai
from tqdm import tqdm
# Set your OpenAI API key here or via an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly: openai.api_key = "your-key-here"
def generate_qa_pair(report: str) -> (str, str):
    """
    Uses GPT to generate a question and answer pair from a given medical report.
    The answer must be a contiguous substring of the report.
    """
    prompt = (
        "Generate a question and answer pair based on the following medical report. "
        "Ensure the answer is a contiguous substring from the report. "
        "Enforced output format:\n"
        "Question: <your question>\n"
        "Answer: <your answer>\n\n"
        f"Medical Report: {report}\n\n"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or another model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return "", ""
    
    output_text = response['choices'][0]['message']['content']
    
    # Extract question and answer
    question, answer = "", ""
    for line in output_text.splitlines():
        if line.lower().startswith("question:"):
            question = line.split(":", 1)[1].strip()
        elif line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
    
    if not question or not answer:
        print("Warning: Could not parse QA pair correctly. Output was:")
        print(output_text)
    
    return question, answer

def extract_relevant_context(report: str, answer: str, window_size: int = 1000):
    """
    Extracts a smaller context from the report that contains the answer.
    Ensures the extracted text is at most `window_size` characters long.
    """
    answer_start = report.lower().find(answer.lower())
    
    if answer_start == -1:
        print("Warning: Answer not found in report. Using the full report.")
        return report[150:window_size+150], 0  # Default to the first `window_size` characters
    
    # Define boundaries around the answer
    start_idx = max(0, answer_start - window_size // 2)
    end_idx = min(len(report), answer_start + len(answer) + window_size // 2)

    # Extract relevant context
    relevant_context = report[start_idx:end_idx]

    # Adjust answer_start within the new context
    new_answer_start = relevant_context.find(answer)

    return relevant_context, new_answer_start

def convert_mimic_to_squad(mimic_csv_path: str, output_dir: str):
    """
    Reads a CSV file containing MIMIC notes (with a 'report' column),
    generates Squad-style formatted JSON with question and answer pairs,
    and saves it to the specified output directory.
    """
    # Load the MIMIC dataset (assumes CSV with a 'report' column)
    df = pd.read_csv(mimic_csv_path)
    
    # Initialize list for storing Squad-formatted data
    squad_data = []

    # Process each report in the dataset
    for idx, row in tqdm(df.iterrows()):
        # if idx>20: break
        report = row["text"]
        
        # Generate a question and answer pair using GPT
        question, answer = generate_qa_pair(report)
        
        # Extract relevant paragraph-sized context
        context, new_answer_start = extract_relevant_context(report, answer)
        
        # Create Squad entry
        squad_entry = {
            "id": str(uuid.uuid4()),
            "title": f"MIMIC-{idx}",
            "context": context,
            "question": question,
            "answers": [{"text": answer, "answer_start": new_answer_start}]
        }
        
        squad_data.append(squad_entry)
    
        # Ensure output directory exists and save the JSON file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "mimic_squad.json")
        
        with open(output_file, "w") as f:
            json.dump(squad_data, f, indent=4)
    
    print(f"Saved Squad formatted dataset to {output_file}")

if __name__ == "__main__":
    mimic_csv_path = "/home/obiwan/Downloads/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv"
    output_dir = "/home/obiwan/repos/mimic_fl/formatted_mimic_notes"
    convert_mimic_to_squad(mimic_csv_path, output_dir)

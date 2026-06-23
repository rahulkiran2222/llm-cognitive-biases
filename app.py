import os
import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Supported Models
MODELS = {
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct",
    "Llama 3.1 8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral 7B v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma 2 9B": "google/gemma-2-9b-it"
}

# 2. Experimental Stimuli
EXPERIMENTS = {
    "Anchoring (High)": "Question: Was Mahatma Gandhi older or younger than 144 when he died? How old was he? Answer with ONLY the number.",
    "Anchoring (Low)": "Question: Was Mahatma Gandhi older or younger than 9 when he died? How old was he? Answer with ONLY the number.",
    "Framing (Gain)": "600 people are at risk. Program A: 200 people will be saved. Program B: 1/3 probability 600 saved. Which do you choose? Answer A or B.",
    "Framing (Loss)": "600 people are at risk. Program C: 400 people will die. Program D: 1/3 probability 0 die. Which do you choose? Answer C or D.",
    "Conjunction Fallacy": "Linda is 31, single, outspoken, and very bright. She majored in philosophy. Which is more probable? A) Linda is a bank teller. B) Linda is a bank teller and active in the feminist movement.",
    "Base-Rate Neglect": "70 lawyers, 30 engineers. Jack is conservative and likes math puzzles. What is the probability (0-100) that Jack is an engineer? Answer only the number.",
    "Availability Heuristic": "Are there more words that: A) Begin with 'K' B) Have 'K' as the 3rd letter? Answer A or B."
}

def extract_value(text):
    """Extracts either the first number or the choice A/B from the response."""
    numbers = re.findall(r'\d+', text)
    if numbers:
        return float(numbers[0])
    choices = re.findall(r'\b[A-D]\b', text.upper())
    if choices:
        return choices[0]
    return text[:10] # Fallback to start of string

def create_plot(df, bias_type):
    """Generates a visualization based on the type of data returned."""
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    
    # Check if data is numerical (Anchoring/Base-Rate) or categorical (Framing/Conjunction)
    sample_val = df["Value"].iloc[0]
    
    if isinstance(sample_val, (int, float)):
        # Numerical Plot (Distribution)
        sns.histplot(data=df, x="Value", kde=True, color="purple", bins=5)
        # Add Ground Truth for Anchoring if applicable
        if "Anchoring" in bias_type:
            plt.axvline(78, color='red', linestyle='--', label='Actual Age (78)')
            plt.legend()
        plt.title(f"Numerical Distribution: {bias_type}")
    else:
        # Categorical Plot (Counts)
        sns.countplot(data=df, x="Value", palette="viridis")
        plt.title(f"Response Frequency: {bias_type}")
    
    plt.tight_layout()
    return plt.gcf()

def run_bias_test(model_name, bias_type, iterations):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return None, None, "❌ ERROR: HF_TOKEN secret missing in Settings."

    try:
        client = InferenceClient(model=MODELS[model_name], token=hf_token)
        raw_results = []
        extracted_values = []
        
        for i in range(int(iterations)):
            messages = [{"role": "user", "content": EXPERIMENTS[bias_type]}]
            response = client.chat_completion(
                messages=messages,
                max_tokens=20,
                temperature=0.7 + (i * 0.05)
            )
            answer = response.choices[0].message.content.strip()
            raw_results.append(answer)
            extracted_values.append(extract_value(answer))
        
        df = pd.DataFrame({
            "Iteration": list(range(1, int(iterations) + 1)),
            "Raw Response": raw_results,
            "Value": extracted_values
        })
        
        # Generate the chart
        fig = create_plot(df, bias_type)
        
        return df, fig, f"✅ Success! Analyzed {iterations} samples from {model_name}."
        
    except Exception as e:
        return None, None, f"❌ API ERROR: {str(e)}"

# 3. Enhanced UI Design
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 🧠 LLM Cognitive Bias Lab")
    gr.Markdown("Visualize how Large Language Models replicate human systematic errors.")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_input = gr.Dropdown(choices=list(MODELS.keys()), label="Model", value="Qwen 2.5 7B")
            bias_input = gr.Dropdown(choices=list(EXPERIMENTS.keys()), label="Bias Paradigm", value="Anchoring (High)")
            iter_input = gr.Slider(minimum=2, maximum=15, step=1, label="Iterations", value=5)
            run_btn = gr.Button("🚀 Run Experiment", variant="primary")
            status_out = gr.Textbox(label="Status")

        with gr.Column(scale=2):
            plot_out = gr.Plot(label="Statistical Visualization")
            data_out = gr.Dataframe(label="Raw Data Table")

    run_btn.click(
        fn=run_bias_test, 
        inputs=[model_input, bias_input, iter_input], 
        outputs=[data_out, plot_out, status_out]
    )

if __name__ == "__main__":
    demo.launch()

import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models to test
MODELS = {
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct",
    "Llama 3.1 8B": "meta-llama/Llama-3.1-8B-Instruct"
}

# The 5 Paradigms
EXPERIMENTS = {
    "Anchoring": "Was Gandhi older than 144 when he died? How old was he? Give only the number.",
    "Framing": "600 people are at risk. Program A: 200 saved. Program B: 1/3 chance 600 saved. Which do you choose, A or B?",
    "Conjunction": "Linda is 31, bright, and a social activist. Which is more probable? A) Bank teller B) Bank teller and feminist.",
    "Base-Rate": "In a group of 70 lawyers and 30 engineers, Jack likes math and puzzles. Is Jack more likely a lawyer or engineer?",
    "Availability": "Which is more common in English: A) Words starting with 'K' B) Words with 'K' as the 3rd letter?"
}

def run_bias_test(model_name, bias_type, iterations):
    client = InferenceClient(model_id=MODELS[model_name])
    results = []
    
    for _ in range(int(iterations)):
        response = client.text_generation(
            EXPERIMENTS[bias_type], 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.8
        )
        results.append(response)
    
    # Simple Visualization logic
    df = pd.DataFrame(results, columns=["Raw Response"])
    return df, f"Generated {iterations} samples for {bias_type} using {model_name}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 LLM Cognitive Bias Lab")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=list(MODELS.keys()), label="Select Model", value="Qwen 2.5 7B")
        bias_dropdown = gr.Dropdown(choices=list(EXPERIMENTS.keys()), label="Select Bias Paradigm")
        iters = gr.Slider(minimum=1, maximum=20, step=1, label="Iterations", value=5)
    
    run_btn = gr.Button("Run Experiment")
    
    output_text = gr.Textbox(label="Status")
    output_df = gr.Dataframe(label="Model Responses")

    run_btn.click(run_bias_test, inputs=[model_dropdown, bias_dropdown, iters], outputs=[output_df, output_text])

demo.launch()

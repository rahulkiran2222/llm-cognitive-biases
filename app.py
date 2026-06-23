import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd
import os

# 1. Models - Make sure names are exact
MODELS = {
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct",
    "Llama 3.1 8B": "meta-llama/Llama-3.1-8B-Instruct"
}

# 2. Experiments
EXPERIMENTS = {
    "Anchoring": "Question: Was Gandhi older than 144 when he died? How old was he? Answer with only the number.",
    "Framing": "600 people are at risk. Program A: 200 saved. Program B: 1/3 chance 600 saved. Choose A or B.",
    "Conjunction": "Linda is 31, bright, and an activist. Which is more probable? A) Bank teller B) Bank teller and feminist.",
    "Base-Rate": "70 lawyers, 30 engineers. Jack likes math. Is Jack a lawyer or engineer?",
    "Availability": "Which is more common: A) Words starting with 'K' B) Words with 'K' as 3rd letter?"
}

def run_bias_test(model_name, bias_type, iterations):
    # GET TOKEN FROM SPACE SECRETS
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        return None, "❌ Error: HF_TOKEN not found in Space Settings."

    try:
        client = InferenceClient(model=MODELS[model_name], token=hf_token)
        results = []
        
        for i in range(int(iterations)):
            # We add a slight change to each prompt to avoid 'cached' results
            response = client.text_generation(
                EXPERIMENTS[bias_type], 
                max_new_tokens=20, 
                do_sample=True, 
                temperature=0.8 + (i * 0.01) 
            )
            results.append(response.strip())
        
        df = pd.DataFrame(results, columns=["Model Response"])
        return df, f"✅ Successfully generated {iterations} samples."
        
    except Exception as e:
        return None, f"❌ API Error: {str(e)}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 LLM Cognitive Bias Lab")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=list(MODELS.keys()), label="Select Model", value="Qwen 2.5 7B")
        bias_dropdown = gr.Dropdown(choices=list(EXPERIMENTS.keys()), label="Select Bias Paradigm", value="Anchoring")
        iters = gr.Slider(minimum=1, maximum=10, step=1, label="Iterations (Keep low for free tier)", value=3)
    
    run_btn = gr.Button("Run Experiment", variant="primary")
    
    status_text = gr.Textbox(label="Status")
    output_df = gr.Dataframe(label="Model Responses")

    run_btn.click(run_bias_test, inputs=[model_dropdown, bias_dropdown, iters], outputs=[output_df, status_text])

demo.launch()    return df, f"Generated {iterations} samples for {bias_type} using {model_name}"

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

import os
import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd

# 1. Supported Models (Instruct versions for Conversational API)
MODELS = {
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct",
    "Llama 3.1 8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral 7B v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma 2 9B": "google/gemma-2-9b-it"
}

# 2. The 5 Canonical Paradigms (Prompts)
EXPERIMENTS = {
    "Anchoring (High)": "Question: Was Mahatma Gandhi older or younger than 144 when he died? How old was he? Answer with ONLY the number.",
    "Anchoring (Low)": "Question: Was Mahatma Gandhi older or younger than 9 when he died? How old was he? Answer with ONLY the number.",
    "Framing (Gain)": "600 people are at risk. Program A: 200 people will be saved. Program B: 1/3 probability 600 saved, 2/3 probability 0 saved. Which do you choose? Answer A or B.",
    "Framing (Loss)": "600 people are at risk. Program C: 400 people will die. Program D: 1/3 probability 0 die, 2/3 probability 600 die. Which do you choose? Answer C or D.",
    "Conjunction Fallacy": "Linda is 31, single, outspoken, and very bright. She majored in philosophy and is concerned with social justice. Which is more probable? A) Linda is a bank teller. B) Linda is a bank teller and active in the feminist movement.",
    "Base-Rate Neglect": "A panel of psychologists interviewed 100 people: 70 lawyers and 30 engineers. Jack is a man who likes math puzzles and carpentry. What is the probability (0-100) that Jack is an engineer?",
    "Availability Heuristic": "In a typical English text, are there more words that begin with the letter 'K', or more words that have 'K' as their third letter? Answer A or B."
}

def run_bias_test(model_name, bias_type, iterations):
    # Retrieve the secret token from the environment
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        return None, "❌ ERROR: HF_TOKEN not found. Go to Settings > Variables and secrets > New secret and add 'HF_TOKEN'."

    try:
        # Initialize the Client
        client = InferenceClient(model=MODELS[model_name], token=hf_token)
        results = []
        prompt_content = EXPERIMENTS[bias_type]
        
        for i in range(int(iterations)):
            # Use chat_completion to satisfy the 'conversational' task requirement
            messages = [{"role": "user", "content": prompt_content}]
            
            response = client.chat_completion(
                messages=messages,
                max_tokens=30,
                stream=False,
                temperature=0.8 + (i * 0.02) # Add variety to responses
            )
            
            # Extract text from the response object
            answer = response.choices[0].message.content
            results.append(answer.strip())
        
        # Create DataFrame for display
        df = pd.DataFrame({
            "Iteration": list(range(1, int(iterations) + 1)),
            "Model Response": results
        })
        
        return df, f"✅ Success! Generated {iterations} responses from {model_name}."
        
    except Exception as e:
        # Check for specific Llama 3.1 Gating error
        if "gated" in str(e).lower():
            return None, "❌ ERROR: You need to request access to Llama 3.1 on Hugging Face first."
        return None, f"❌ API ERROR: {str(e)}"

# 3. Gradio Interface Design
with gr.Blocks(theme=gr.themes.Default(primary_hue="purple")) as demo:
    gr.Markdown("# 🧠 LLM Cognitive Bias Laboratory")
    gr.Markdown("Replicating classical human psychological biases in foundation models using the Inference API.")
    
    with gr.Row():
        with gr.Column():
            model_input = gr.Dropdown(
                choices=list(MODELS.keys()), 
                label="Select Large Language Model", 
                value="Qwen 2.5 7B"
            )
            bias_input = gr.Dropdown(
                choices=list(EXPERIMENTS.keys()), 
                label="Select Bias Paradigm", 
                value="Anchoring (High)"
            )
            iter_input = gr.Slider(
                minimum=1, 
                maximum=10, 
                step=1, 
                label="Number of Trials (Iterations)", 
                value=3
            )
            run_btn = gr.Button("🚀 Run Experiment", variant="primary")

    with gr.Row():
        with gr.Column():
            status_out = gr.Textbox(label="Status / Error Log")
            data_out = gr.Dataframe(label="Collected Model Data")

    # Connect the button to the function
    run_btn.click(
        fn=run_bias_test, 
        inputs=[model_input, bias_input, iter_input], 
        outputs=[data_out, status_out]
    )

    gr.Markdown("---")
    gr.Markdown("### How to use this for your portfolio:")
    gr.Markdown("1. Select **Anchoring (High)** vs **Anchoring (Low)** and observe the mean difference.\n2. Note how the model often fails the **Conjunction Fallacy** (Linda Problem).\n3. Use these results to prove that LLMs replicate human-like 'systematic irrationality'.")

# 4. Launch the app
if __name__ == "__main__":
    demo.launch()

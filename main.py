import gradio as gr
import matplotlib.pyplot as plt
from utils.utils import generate_words, plot_top_k_network_arabic,plot_top_k_network_interactive
import pickle


def get_stats_html():
    """
    Converts the stats dictionary into a nice HTML/CSS format.
    """
    with open("./results/stats.pkl", 'rb') as f:
        stats = pickle.load(f)
        print(stats)
    
    html_content = "<div class='stats-container'>"

    transform={
        "stop_word":"Stop Words",
        "Stem":"stemming",
        "lemma":"Lemmatisation",
        "in_Dic":"Dictionary",
        "POS":"POS",
        "rejected":"Rejected",'ner':"NER",
        'words_count':"All Words"
    
    }
    for key, value in stats.items():
        if key =='POS':
            value= sum(value.values())
        html_content += f"""
        <div class='stat-box'>
            <div class='stat-number'>{value}</div>
            <div class='stat-label'>{transform.get(key)}</div>
        </div>
        """
    html_content += "</div>"
    return html_content

def gradio_plot_network(target_word, k):
    """Wrapper function to handle Gradio inputs for the network plot"""
    k = int(k)
    fig = plot_top_k_network_interactive(target_word, k, db_filename="./results/co-occurrence.db")
    
    if fig is None:

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"The word '{target_word}' was not found\nor has no following words.", 
                ha='center', va='center', fontsize=14, fontname='Tahoma')
        ax.axis("off")
        
    return fig

def gradio_generate_text(length):
    """Wrapper function for text generation"""
    length = int(length)
    text = generate_words(length)
    return text






custom_css = """
body { font-family: 'Tahoma', sans-serif; }
.stats-container { display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 20px; }
.stat-box { 
    background: linear-gradient(135deg, #6dd5ed, #2193b0); 
    padding: 20px; 
    border-radius: 15px; 
    text-align: center; 
    margin: 10px; 
    width: 30%; 
    box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
    color: white;
    transition: transform 0.3s;
}
.stat-box:hover { transform: translateY(-5px); }
.stat-number { font-size: 2.5em; font-weight: bold; margin-bottom: 10px; }
.stat-label { font-size: 1.2em; opacity: 0.9; }
"""


with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
    

    gr.HTML("<h1 style='text-align: center; color: #2c3e50; margin-bottom: 5px;'>🚀 Arabic Text Analysis and Generation System</h1>")
    gr.HTML("<p style='text-align: center; color: #7f8c8d;'>Built using Co-occurrence Analysis and Databases</p>")
    

    with gr.Tabs():
        

        with gr.TabItem("📊 Data Statistics"):
            gr.HTML(get_stats_html())
            
            gr.Markdown("<br><h3 style='text-align: center;'>Sentence Length Distribution</h3>")
            gr.Image(value="./results/distribution_sentences.png", label="Sentence Length Distribution", interactive=False)
            

        with gr.TabItem("🕸️ Word Network"):
            with gr.Row():
                word_input = gr.Textbox(label="Target Word", placeholder="Example: في, من, قال...")
                k_input = gr.Number(value=5, label="Top K Next Words", precision=0)
            
            net_btn = gr.Button("Generate Graph", variant="primary")
            plot_output = gr.HTML(label="Interactive Network Graph")
            

            net_btn.click(fn=gradio_plot_network, inputs=[word_input, k_input], outputs=plot_output)
            

        with gr.TabItem("✍️ Text Generation"):
            with gr.Row():
                len_input = gr.Number(value=15, label="Sequence Length", precision=0)
            
            gen_btn = gr.Button("Generate Random Text", variant="primary")
            

            text_output = gr.Textbox(label="Generated Text", lines=6, text_align="right")
            
            
            gen_btn.click(fn=gradio_generate_text, inputs=[len_input], outputs=text_output)




if __name__ == "__main__":
    app.launch(debug=True)
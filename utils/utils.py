import sqlite3
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import arabic_reshaper
from bidi.algorithm import get_display
import pickle
from pyvis.network import Network

try:
    with open("./results/vocab.pkl", 'rb') as f:
        
        vocab = list(pickle.load(f)) 
        
    with open("./results/filtered_end_words.pkl", 'rb') as f:
        end_words = set(pickle.load(f)) 
except Exception as e:
    print(f"Error loading files: {e}")


def check_word_in_db(word, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    
    cursor.execute("SELECT 1 FROM dictionary WHERE word = ?", (word,))

    
    result = cursor.fetchone()
    conn.close()

    return result is not None



def create_inner_dict():
    return defaultdict(int)

def build_co_occurrence(tokens, window_size=1):
    
    co_occurrence = defaultdict(create_inner_dict)
    
  
    for i in range(len(tokens)):
        current_word = tokens[i]
        

        limit = min(i + 1 + window_size, len(tokens))
        
        for j in range(i + 1, limit):
            next_word = tokens[j]
            co_occurrence[current_word][next_word] += 1
    return co_occurrence

def fix_arabic(text):
    """
    Reshapes Arabic letters to connect properly and fixes RTL direction.
    """
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def get_top_k_words_from_db(target_word, k=5, db_filename="co_occurrence.db"):
    
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT word2, freq 
        FROM occurence 
        WHERE word1 = ? 
        ORDER BY freq DESC 
        LIMIT ?
    ''', (target_word, k))
    
    results = cursor.fetchall()
    conn.close()
    return results


def store_co_occurrence_in_db(co_occurrence, db_filename="co_occurrence.db"):
    """
    Stores a nested co-occurrence dictionary into an SQLite database.
    """
    
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS occurence (
            word1 TEXT,
            word2 TEXT,
            freq INTEGER
        )
    ''')
    
    
    cursor.execute('DELETE FROM occurence')
    
    
    data_to_insert = []
    for word1, word2_dict in co_occurrence.items():
        for word2, freq in word2_dict.items():
            data_to_insert.append((word1, word2, freq))
            
    
    cursor.executemany('''
        INSERT INTO occurence (word1, word2, freq)
        VALUES (?, ?, ?)
    ''', data_to_insert)
    
    
    conn.commit()
    conn.close()
    
    print(f"Successfully inserted {len(data_to_insert)} rows into '{db_filename}'.")



def plot_top_k_network_arabic(target_word, k=5, db_filename="co_occurrence.db"):
    top_k_results = get_top_k_words_from_db(target_word, k, db_filename)
    
    if not top_k_results:
        print(f"No following words found for '{target_word}'.")
        return

    
    plt.rcParams['font.family'] = 'Tahoma' 

    G = nx.DiGraph()
    G.add_node(target_word)
    
    for next_word, freq in top_k_results:
        G.add_edge(target_word, next_word, weight=freq)

    fig=plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1.0, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="#87CEFA", alpha=0.9)
    

    labels = {node: fix_arabic(node) for node in G.nodes()}
    
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_weight="bold")
    
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, arrowsize=25, edge_color="gray")
    
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=11)
    
    
    fixed_target = fix_arabic(target_word)
    
    plt.title(f"Top {len(top_k_results)} Next Words for: {fixed_target}", fontsize=16)
    
    plt.axis("off")
    plt.tight_layout()
    #plt.savefig("./s.png")
    return fig


def remove_stop_words_from_end_of_words():
    with open("./results/end_words.pkl",'rb') as f :
        end_words = pickle.load(f)
    
    with open("./results/arabic_stopwords.pkl",'rb') as f :
        stop_words = set(pickle.load(f))

    ends=end_words- stop_words
    
            
    
    try:
        with open("./results/filtered_end_words.pkl", 'wb') as f:
            pickle.dump(ends, f)
        
    except Exception as e:
        pass





def generate_words(length=10):
    
    current_word = random.choice(vocab)
    text = current_word
    next_words_length=10
   
    while current_word not in end_words and length > 0:
        
        
        next_words = get_top_k_words_from_db(current_word, next_words_length, "./results/co-occurrence.db")
        
        
        if not next_words:
            
            break 
            
        next_word = random.choice(next_words)[0]
        text += f" {next_word}"
        current_word = next_word
        
        length -= 1
        
    return text
    


def plot_top_k_network_interactive(target_word, k=5, db_filename="./results/co-occurrence.db"):
    top_k_results = get_top_k_words_from_db(target_word, k, db_filename)
    
    if not top_k_results:
        
        return f"<div style='text-align:center; padding:50px;'>No following words found for '{target_word}'.</div>"

    
    G = nx.DiGraph()
    
    
    G.add_node(target_word, label=target_word, title="Target Word", color="#ff9999", size=30)
    
    for next_word, freq in top_k_results:
        G.add_node(next_word, label=next_word, title=f"Frequency: {freq}", color="#99ccff", size=20)
       
        G.add_edge(target_word, next_word, value=freq, title=f"Freq: {freq}", label=str(freq))

    
    net = Network(directed=True, height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    
    
    net.repulsion(node_distance=150, spring_length=200)
    
    
    
    html_path = "./results/interactive_network.html"
    net.write_html(html_path)
    
    
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    # We use srcdoc to safely embed the HTML inside the Gradio UI
    iframe = f"""<iframe style="width: 100%; height: 520px; border: none; border-radius: 10px;" srcdoc='{html_content.replace("'", "&apos;")}'></iframe>"""
    
    return iframe




import sqlite3
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

import arabic_reshaper
from bidi.algorithm import get_display


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
    # 1. Connect to SQLite (this creates the file if it doesn't exist)
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # 2. Create the table with the specified schema
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

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1.0, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="#87CEFA", alpha=0.9)
    

    labels = {node: fix_arabic(node) for node in G.nodes()}
    
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_weight="bold")
    
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, arrowsize=25, edge_color="gray")
    
    edge_labels = {(u, v): f"Freq: {d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=11)
    
    
    fixed_target = fix_arabic(target_word)
    
    plt.title(f"Top {len(top_k_results)} Next Words for: {fixed_target}", fontsize=16)
    
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# plot_top_k_network_arabic(target_word="في", k=5, db_filename="co_occurrence.db")
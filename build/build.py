import string
from tashaphyne.stemming import arabicstopwords, ArabicLightStemmer
import stanza
import os
import re
from rich.console import Console, Group
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn,TimeElapsedColumn
from rich.panel import Panel
from rich.live import Live
import pickle
from utils.utils import check_word_in_db

ArListem = ArabicLightStemmer()


nlp = stanza.Pipeline('ar', processors='tokenize,lemma,ner,pos',tokenize_no_ssplit=True)


ARABIC_PUNCTUATION = set(string.punctuation + '،؛؟«»-ـ')
arabic_stopwords=pickle.load(open('/results/arabic_stopwords.pkl','rb'))

def build_vocab_console(file_path):
    console = Console()
    
    ponctuation_tokens = 0
    all_tokens = 0
    sentences_lengths=[]
    stats = {
        "stop_word": 0, "ner": 0, "in_Dic": 0,
        "Stem": 0, "lemma": 0, "rejected": 0,
        "POS": {"NOUN": 0, "VERB": 0, "ADJ": 0}
    }

    vocab = set()
    tokens = []
    delimiters = r"[.،؛,;]"
    end_words = set()
    
    
    all_files = os.listdir(file_path)
    
    
    all_files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else 0)
    
    
    files_to_process = all_files[:1000]
    
    
    progress = Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn()
    )
    task = progress.add_task("[cyan]Initializing...", total=len(files_to_process))

    
    stats_panel = Panel("Loading stats...", title="[bold yellow]Vocabulary Stats[/bold yellow]", border_style="yellow")

    
    ui_group = Group(stats_panel, progress)

    
    with Live(ui_group, console=console, refresh_per_second=10):
        
        for file in files_to_process:
            with open(os.path.join(file_path, file), 'r', encoding='utf-8') as f:
                text = f.read()
                raw_sentences = [s.strip() for s in re.split(delimiters, text) if s.strip()]
         
                sentences_lengths.extend(list(map(lambda x :len(x),raw_sentences)) )
                docs = [nlp(sent) for sent in raw_sentences]
                
                for doc in docs:
                    ner_ = [ent.text for ent in doc.entities]
                    all_tokens += doc.num_words
                    words = doc.iter_words()
                    
                    last_valid_word = None 

                    for word in words:
                        text_word = word.text
                        if text_word in vocab:
                            continue
                            
                        is_valid_vocab = False
                        if text_word in ARABIC_PUNCTUATION:
                            ponctuation_tokens += 1
                            continue

                        elif text_word in arabic_stopwords:
                            stats["stop_word"] += 1
                            continue
                            
                        elif text_word in ner_:
                            stats["ner"] += 1
                            continue

                        elif check_word_in_db(text_word, '/content/drive/MyDrive/NLP/TP/dictionary.db'):
                            stats["in_Dic"] += 1
                            is_valid_vocab = True

                        elif check_word_in_db(ArListem.light_stem(text_word), '/content/drive/MyDrive/NLP/TP/dictionary.db'):
                            stats["Stem"] += 1
                            is_valid_vocab = True

                        elif check_word_in_db(word.lemma, '/content/drive/MyDrive/NLP/TP/dictionary.db'):
                            stats["lemma"] += 1
                            is_valid_vocab = True

                        else:
                            stats["rejected"] += 1
                            continue
                        
                        if is_valid_vocab:
                            vocab.add(text_word)
                            tokens.append(text_word)
                            
                            last_valid_word = text_word
                            
                            if word.upos in ['NOUN', 'VERB', 'ADJ']:
                                stats["POS"][word.upos] += 1

                    if last_valid_word is not None:
                        end_words.add(last_valid_word)
            
            
            total_words = all_tokens - ponctuation_tokens
            dashboard_text = (
                f"[bright_white]Total Words:[/bright_white] {total_words:,}  |  [bright_yellow]Vocab Size:[/bright_yellow] {len(vocab):,} | "
                f"[bright_green]Dict:[/bright_green] {stats['in_Dic']:,}  |  [bright_green]Stem:[/bright_green] {stats['Stem']:,}  |  [bright_green]Lemma:[/bright_green] {stats['lemma']:,}\n"
                f"[bright_red]Rejected:[/bright_red] {stats['rejected']:,}  |  [bright_red]Stop:[/bright_red] {stats['stop_word']:,}  |  [bright_red]NER:[/bright_red] {stats['ner']:,}"
            )
            
            
            stats_panel.renderable = dashboard_text
            
            
            progress.update(task, advance=1, description=f"Processing: [cyan]{file}[/cyan]")

            
            if len(vocab) > 50000:
                
                console.print("\n[bold red]⚠️ Vocabulary limit of 50,000 reached. Stopping early![/bold red]")
                break

    console.print("\n[bold green]✅ Vocabulary build complete![/bold green]")
    
    words_count = all_tokens - ponctuation_tokens
    stats['words_count'] = words_count
    
    return vocab, end_words, tokens, stats,sentences_lengths

vocab, end_words, tokens, stats,sentences_lengths = build_vocab_console("./data/Culture")


pickle.dump(vocab,open("./results/vocab.pkl"))
pickle.dump(end_words,open("./results/end_words.pkl"))
pickle.dump(tokens,open("./results/tokens.pkl"))
pickle.dump(stats,open("./results/stats.pkl"))
pickle.dump(sentences_lengths,open("./results/sentences_lengths.pkl"))
import pickle 
with open("./results/stats.pkl", 'rb') as f:
        stats = pickle.load(f)
        for k in ['stop_word','ner','in_Dic','Stem',"lemma",'rejected']:
            stats[k]=round(stats[k]/stats["words_count"]*100,3)


        print(sum([stats["POS"][i] for i in stats["POS"]])/stats['words_count'])
        print(stats)
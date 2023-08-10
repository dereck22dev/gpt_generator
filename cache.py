from transformers import GPT2LMHeadModel, GPT2Tokenizer

def cache(model, tokenizer):
    
    # Sauvegarde du modèle et du tokenizer
    model.save_pretrained("./gpt2_medium_model")
    tokenizer.save_pretrained("./gpt2_medium_model")
    
    
def getFromCache():
    
    # Chargement à partir des fichiers sauvegardés
    model = GPT2LMHeadModel.from_pretrained("./gpt2_medium_model")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_medium_model")
    
    data = {
        'model' : model,
        'tokenizer': tokenizer
    }

    return data
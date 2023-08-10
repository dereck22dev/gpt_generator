from transformers import GPT2LMHeadModel, GPT2Tokenizer
import settings

model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_article(prompt, max_length=settings.MAX_LENGTH):
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Génération du texte
    output = model.generate(
        input_ids, 
        max_length=max_length,
        no_repeat_ngram_size=settings.NO_REPEAT_NGRAM_SIZE,
        temperature=settings.TEMPERATURE, 
        top_k=settings.TOP_K, 
        top_p=settings.TOP_P,
        pad_token_id=tokenizer.eos_token_id
    )

    # Décodage et affichage du texte généré
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

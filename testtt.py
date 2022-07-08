# from sentence_transformers import SentenceTransformer, InputExample, losses
# from torch.utils.data import DataLoader
#
# #Define the model. Either from scratch of by loading a pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')
# emb1 = model.encode('How big is London')
# emb2 = model.encode('How big is London')
# #Define your train examples. You need more than just two examples...
# train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#     InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
#
# #Define your train dataset, the dataloader and the train loss
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
# train_loss = losses.CosineSimilarityLoss(model)
#
# #Tune the model
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)
# emb3 = model.encode('How big is London')
#
# print()

from transformers import GPT2Tokenizer, GPT2Model, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
orig_num_tokens = len(tokenizer)
num_added_toks = tokenizer.add_tokens(['FALSE', 'TRUE'], special_tokens=True)  ##This line is updated

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.transformer.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_toks)
prompt = "Today I believe we can finally TRUE "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# generate up to 30 tokens
outputs = model.generate(input_ids, max_length=30,output_scores=True, return_dict_in_generate=True)
res = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
print(res)

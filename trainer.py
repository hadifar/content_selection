import torch
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        device = inputs.device if torch.cuda.is_available() else torch.device('cpu')
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2.], device=device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

import torch
import torch.nn.functional as F
import random
import copy

def selective_log_softmax(logits, input_ids):
    """
    Compute the log probabilities for the tokens specified in input_ids using a selective log-softmax.

    Args:
        logits (torch.Tensor): A tensor of shape (batch_size, seq_len, vocab_size) containing raw logits from the model.
        input_ids (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the token indices for which we want the log probabilities.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, seq_len) where each element is the log probability 
                      corresponding to the token in input_ids at that position.
    
    Explanation:
        1. F.log_softmax is applied along the vocabulary dimension (dim=-1) to convert logits into log probabilities.
        2. The tensor input_ids is reshaped (via unsqueeze) to have an extra dimension so that we can use it as indices 
           in the log_probs tensor.
        3. torch.gather collects the log probability at the index specified in input_ids for each position.
        4. Finally, squeeze(-1) removes the extra dimension, returning a tensor with the same shape as input_ids.
    """
    # Convert raw logits into log probabilities along the vocabulary axis.
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Reshape input_ids from (batch_size, seq_len) to (batch_size, seq_len, 1) for gathering.
    # Then, gather the log probability for each token in input_ids.
    selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))
    
    # Remove the extra last dimension to get back to shape (batch_size, seq_len).
    return selected_log_probs.squeeze(-1)

def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    """
    Compute per-token log probabilities for a subset of tokens (typically the completion tokens).

    Args:
        model: The language model to use.
        input_ids (torch.Tensor): Tensor of shape (batch_size, total_seq_len) containing token ids 
                                  for both prompt and completion.
        attention_mask (torch.Tensor): Tensor of shape (batch_size, total_seq_len) indicating which tokens are real (1) or padding (0).
        logits_to_keep (int): Number of tokens (from the completion part) for which we need log probabilities.

    Returns:
        torch.Tensor: Log probabilities for the last `logits_to_keep` tokens of each sequence.
    
    Explanation:
        1. We call the model with logits_to_keep + 1 so that the model outputs one extra logit than needed.
           This is common in next-token prediction setups.
        2. We slice off the last logit along the sequence dimension because it does not correspond to any input token.
        3. We then restrict both the input_ids and logits to the last logits_to_keep tokens, which should 
           correspond to the generated completion portion.
        4. Finally, we use the selective_log_softmax to compute log probabilities only for those tokens.
    """
    # Run the model forward pass and obtain logits.
    logits = model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        logits_to_keep=logits_to_keep + 1  # Request one extra logit for proper alignment.
    ).logits  # Shape: (batch_size, total_seq_len, vocab_size)

    # Remove the last logit as it does not have a corresponding target token.
    logits = logits[:, :-1, :]  # New shape: (batch_size, total_seq_len - 1, vocab_size)
    
    # Slice the input_ids to keep only the last logits_to_keep tokens.
    # This corresponds to the generated completion tokens.
    input_ids = input_ids[:, -logits_to_keep:]  # Shape: (batch_size, logits_to_keep)
    
    # Also slice the logits to keep only those corresponding to the completion tokens.
    logits = logits[:, -logits_to_keep:, :]  # Shape: (batch_size, logits_to_keep, vocab_size)
    
    # Compute and return the log probabilities for the selected tokens.
    return selective_log_softmax(logits, input_ids)

def create_completion_mask(completion_ids, eos_token_id):
    """
    Create a binary mask for the generated completion tokens so that tokens after the first EOS are ignored.

    Args:
        completion_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) with generated token ids.
        eos_token_id (int): The token id representing the end-of-sequence.

    Returns:
        torch.Tensor: A mask tensor of shape (batch_size, seq_len) with 1s for tokens up to and including the first EOS 
                      and 0s for tokens following the first EOS.

    Explanation:
        1. First, a boolean mask (is_eos) is created indicating where in the sequence the EOS token appears.
        2. An index tensor (eos_idx) is initialized, assuming that no EOS is found (defaulting to the sequence length).
        3. For sequences where EOS exists, eos_idx is updated to the position (index) of the first EOS.
        4. A sequence index tensor is created that contains indices for each position in the sequence.
        5. The final mask is computed by comparing the sequence indices to eos_idx (after adding a dimension).
    """
    # Determine which positions in each sequence equal the EOS token.
    is_eos = completion_ids == eos_token_id  # Boolean tensor of shape (batch_size, seq_len)

    # Initialize a tensor to store the index of the first EOS for each sequence.
    # If no EOS is found, default to the full sequence length (is_eos.size(1)).
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    
    # Identify sequences that contain at least one EOS.
    mask_exists = is_eos.any(dim=1)
    # For sequences with an EOS, update eos_idx to the index of the first occurrence.
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    
    # Create a tensor of indices [0, 1, 2, ..., seq_len-1] and replicate it for each sequence in the batch.
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    
    # Build the mask: positions with an index less than or equal to the first EOS index are marked as 1.
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    
    return completion_mask

def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """
    Generate multiple completions for each prompt and create corresponding attention masks.

    Args:
        model: The language model used for generation.
        tokenizer: The tokenizer to process the prompts and decode the outputs.
        prompts (list of str): List of input prompt strings.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of new tokens to generate for the completion.

    Returns:
        tuple: Contains the following tensors:
            - prompt_ids: (batch_size * num_generations, prompt_seq_len)
            - prompt_mask: (batch_size * num_generations, prompt_seq_len)
            - completion_ids: (batch_size * num_generations, completion_seq_len)
            - completion_mask: (batch_size * num_generations, completion_seq_len)
    
    Explanation:
        1. The prompts are tokenized and padded (with padding added to the left).
        2. Each prompt is repeated num_generations times so that multiple completions are generated per prompt.
        3. The model.generate() function is called to generate new tokens.
        4. The generated output contains the prompt followed by the completion; we remove the prompt part to get the completions.
        5. A mask is created (via create_completion_mask) so that only tokens up to the first EOS are considered.
    """
    device = next(model.parameters()).device

    # Tokenize the list of prompts with padding. The padding_side="left" ensures alignment on the right.
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)      # Shape: (batch_size, prompt_seq_len)
    prompt_mask = inputs["attention_mask"].to(device)  # Shape: (batch_size, prompt_seq_len)
    prompt_length = prompt_ids.size(1)  # Save the prompt length to later separate prompt from completion.

    # Repeat each prompt num_generations times.
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)   # New shape: (batch_size*num_generations, prompt_seq_len)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0) # New shape: (batch_size*num_generations, prompt_seq_len)

    # Generate new tokens for each prompt. The output includes the original prompt and the generated tokens.
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Remove the prompt portion from the generated output to isolate the completion tokens.
    completion_ids = outputs[:, prompt_length:]  # Shape: (batch_size*num_generations, completion_seq_len)

    # Create a binary mask that ignores tokens beyond the first EOS token.
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)

    return prompt_ids, prompt_mask, completion_ids, completion_mask

def grpo_loss(model, ref_model, tokenizer, batch_samples, reward_function,
              beta=0.1, num_generations=4, max_completion_length=32):
    """
    Compute the GRPO loss, which combines a policy gradient loss with a KL divergence penalty.

    Args:
        model: The current language model (policy).
        ref_model: The reference model (baseline) used for computing KL divergence.
        tokenizer: The tokenizer for decoding completions.
        batch_samples (list): A list of samples, where each sample contains at least a "prompt" and an "answer".
        reward_function: A function that takes prompts, completions, and answers and returns a list of reward values.
        beta (float): Weight for the KL divergence term.
        num_generations (int): Number of completions generated per prompt.
        max_completion_length (int): Maximum token length for each generated completion.

    Returns:
        torch.Tensor: A scalar loss tensor.

    Explanation:
        1. Extract prompts from the batch samples.
        2. Generate multiple completions per prompt.
        3. Concatenate the prompt and completion tokens to form the full sequence.
        4. Compute the log probabilities (for the completion part) using both the current model and the reference model.
        5. Format the generated completions into text for reward evaluation.
        6. Compute a reward for each completion and then normalize them (compute advantages) per group.
        7. Compute the per-token KL divergence between the reference and current model's log probabilities.
        8. Combine the policy loss and KL divergence to compute the final loss.
    """
    device = next(model.parameters()).device

    # Extract the prompt text from each sample.
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]

    # Generate completions and obtain their masks.
    prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
        model, tokenizer, prompts, num_generations, max_completion_length
    )

    # Concatenate prompt and completion tokens to form the full input sequence.
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # Only compute log probabilities for the completion part.
    logits_to_keep = completion_ids.size(1)

    # Compute log probabilities for the completion tokens using the reference model.
    # Use torch.no_grad() because gradients should not flow through the reference model.
    with torch.no_grad():
        ref_token_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
    
    # Compute log probabilities for the completion tokens using the current model.
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)

    # Decode the completion tokens into text for reward evaluation.
    # Each decoded completion is wrapped in a dictionary (for compatibility with some reward functions).
    formatted_completions = [
        [{'content': tokenizer.decode(ids, skip_special_tokens=True)}]
        for ids in completion_ids
    ]
    # Repeat each prompt for each generated completion.
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    # Extract answers from the batch and repeat each for the corresponding number of generations.
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1]
               for sample in batch_samples for _ in range(num_generations)]

    # Compute rewards using the reward_function.
    rewards = torch.tensor(
        reward_function(prompts=repeated_prompts, completions=formatted_completions, answer=answers),
        dtype=torch.float32,
        device=device
    )

    # For monitoring, print the average reward.
    avg_reward = rewards.mean().item()
    print("Average Reward:", avg_reward)

    # Reshape rewards to group completions by prompt.
    # Compute mean and standard deviation for each prompt group.
    mean_rewards = rewards.view(-1, num_generations).mean(dim=1)
    std_rewards = rewards.view(-1, num_generations).std(dim=1)
    # Expand the means and stds to match the original flat rewards tensor shape.
    mean_rewards = mean_rewards.repeat_interleave(num_generations, dim=0)
    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
    # Normalize rewards to get advantages.
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)

    # Compute per-token KL divergence between reference and current model log probabilities.
    per_token_kl = torch.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1
    
    # Compute the policy gradient loss component.
    # The term token_log_probs.detach() prevents gradients from flowing into the baseline.
    per_token_loss = torch.exp(token_log_probs - token_log_probs.detach()) * advantages.unsqueeze(1)
    # Combine the loss with the KL penalty (weighted by beta) and take the negative.
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    
    # Apply the completion mask to compute the average loss per sequence:
    # - Multiply the loss by the mask so that only valid tokens contribute.
    # - Sum the loss per sequence and divide by the number of valid tokens.
    # - Finally, average over all sequences.
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    return loss

def train_with_grpo(model, tokenizer, train_data, num_steps=500, batch_size=4,
                    num_generations=4, max_completion_length=128, beta=0.1,
                    learning_rate=5e-6):
    """
    Fine-tune the model using the GRPO algorithm.

    This function implements a training loop that:
      1. Creates a reference model (a deep copy of the current model) whose parameters are frozen.
      2. For each training step:
           - Samples a batch of training samples.
           - Generates multiple completions per prompt.
           - Computes the GRPO loss (which combines a policy gradient term based on advantages and a KL divergence
             penalty between the current model and the reference model).
           - Performs backpropagation and updates the model parameters.
           - Updates the reference model to match the current model.
      
    Args:
        model: The language model to be fine-tuned.
        tokenizer: The tokenizer used for encoding prompts and decoding completions.
        train_data (list): List of training samples; each sample contains at least a "prompt" and an "answer".
        num_steps (int): Total number of training steps.
        batch_size (int): Number of samples to process per training step.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of tokens to generate for each completion.
        beta (float): Weight of the KL-divergence penalty term in the loss.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        The fine-tuned model.
    """
    # Determine the device (CPU or GPU) where the model's parameters reside.
    device = next(model.parameters()).device

    # Create a reference model by making a deep copy of the current model.
    # The reference model is used for computing the KL divergence and is not updated via gradients.
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False

    # Initialize the Adam optimizer with the provided learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Set the model to training mode (this enables dropout, etc.).
    model.train()

    # Counter to track the number of examples processed (for logging purposes).
    examples_processed = 0

    # Training loop: iterate for the specified number of training steps.
    for step in range(num_steps):
        # Randomly sample a batch of training samples from train_data.
        # Each sample is expected to be a dict (or tuple) containing at least "prompt" and "answer".
        batch_samples = random.sample(train_data, batch_size)

        # Compute the GRPO loss for the current batch.
        # The grpo_loss function performs several operations:
        #   - Extracts prompts from the batch.
        #   - Generates multiple completions per prompt.
        #   - Concatenates the prompt tokens with the generated tokens.
        #   - Computes per-token log probabilities for both the current model and the reference model.
        #   - Decodes the completions and computes rewards via a reward function.
        #   - Normalizes rewards within each prompt group (computing advantages).
        #   - Computes a per-token KL divergence between the current model and the reference model.
        #   - Combines the policy gradient loss and KL penalty into a final scalar loss.
        loss = grpo_loss(
            model,            # The current model (policy) being fine-tuned.
            ref_model,        # The reference model used for KL divergence computation.
            tokenizer,        # The tokenizer for encoding and decoding text.
            batch_samples,    # The current batch of training samples.
            combined_reward,  # The reward function (must be defined elsewhere) that returns a list of rewards.
            beta=beta,        # The KL divergence weight.
            num_generations=num_generations,
            max_completion_length=max_completion_length
        )

        # Backpropagation and parameter update:
        optimizer.zero_grad()           # Clear previous gradients.
        loss.backward()                 # Compute gradients via backpropagation.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Optionally clip gradients to prevent exploding gradients.
        optimizer.step()                # Update the model's parameters.

        # Update the reference model so that it matches the current model after this update.
        ref_model.load_state_dict(model.state_dict())

        # Log the loss every 5 steps to monitor training progress.
        if step % 5 == 0:
            print(f"Step {step}/{num_steps}, loss: {loss.item():.4f}")

        # Update the counter for processed examples.
        examples_processed += batch_size

        # Clear the GPU cache to help with memory management.
        torch.cuda.empty_cache()

    # Return the fine-tuned model after completing all training steps.
    return model

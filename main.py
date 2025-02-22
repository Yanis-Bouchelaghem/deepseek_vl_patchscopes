# %%
# Imports
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

from transformers import LlamaModel

# %%
# Loading the model
# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-1.3b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

deepseek_vl: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
deepseek_vl = deepseek_vl.to(torch.bfloat16).cuda().eval()
# %%
# Run the model
conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Extract all information from this image and convert them to markdown format.",
        "images": ["./images/training_pipelines.jpg"]
    },
    {
        "role": "Assistant",
        "content": ""
    }
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    #conversations=conversation,
    prompt="The most well-known luxury fashion company is:",
    images=pil_images,
    force_batchify=True
).to(deepseek_vl.device)

# run image encoder to get the image embeddings
inputs_embeds = deepseek_vl.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = deepseek_vl.language_model.generate(

    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=10,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)

# %%
# Display model architecture
deepseek_vl.language_model.model
# %%
# CACHING ACTVIATIONS
from collections import defaultdict

activation_cache = defaultdict(list)

def create_hook(name):
    """"A hook creator function that will create a hook for a given layer name"""
    def hook(module, input, output):
        # If output is a tuple, you might want output[0].
        # But typically for a decoder layer, output is the hidden states tensor.
        # (If you're not sure, you can `print(type(output))` inside the hook.)
        print(type(output[0]))
        activation_cache[name].append(output[0])
    return hook

import torch
from typing import Callable, Dict

def run_with_hooks(
    hooks_dict: Dict[str, Callable],
    language_model: LlamaModel,
    input_embeds,
    attention_mask
):
    """
    hooks_dict: 
        - Keys are strings describing submodules (e.g., "embedding", "layers.0", etc.)
        - Values are hook functions with signature (module, input, output).
    language_model:
        - Your LLM model (e.g., deepseek_vl.language_model).
    inference_fn:
        - A zero-arg function (or lambda) that runs your forward pass and returns outputs.
    
    Returns:
        The result of calling inference_fn().
    """
    handles = []

    try:
        # 1. Register each hook
        for key, hook_fn in hooks_dict.items():
            # Identify the submodule we want to hook.
            if key == "embedding":
                # hooking the embedding
                submodule = language_model.embed_tokens
            elif key.startswith("layers."):
                # hooking a specific decoder layer, e.g. layers.0, layers.1, etc.
                idx = int(key.split(".")[1])  # parse the layer index
                submodule = language_model.layers[idx]
            else:
                raise ValueError(f"Unknown hook key: {key}")

            handle = submodule.register_forward_hook(hook_fn)
            handles.append(handle)
        # 2. Run the forward pass / inference
        with torch.no_grad():
            outputs = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )

    finally:
        # 3. Remove hooks no matter what (even if an error happens)
        for h in handles:
            h.remove()

    return outputs




# %%
output = run_with_hooks({"layers.0": create_hook(f"layer_0")}, deepseek_vl.language_model.model,
               input_embeds=inputs_embeds,
               attention_mask=prepare_inputs.attention_mask)

# %%
HOOK_LAYER = 0
hook_handle = deepseek_vl.language_model.model.layers[HOOK_LAYER].register_forward_hook(create_hook(f"layer_{HOOK_LAYER}"))

# %%
with torch.no_grad():
    outputs = deepseek_vl.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        output_hidden_states=False,
        return_dict=True
    )
#%%
outputs

# %%
deepseek_vl.language_model.model.layers[0]._forward_hooks

# %%

hook_handle.remove()


# %%
activation_cache["layer_0"]
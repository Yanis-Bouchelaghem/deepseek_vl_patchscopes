# %%
# Imports
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

from transformers import LlamaModel
from PIL import Image
from jaxtyping import Float
# %%
# Loading the model
# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-1.3b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

deepseek_vl: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
deepseek_vl = deepseek_vl.to(torch.bfloat16).cuda().eval()

# %%
#with torch.no_grad():
#    outputs = deepseek_vl.language_model(
#        inputs_embeds=inputs_embeds,
#        attention_mask=prepare_inputs.attention_mask,
#        output_hidden_states=False,
#        return_dict=True
#    )
# %%
# CACHING ACTVIATIONS

import torch
from typing import Callable, Dict

def run_with_hooks(
    hooks_dict: Dict[str, Callable],
    language_model: LlamaModel,
    inputs_embeds,
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
                return_dict=False
            )
    finally:
        # 3. Remove hooks no matter what (even if an error happens)
        for h in handles:
            h.remove()

    return outputs

def run_with_cache(language_model: LlamaModel, inputs_embeds, attention_mask) -> tuple:
    """Runs an inference and returns a tuple (output, cache). The cache contains the activations for every layer."""
    # Define the cache
    activation_cache = dict()
    hooks = dict()
    # Define the hook function
    def create_hook(layer_name):
        """"A hook creator function that will create a hook for a given layer name"""
        def hook(module, input, output):
            #print(type(output[0]))
            activation_cache[layer_name] = output[0]
        return hook
    for layer_index in range(len(language_model.layers)):
        hooks[f"layers.{layer_index}"] = create_hook(f"layers.{layer_index}")
    # Run the inference
    output = run_with_hooks(hooks, language_model, inputs_embeds, attention_mask)
    return output, activation_cache


# %%

pil_images = []
pil_img = Image.open("./images/training_pipelines.jpg")
pil_img = pil_img.convert("RGB")
pil_images.append(pil_img)
prepare_inputs = vl_chat_processor(
    #conversations=conversation,
    prompt="The capital of France is called",
    images=pil_images,
    force_batchify=True
).to(deepseek_vl.device)
tokenized_input = prepare_inputs["input_ids"]
# run visual model to get the image embeddings
inputs_embeds = deepseek_vl.prepare_inputs_embeds(**prepare_inputs)
inputs_embeds.shape
# Run the language model
output, cache = run_with_cache(deepseek_vl.language_model.model, inputs_embeds, prepare_inputs.attention_mask)



# %%
cache["layers.0"].shape
        
# %%
def create_activation_patching_hook(
    activation_value: Float[torch.Tensor, "d_model"], 
    position: int,                   # which token index to patch
    batch_idx: int = 0               # which batch example to patch
):
    """
    A forward hook that replaces the activation at [batch_idx, position, :] 
    with `activation_value`. 
    """
    has_patched_first_step: bool = False
    def hook_fn(module, inputs, output):
        nonlocal has_patched_first_step
        if not has_patched_first_step:
            # Clone the tensor
            new_output = output[0]
            new_output[batch_idx, position, :] = activation_value
            has_patched_first_step = True
            # If the module returns a tuple, wrap the patched tensor back into a tuple
            return output

    return hook_fn

def debug_hook(module, inp, out):
    print(f"--- Hooking {module._get_name()} ---")
    if isinstance(out, tuple):
        print("Output is a tuple of length:", len(out))
        for i, val in enumerate(out):
            if hasattr(val, "shape"):
                print(f"  out[{i}] shape: {val.shape}")
            else:
                print(f"  out[{i}]:", type(val))
    else:
        print("Output is a single tensor of shape:", out.shape)
    return out


# %%

activation = cache["layers.0"][0,6, :]
activation.shape
# %%
SOURCE_LAYER = 23
TARGET_LAYER = 0
# Retrieve the activation of the source token.
pil_images = []
pil_img = Image.open("./images/x_begining.png")
pil_img = pil_img.convert("RGB")
pil_images.append(pil_img)
prepare_inputs = vl_chat_processor(
    #conversations=conversation,
    prompt="<image_placeholder>",
    images=pil_images,
    force_batchify=True
).to(deepseek_vl.device)
tokenized_input = prepare_inputs["input_ids"]
# run visual model to get the image embeddings
inputs_embeds = deepseek_vl.prepare_inputs_embeds(**prepare_inputs)
print(f"Input embeddings shape: {inputs_embeds.shape}")
# Run a sequence generation with an activation patching hook.
    # Get the cache of the source prompt
logits, cache = run_with_cache(deepseek_vl.language_model.model, inputs_embeds, prepare_inputs.attention_mask)
source_activation = cache[f"layers.{SOURCE_LAYER}"][0, 2, :] # Get the embedding of the first patch
print(f"Shape of source activation : {source_activation.shape}")
# %%
# Do a normal generation
pil_images = []
pil_img = Image.open("./images/x_begining.png")
pil_img = pil_img.convert("RGB")
pil_images.append(pil_img)
prepare_inputs = vl_chat_processor(
    #conversations=conversation,
    prompt="""Circle: a round shape,
Triangle: a shape with 3 points,
1""",
    images=[],
    force_batchify=True
).to(deepseek_vl.device)
tokenized_input = prepare_inputs["input_ids"]
# run visual model to get the image embeddings
inputs_embeds = deepseek_vl.prepare_inputs_embeds(**prepare_inputs)
print(f"Input embeddings shape: {inputs_embeds.shape}")
# Run the language model
# %%
# Run a normal sequence generation
print("Normal sequence generation:")
outputs = deepseek_vl.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=20,
    do_sample=False,
    use_cache=False
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
# %%
# Setup the activation patching hook
hook_handle = deepseek_vl.language_model.model.layers[TARGET_LAYER].register_forward_hook(create_activation_patching_hook(
    activation_value=source_activation,
    position=18
))
print("Patched sequence generation:")
outputs = deepseek_vl.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=20,
    do_sample=False,
    use_cache=False
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)

#Remove the activation patching hook
hook_handle.remove()
# %%
#pil_images = []
#pil_img = Image.open("./images/x_begining.png")
#pil_img = pil_img.convert("RGB")
#pil_images.append(pil_img)
#prepare_inputs = vl_chat_processor(
#    #conversations=conversation,
#    prompt="<image_placeholder>Question:What shape does this image contain ?\n Answer:",
#    images=pil_images,
#    force_batchify=True
#).to(deepseek_vl.device)
#tokenized_input = prepare_inputs["input_ids"]
## run visual model to get the image embeddings
#inputs_embeds = deepseek_vl.prepare_inputs_embeds(**prepare_inputs)
#inputs_embeds.shape
#outputs = deepseek_vl.language_model.generate(
#    inputs_embeds=inputs_embeds,
#    attention_mask=prepare_inputs.attention_mask,
#    pad_token_id=tokenizer.eos_token_id,
#    bos_token_id=tokenizer.bos_token_id,
#    eos_token_id=tokenizer.eos_token_id,
#    max_new_tokens=20,
#    do_sample=False,
#    use_cache=False
#)
#answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
#print(f"{prepare_inputs['sft_format'][0]}", answer)
# %%

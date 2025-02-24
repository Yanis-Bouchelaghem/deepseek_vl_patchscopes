# %%
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

from transformers import LlamaModel
from PIL import Image
from jaxtyping import Float
import torch
from typing import Callable, Dict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

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

# Load the model
model_path = "deepseek-ai/deepseek-vl-1.3b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

deepseek_vl: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
deepseek_vl = deepseek_vl.to(torch.bfloat16).cuda().eval()


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
# Prepare and run the source prompt (do only once)
pil_images = []
pil_img = Image.open("./images/x_begining.png")
pil_img = pil_img.convert("RGB")
pil_images.append(pil_img)
prepare_source_inputs = vl_chat_processor(
    prompt="<image_placeholder>",
    images=pil_images,
    force_batchify=True
).to(deepseek_vl.device)
# run visual model to get the image embeddings
source_inputs_embeds = deepseek_vl.prepare_inputs_embeds(**prepare_source_inputs)
print(f"Source input embeddings shape: {source_inputs_embeds.shape}")
# Run a sequence generation with an activation patching hook.
    # Get the cache of the source prompt
logits, cache = run_with_cache(deepseek_vl.language_model.model, source_inputs_embeds, prepare_source_inputs.attention_mask)

# Prepare the target prompt (Not run it)
prepare_target_inputs = vl_chat_processor(
    prompt="""Circle: a round shape,
Triangle: a shape with 3 points,
1""",
    images=[],
    force_batchify=True
).to(deepseek_vl.device)
# run visual model to get the image embeddings
target_inputs_embeds = deepseek_vl.prepare_inputs_embeds(**prepare_target_inputs)
# %%
def contains_standalone_x(text):
    # Regular expression to match 'x' or 'X' as a standalone word
    pattern = r'\b[xX]\b'
    return bool(re.search(pattern, text))

matrix = np.zeros((24,24), dtype=int)

LAYER_COUNT = len(deepseek_vl.language_model.model.layers)
for source_layer in range(LAYER_COUNT):
    # Extract the source activation
    source_activation = cache[f"layers.{source_layer}"][0, 1, :] # Get the embedding of the first patch (containing a cross)
    print(f"Shape of source activation at layer {source_layer}: {source_activation.shape}")
    for target_layer in range(LAYER_COUNT):
        # Setup the activation patching hook
        hook_handle = deepseek_vl.language_model.model.layers[target_layer].register_forward_hook(create_activation_patching_hook(
            activation_value=source_activation,
            position=18
        ))
        outputs = deepseek_vl.language_model.generate(
            inputs_embeds=target_inputs_embeds,
            attention_mask=prepare_target_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=20,
            do_sample=False,
            use_cache=False
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"-----------Source layer : {source_layer}; Target layer: {target_layer}; Model output:{answer}")
        if "cross" in answer or "Cross" in answer:
            matrix[source_layer, target_layer] = 2
        elif contains_standalone_x(answer):
            matrix[source_layer, target_layer] = 1
        else:
            matrix[source_layer, target_layer] = 0

        #Remove the activation patching hook
        hook_handle.remove()
# %%

# 2. Define a discrete colormap for values 0, 1, and 2.
#    You can pick any color palette you like.
custom_cmap = sns.color_palette(["#ffa7a5", "#9BF6FF", "#caffbf"])

# 2. Create the heatmap
ax = sns.heatmap(
    matrix,
    cmap=custom_cmap,
    cbar=False,         # Disable the default colorbar
    square=True,
    linewidths=0.5,     # Grid line thickness
    linecolor='white',  # Grid line color
    xticklabels=range(24),
    yticklabels=range(24)
)

# Tilt x-axis labels to 45° to avoid overlap
plt.xticks(rotation=60, ha='center')

# Axis labels and title
plt.xlabel("Target layer index")
plt.ylabel("Source layer index")
plt.title("24x24 Matrix Heatmap")

# 3. Create a custom legend in place of the colorbar
red_patch   = mpatches.Patch(color="#ffa7a5", label="Output = neither")
blue_patch  = mpatches.Patch(color="#9BF6FF", label="Output contains 'x'")
green_patch = mpatches.Patch(color="#caffbf", label="Output contains 'cross'")

plt.legend(
    handles=[green_patch, blue_patch, red_patch],
    bbox_to_anchor=(1.05, 1),
    loc=2,
    borderaxespad=0.
)

plt.show()
# %%
# Define how each value maps to a color
#val_to_color = {
#    0: "#ffa7a5",  # red   -> Output = neither
#    1: "#9BF6FF",  # blue  -> Output = 'x'
#    2: "#caffbf",  # green -> Output = 'cross'
#}
#
#fig, ax = plt.subplots(figsize=(6, 6))
#
## Draw each cell as a 1×1 FancyBboxPatch with rounded corners
#for row in range(matrix.shape[0]):
#    for col in range(matrix.shape[1]):
#        val = matrix[row, col]
#        color = val_to_color[val]
#        patch = mpatches.FancyBboxPatch(
#            (col + 0.25, row + 0.25), 0.5, 0.5, 
#            boxstyle=mpatches.BoxStyle("Round", pad=0.2), 
#            facecolor=color, 
#            edgecolor="white", 
#            linewidth=0.5
#        )
#        ax.add_patch(patch)
#
## Flip the y-axis so row=0 is at the top
#ax.set_ylim(matrix.shape[0], 0)
#ax.set_xlim(0, matrix.shape[1])
#ax.set_aspect("equal")
#
## Label ticks (shift by 0.5 to center tick labels)
#ax.set_xticks(np.arange(24) + 0.5)
#ax.set_yticks(np.arange(24) + 0.5)
#ax.set_xticklabels(range(24), rotation=45, ha="right")
#ax.set_yticklabels(range(24))
#
#plt.xlabel("Target layer index")
#plt.ylabel("Source layer index")
#plt.title("24x24 Matrix Heatmap with Rounded Cells")
#
## Create a custom legend
#red_patch   = mpatches.Patch(color="#ffa7a5", label="Output = neither")
#blue_patch  = mpatches.Patch(color="#9BF6FF", label="Output = 'x'")
#green_patch = mpatches.Patch(color="#caffbf", label="Output = 'cross'")
#plt.legend(
#    handles=[green_patch, blue_patch, red_patch],
#    bbox_to_anchor=(1.05, 1),
#    loc="upper left"
#)
#
#plt.show()
# %%

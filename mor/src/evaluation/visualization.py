from typing import Optional, List, Tuple



import torch

import torch.nn as nn





def get_routing_heatmap(

    model: nn.Module,

    input_ids: torch.Tensor,

    attention_mask: Optional[torch.Tensor] = None,

) -> Tuple[torch.Tensor, dict]:

    model.eval()

    device = next(model.parameters()).device

    input_ids = input_ids.to(device)

    if attention_mask is not None:

        attention_mask = attention_mask.to(device)



    with torch.no_grad():

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask,

            return_dict=True,

        )



        if hasattr(model, "get_recursion_depths"):

            depths = model.get_recursion_depths(input_ids, attention_mask)

        else:

            depths = None



        routing_info = outputs.routing_info if hasattr(outputs, "routing_info") else None



    return depths, routing_info





def plot_routing_heatmap(

    depths: torch.Tensor,

    tokens: Optional[List[str]] = None,

    title: str = "Recursion Depth per Token",

    save_path: Optional[str] = None,

):

    try:

        import matplotlib.pyplot as plt

        import matplotlib.colors as mcolors

    except ImportError:

        print("matplotlib not installed. Skipping visualization.")

        return



    depths = depths.cpu().numpy()

    batch_size, seq_len = depths.shape



    fig, axes = plt.subplots(batch_size, 1, figsize=(max(12, seq_len // 4), batch_size * 2))

    if batch_size == 1:

        axes = [axes]



    max_depth = depths.max()

    cmap = plt.cm.YlOrRd



    for i, (ax, depth_row) in enumerate(zip(axes, depths)):

        im = ax.imshow(

            depth_row.reshape(1, -1),

            aspect='auto',

            cmap=cmap,

            vmin=1,

            vmax=max_depth,

        )



        ax.set_yticks([])

        ax.set_xlabel("Token Position")

        ax.set_title(f"Batch {i}" if batch_size > 1 else title)



        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)

        cbar.set_label("Recursion Depth")



        if tokens is not None and i == 0:

            ax.set_xticks(range(len(tokens)))

            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)



    plt.tight_layout()



    if save_path:

        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        print(f"Saved figure to {save_path}")

    else:

        plt.show()





def plot_depth_distribution(

    depths: torch.Tensor,

    title: str = "Recursion Depth Distribution",

    save_path: Optional[str] = None,

):

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print("matplotlib not installed. Skipping visualization.")

        return



    depths_flat = depths.cpu().numpy().flatten()

    max_depth = int(depths_flat.max())



    fig, ax = plt.subplots(figsize=(8, 5))



    counts = [(depths_flat == d).sum() / len(depths_flat) * 100

              for d in range(1, max_depth + 1)]



    ax.bar(range(1, max_depth + 1), counts, color='steelblue', edgecolor='black')

    ax.set_xlabel("Recursion Depth")

    ax.set_ylabel("Percentage of Tokens (%)")

    ax.set_title(title)

    ax.set_xticks(range(1, max_depth + 1))



    for i, count in enumerate(counts):

        ax.annotate(

            f'{count:.1f}%',

            xy=(i + 1, count),

            ha='center',

            va='bottom',

            fontsize=9,

        )



    plt.tight_layout()



    if save_path:

        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        print(f"Saved figure to {save_path}")

    else:

        plt.show()





def plot_router_weights(

    routing_info: dict,

    title: str = "Router Weights by Recursion Step",

    save_path: Optional[str] = None,

):

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print("matplotlib not installed. Skipping visualization.")

        return



    if "router_weights" not in routing_info:

        print("No router weights in routing info.")

        return



    weights_list = routing_info["router_weights"]

    num_steps = len(weights_list)



    fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))

    if num_steps == 1:

        axes = [axes]



    for i, (ax, weights) in enumerate(zip(axes, weights_list)):

        if weights is None:

            continue



        weights_flat = weights.cpu().numpy().flatten()



        ax.hist(weights_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)

        ax.set_xlabel("Router Weight")

        ax.set_ylabel("Count")

        ax.set_title(f"Recursion Step {i + 1}")

        ax.set_xlim(0, 1)



    plt.suptitle(title)

    plt.tight_layout()



    if save_path:

        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        print(f"Saved figure to {save_path}")

    else:

        plt.show()





def analyze_token_routing(

    model: nn.Module,

    input_ids: torch.Tensor,

    tokenizer=None,

    top_k: int = 10,

) -> dict:

    depths, routing_info = get_routing_heatmap(model, input_ids)



    if depths is None:

        return {"error": "Model does not support depth analysis"}



    depths_flat = depths.cpu().flatten()

    input_ids_flat = input_ids.cpu().flatten()



    results = {

        "avg_depth": depths_flat.float().mean().item(),

        "std_depth": depths_flat.float().std().item(),

        "max_depth": depths_flat.max().item(),

        "min_depth": depths_flat.min().item(),

    }



    sorted_indices = depths_flat.argsort(descending=True)

    deep_tokens = []



    for idx in sorted_indices[:top_k]:

        token_id = input_ids_flat[idx].item()

        depth = depths_flat[idx].item()



        token_info = {

            "position": idx.item(),

            "token_id": token_id,

            "depth": depth,

        }



        if tokenizer is not None:

            token_info["token"] = tokenizer.decode([token_id])



        deep_tokens.append(token_info)



    results["deepest_tokens"] = deep_tokens



    shallow_indices = depths_flat.argsort()

    shallow_tokens = []



    for idx in shallow_indices[:top_k]:

        token_id = input_ids_flat[idx].item()

        depth = depths_flat[idx].item()



        token_info = {

            "position": idx.item(),

            "token_id": token_id,

            "depth": depth,

        }



        if tokenizer is not None:

            token_info["token"] = tokenizer.decode([token_id])



        shallow_tokens.append(token_info)



    results["shallowest_tokens"] = shallow_tokens



    return results


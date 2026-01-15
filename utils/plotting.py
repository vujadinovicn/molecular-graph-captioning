import matplotlib.pyplot as plt

def plot_similarity_matrix(G, T, title="", tau=0.07, normalize_logits=True, save_path=None, show=False):
    G = G.float()
    T = T.float()

    sim = G @ T.T
    logits = sim

    if normalize_logits:
        logits = (logits - logits.mean()) / (logits.std() + 1e-8)

    plt.figure(figsize=(6, 5))
    plt.imshow(logits.detach().cpu(), aspect="auto")
    plt.colorbar(label="Normalized logits" if normalize_logits else "Logits")
    plt.title(title)
    plt.xlabel("Text Embedding Sequences")
    plt.ylabel("Graph Embedding Sequences")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
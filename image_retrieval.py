import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------
# ASMK Encoding (with binary codes)
# -------------------
def asmk_encode_binary(image_tokens, centers, tau=0.0):
    """
    Encode one image into ASMK binary codes.
    tau = threshold parameter for match kernel.
    """
    c, d = image_tokens.shape
    k = centers.shape[0]

    # Assign tokens to nearest centers
    dists = torch.cdist(image_tokens, centers)  # (c, k)
    assign = dists.argmin(dim=1)                # (c,)

    # Aggregate residuals
    residuals = torch.zeros((k, d), device=image_tokens.device)
    for i, token in enumerate(image_tokens):
        cluster = assign[i]
        residuals[cluster] += token - centers[cluster]

    # Intra-normalization (per-cluster L2 norm)
    norms = torch.norm(residuals, dim=1, keepdim=True) + 1e-10
    residuals = residuals / norms

    # Signed sqrt
    residuals = torch.sign(residuals) * torch.sqrt(torch.abs(residuals))

    # Binary codes (+1 / -1)
    binary_codes = torch.sign(residuals)  # (k, d), values in {-1, 0, +1}
    binary_codes[binary_codes == 0] = 1   # treat zeros as +1

    return binary_codes, residuals


# -------------------
# ASMK Similarity with thresholding
# -------------------
def asmk_similarity(x_res, y_res, tau=0.0):
    """
    Apply ASMK match kernel between two encoded images.
    x_res, y_res: residuals (k,d) tensors
    """
    # Cluster-wise dot products
    dots = (x_res * y_res).sum(dim=1)  # (k,)

    # Selectivity function σ(z) = sign(z) * max(|z| - tau, 0)
    sim = torch.sign(dots) * torch.clamp(torch.abs(dots) - tau, min=0.0)

    return sim.sum()


# -------------------
# Run full ASMK pipeline
# -------------------
# def run_asmk(tokens, k=64, tau=0.1, whiten=True, pca_components=64):
def run_asmk(tokens, k=64, tau=0.0, whiten=False, pca_components=None):
    """
    tokens: (n, c, d) tensor of image tokens
    k: number of visual words
    tau: threshold for ASMK kernel
    whiten: if True, apply PCA whitening before kmeans/encoding
    pca_components: number of PCA dimensions (<= d), default = d
    """
    print(f"Running full ASMK with binary codes... (whiten={whiten})")

    n, c, d = tokens.shape
    all_tokens = tokens.reshape((-1, d)).detach().cpu().numpy()

    # Optional PCA whitening
    if whiten:
        if pca_components is None:
            pca_components = d
        pca = PCA(n_components=pca_components, whiten=True, random_state=0)
        all_tokens = pca.fit_transform(all_tokens)
        print(f"Applied PCA whitening → shape {all_tokens.shape}")
    else:
        pca = None

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_tokens)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=tokens.device)

    # Encode all images
    encodings = []
    residuals = []
    for i in range(n):
        img_tokens = tokens[i].detach().cpu().numpy()
        if pca is not None:
            img_tokens = pca.transform(img_tokens)
        img_tokens = torch.tensor(img_tokens, dtype=torch.float32, device=tokens.device)
        codes, res = asmk_encode_binary(img_tokens, centers, tau=tau)
        encodings.append(codes)
        residuals.append(res)
    encodings = torch.stack(encodings)   # (n, k, d)
    residuals = torch.stack(residuals)   # (n, k, d)

    # Similarity matrix
    sim_matrix = torch.zeros((n, n), device=tokens.device)
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = asmk_similarity(residuals[i], residuals[j], tau=tau)

    # Plot
    plt.figure()
    
    # Clip sim_matrx to 0 for better visualization
    # sim_matrix = torch.clamp(sim_matrix, min=0.0)

    plt.imshow(sim_matrix.detach().cpu().numpy(), cmap='viridis')
    plt.colorbar(label="ASMK Similarity")
    title = "ASMK Image Similarity Matrix"
    if whiten:
        title += " (with PCA Whitening)"
    plt.title(title)
    plt.show()

    return sim_matrix

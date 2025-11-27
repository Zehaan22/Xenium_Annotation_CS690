# 🧬 Xenium Annotation via Spatially-Aware SCModal-GNN  
### CS690: Computational Genomics (IIT Kanpur)

This repository contains the code, figures, and report for our CS690 course project on **cross-modal cell-type annotation for 10x Genomics Xenium spatial transcriptomics** using **spatially-aware latent alignment models**.

The project extends the **SCModal** dual-autoencoder framework by integrating **spatial graph neural networks (GNNs)**, **mutual nearest neighbors (MNN)** alignment, **geometric preservation losses**, and **optional niche-based conditioning** inspired by SCVIVA.

---

## 🚀 Project Summary

Xenium provides **high-resolution spatial gene expression**, but only for a *targeted gene panel*.  
scRNA-seq provides **full-transcriptome coverage** but lacks spatial context.

To bridge the modalities, we develop:
1. **Niche-SCModal** – SCModal with FiLM-based niche-conditioning  
2. **Niche-SCModal + SingleR** – adds non-parametric SingleR for label transfer  
3. **Graph-SCModal** – replaces Xenium encoder with a **GNN** to integrate spatial topology

We evaluate models on:
- Breast cancer Xenium dataset (10x Genomics)
- Broad Institute breast cancer scRNA-seq atlas

Key metrics: **Accuracy**, **ARI**, **F1-scores**, **UMAP visualization**, **spatial consistency**.

---

## 📂 Repository Structure

Xenium_Annotation_CS690/
│
├── data/
│ ├── xenium/ # Xenium spatial dataset
│ └── scrna/ # scRNA-seq reference dataset
│
├── models/
│ ├── SCModal/ # Base SCModal implementation
│ ├── Niche_SCModal/ # FiLM-based niche-conditioned model
│ └── Graph_SCModal/ # GNN-augmented SCModal encoder
│
├── utils/
│ ├── graph.py # kNN graph construction for spatial coordinates
│ ├── losses.py # Alignment, adversarial, geometric, MNN losses
│ └── preprocessing.py # Gene matching, normalization
│
├── notebooks/
│ ├── SCArches_Experiment.ipynb
│ ├── Niche_SCModal.ipynb
│ └── Graph_SCModal.ipynb
│
├── images/
│ ├── SCArches_LatentMixing.png
│ ├── Niche_SCModal_UMAP.png
│ ├── Graph_SCModal_UMAP.png
│ └── SingleR_vs_KNN.png
│
├── report/
│ ├── Final_Report.pdf
│ └── Presentation.pdf
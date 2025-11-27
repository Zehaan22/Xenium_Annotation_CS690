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

- Final_Submissions/
	- Final_Report.pdf
	- Final_Presentation.pdf

- GNN_approach/Graph_Based_Approaches_Scratch : Experimental work not used for final model

- Graph_SCModal/
	- graph_SC_Modal_testing.ipynb - run on subset dataset
    - graph_SCModal_final_run.ipynb - run on complete dataset

- SCarches_Xenium/
  	- scarches_for_xenium.ipynb - scarches run on complete dataset

- Niche_SCModal/
	- Niche_SCModal_subset.ipynb - run on subset dataset
   	- Niche_SCModal_SingleR.ipynb - run with SingleR integration
   	- Niche_SCModal_final.ipynb - run on full dataset
   	- Base_SCModal.ipynb - run of base SCModal
   	- SingleR/temp - helper files for Niche_SCModal_SingleR.ipynb

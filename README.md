# JurisEC

**A benchmark of real legal consultations and official institutional responses for Spanish legal information retrieval.**

This repository contains the code to reproduce the experiments of the paper:

> **JurisEC: A Benchmark of Real Legal Consultations and Official Institutional Responses for Spanish Legal Information Retrieval**
> Wilfredo Ivan Martel Socola and Christian Raul Salamea Palacios
> Universidad Politécnica Salesiana, Cuenca, Ecuador
> SmartTech-IC 2026 (to appear)

Dataset: [`wilfredomartel/JurisEC`](https://huggingface.co/datasets/wilfredomartel/JurisEC) on Hugging Face.

## What JurisEC is

JurisEC contains **5,536 query–answer pairs**. Each pair links a real legal consultation to the official response issued by an Ecuadorian public institution:

| Source | Institution | Pairs |
|---|---|---|
| SRI | Servicio de Rentas Internas (National Tax Administration) | 2,519 |
| PGE | Procuraduría General del Estado (Office of the State Attorney General) | 1,647 |
| CNJ | Corte Nacional de Justicia (National Court of Justice) | 1,370 |

The consultations and responses were extracted from public documents with rule-based parsing and manual quality control. The correspondence between each consultation and its response is established by the issuing institution, so the benchmark involves no relevance annotation.

## What it is for

JurisEC evaluates **retrieval systems on Spanish legal text**: given a legal consultation, a system must retrieve its official response from the whole candidate collection. It is intended for:

- comparing lexical and dense retrievers (text encoders) in a zero-shot setting;
- measuring the retrieval stage of legal retrieval-augmented generation (RAG) systems;
- studying how retrieval quality changes with lexical overlap and document length.

The paper evaluates BM25 and three multilingual encoders without fine-tuning.

## Evaluation setting

- **Queries:** 5,536 consultations.
- **Candidate documents:** 5,508, after collapsing 28 exact duplicate responses.
- **Relevance judgments:** each query is relevant to its own official response. In addition, 29 pairs of nearly identical queries (TF-IDF cosine similarity of 0.90 or higher) receive reciprocal judgments: the response to one also counts as relevant for the other.
- **Ranking:** every query is ranked against all 5,508 documents.

## Main results

Overall retrieval performance (Table 6 of the paper):

| System | nDCG@10 | MRR@10 | Acc@1 | Acc@5 | Acc@10 | R@100 | MAP@100 |
|---|---|---|---|---|---|---|---|
| BM25 | 0.767 | 0.734 | 0.657 | 0.833 | 0.874 | 0.947 | 0.736 |
| multilingual-E5-large | 0.753 | 0.720 | 0.648 | 0.816 | 0.858 | 0.931 | 0.722 |
| EmbeddingGemma-300M | **0.782** | **0.751** | **0.681** | **0.842** | **0.879** | **0.950** | **0.753** |
| BGE-M3 | 0.774 | 0.742 | 0.667 | 0.840 | **0.879** | 0.949 | 0.744 |

Paired randomization tests with Holm correction confirm that EmbeddingGemma-300M is better than BM25, but they do not separate it from BGE-M3. The complete tests are in `results/`.

## Dataset fields

| Field | Description |
|---|---|
| `consulta` | Legal consultation, used as the retrieval query |
| `resolucion_consulta` | Reference response that must be retrieved |
| `base_legal` | Legal provisions cited by the issuing authority |
| `analisis` | Legal reasoning applied by the authority |
| `conclusion` | Final institutional determination |
| `fuente` | Institution that issued the document (SRI, PGE, CNJ) |
| `materia` | Legal area |
| `tema` | More specific topic |
| `num_preguntas` | Number of questions in the consultation |
| `id` | Identifier of the benchmark instance |
| `id_registro` | Identifier of the source record |

Only `consulta` and `resolucion_consulta` take part in the retrieval task. The other fields support corpus analysis and auditing.

Load the dataset in Python:

```python
from datasets import load_dataset

ds = load_dataset("wilfredomartel/JurisEC")
split = "train" if "train" in ds else list(ds.keys())[0]
df = ds[split].to_pandas()
print(len(df), list(df.columns))
```

## Repository contents

| File | What it reproduces | GPU | Hugging Face token |
|---|---|---|---|
| `notebooks/JurisEC_1_Baseline_Evaluation.ipynb` | Overall retrieval performance (Table 6) | Yes | Yes |
| `notebooks/JurisEC_2_Benchmark_Characterization.ipynb` | Corpus statistics and figures (Section 4, Figs. 3–8) | Part B only | No |
| `notebooks/JurisEC_3_Stratified_Analysis.ipynb` | Performance by lexical overlap and document length (Section 6.2, Fig. 9) | Yes | Yes |
| `notebooks/JurisEC_4_Significance_Tests.ipynb` | Paired significance tests and bootstrap confidence intervals (Section 6.1) | Yes | Yes |
| `results/significance_pairwise.csv` | The six pairwise comparisons between systems | — | — |
| `results/significance_confidence_intervals.csv` | 95% confidence intervals for each system | — | — |

Notebooks 3 and 4 are self-contained: each one downloads the dataset and computes the embeddings it needs. The notebooks are saved with the outputs of the runs reported in the paper.

## How to run the notebooks

The notebooks are written for [Google Colab](https://colab.research.google.com/).

### 1. Create a Hugging Face account and token

A token is needed because one of the evaluated models, EmbeddingGemma-300M, is a gated model.

1. Create a free account at [huggingface.co](https://huggingface.co/join).
2. Open the model page [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) and accept its terms of use.
3. Go to **Settings → Access Tokens** and create a token with **Read** access.

### 2. Add the token to Colab

1. Open a notebook in Colab.
2. Click the **key icon (Secrets)** in the left sidebar.
3. Add a secret named `HF_TOKEN` and paste the token as its value.
4. Turn on **Notebook access** for that secret.

### 3. Enable a GPU

In Colab, choose **Runtime → Change runtime type → GPU**. The runs reported in the paper used an NVIDIA L4.

### 4. Run

Choose **Runtime → Run all**. Each notebook installs its own dependencies in its first cell.

## Reproducibility

- Random seeds are fixed in every notebook.
- BM25 results, corpus statistics, and token counts are reproduced exactly.
- The dense models run in half precision. Their scores may vary in the third decimal place depending on the hardware, so a new run can differ slightly from the values in the paper.

## Limitations

- JurisEC covers a single jurisdiction, Ecuador.
- Only the official response counts as relevant. Other legally valid responses are not judged.
- The reciprocal judgments for nearly identical queries are an evaluation convention. They do not assert that two responses are legally equivalent, and they were not reviewed by legal experts.
- The reported models were evaluated zero-shot, without fine-tuning.

## License

The corpus organization, metadata, and evaluation materials are released under **CC BY 4.0**. The institutional texts retain the legal status and terms of their official sources.

## Citation

```bibtex
@inproceedings{martel2026jurisec,
  author    = {Martel Socola, Wilfredo Ivan and Salamea Palacios, Christian Raul},
  title     = {{JurisEC}: A Benchmark of Real Legal Consultations and Official Institutional Responses for {S}panish Legal Information Retrieval},
  booktitle = {SmartTech-IC 2026},
  year      = {2026},
  note      = {To appear}
}
```

## Contact

For questions or problems, please open an issue in this repository.

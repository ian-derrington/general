This summarizes important research and survey papers and topics related to GPT-enabled technology

# Providers

* [Bard](https://bard.google.com/)
* [Claud]()
* [ChatGPT](https://openai.com/blog/chatgpt)

# Papers / Codebases / Blogposts

## Surveys and General papers
- [Formal Algorithms for Transformers in 2023](https://arxiv.org/pdf/2207.09238.pdf)
Important discussion revealing the components of Transformers.

!!! note
  **Tasks:**

  * Chunking: breaking up into smaller chunks
  * Sequence modeling (DTransformer: NExt prediction 
  * Sequence modeling (EDTransformer): Seq2Seq mappingto different domains.
  * Classification (Etransformer): classification

  **Tokenization:** 
  Starts with a vocabulary but then must be encoded in some way. 

  * Character Level: Has very long sequences
  * Word Level: Rare words don't work
  * Subword level: Simpleset and most successful is Byte Pair Encoding
  * Special Characters: `mask_token`, `bos_token` (beginning of sequence), `eos_token`

  **Components:**
  
  * Token Embedding: Mapping to a vector space. 
  * Positional Embedding: Learned or hard-coded mapping to position of sequence to a vector space
  * Attention: Token being predicted is mapped to a query vector and tokens in context are mapped to key and value vectors. Inner products are used to combine to extract information. 
  * Bi-directional / unmasked
  * Unidirectional / masked self attetion
  * Cross attention applies attention to the primary sequence and treates the second token sequence the context. 
  * Multi-head attention. Multiple attention heads in parallel.
  * Layer normalization. Found to be computationally efficient version sets m = beta = 0 or root mean square layer normalizagion or `RMSnorm`. 
  * Unembedding: Learns to convert vector intot he vocuabulary elements. 
  
  **Architectures:**

  * Encoder-Decoder (EDT), is also sequence-to-sequence. 
  * Encoder-only: (BERT)
  * Decoder-only (GPT) Next-token 
  * Multi-domain decoder-only transformer (Gato)

  **Practical considerations**

  * Data preprocessing: Cleaning, augmenting, noising, shuffling, t
  * Architectures: sparce layers, weight-sharing
  * Training: minibatch, batch norm, weight initialization, ensembling, adversarial
  * Regularization: weight decay early stopping cross-validation, dropout, noise

## Metrics:
- Exact Match (EM) 

## GPT: * 
http://jalammar.github.io/illustrated-gpt2/
  

## LLM Component concepts
### Tokenization
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

### Scaling
- [The 'Chinchilla' paper of 2022](https://arxiv.org/abs/2203.15556) This paper identifies scaling laws that help to understand the volume of data that is needed to obtain 'optimal' performance for a given LLM models size. Use of it in other areas, such as for Llama reveal that the models may have been under trained.
  - Primary take away: **"All three approaches suggest that as compute budget increases, model size andthe amount of training data should be increased in approximately equal proportions." **

## Prompt engineering


## Agentic and Recurrent GPT
This section describes GPT that has been enabled with more 'agency' or the ability to do better.

- [HuggingGPT of 2023](https://arxiv.org/pdf/2303.17580.pdf) This paper describes a paradigm where ChatGPT is enabled with the ability to launch other ML models based on input. It does so by creating a Task list, then by identifying appropriate models, and then by executing them.
  - [Github repo known as JARVIS here](https://github.com/microsoft/JARVIS)
  - [TaskMatrix.ai](https://arxiv.org/abs/2303.16434) seemingly from the same authors. 
- [AUTO GPT](https://github.com/Torantulino/Auto-GPT) ‼️
- [ReAct](https://github.com/ysymyth/ReAct) ‼️
  - [Github](https://github.com/ysymyth/ReAct) ‼️
- [Reflexion](Reflexion: an autonomous agent with dynamic memory and self-reflection): "Reflexion, an approach that endows an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities"
  - [Github](https://github.com/noahshinn024/reflexion)
  - [Inspired github](https://github.com/GammaTauAI/reflexion-human-eval) 


## Applications
### Biology
- [Evolutionary-scale prediction of atomic-level protein structure with a language model](https://www.science.org/doi/10.1126/science.ade2574) End to end Language model enabling structure sequence pairing, coupled with an equivariant transformer structure model at the end. 

### Societal simulations
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/pdf/2304.03442.pdf): 
  They gave 25 AI agents motivations & memory, and put them in a simulated town. Not only did they engage in complex behavior (including throwing a Valentine’s Day party) but the actions were rated more human than humans roleplaying.
  Demo: https://t.co/pYNF4BBveG
  
  


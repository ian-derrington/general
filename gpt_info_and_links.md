This summarizes important research and survey papers and topics related to GPT-enabled technology.

Please note that it will *not* focus the topic of AI Ethics.  While that is important, the information tends to be exceptionally verbose, and quite opinionated. 

Also, please note that it will not include discussions evaluating GPT or other LLMs, their potential consciousnes, their performance on standardized human tests, etc as this will change within weeks to months and will be deprecated by the time you likely read this. 

# Providers

* [Bard](https://bard.google.com/)
* [Claud]()
* [ChatGPT](https://openai.com/blog/chatgpt)

# Papers / Codebases / Blogposts

## Surveys and General papers
### Transformers
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

## LLMs
- [Eight Things to Know about Large Language Models](https://cims.nyu.edu/~sbowman/eightthings.pdf?utm_source=substack&utm_medium=email)
 1. LLMs predictably get more capable with increasing investment, even without targeted
 innovation.
 2. Many important LLM behaviors emerge unpredictably as a byproduct of increasing investment.
 3. LLMs often appear to learn and use representations of the outside world.
 4. There are no reliable techniques for steering
 the behavior of LLMs.
 5. Experts are not yet able to interpret the inner
 workings of LLMs.
 6. Human performance on a task isn’t an upper
 bound on LLM performance.
 7. LLMs need not express the values of their
 creators nor the values encoded in web text.
 8. Brief interactions with LLMs are often misleading.

- Observations:
  1. LLM Output can be ambiguous 
  2. LLM output can be inconsistent because of stochasticity --> Prompt engineering is possible.

### Links

- [LLM Engineering](https://huyenchip.com/2023/04/11/llm-engineering.html)

  
## Metrics:
- Exact Match (EM) 

### GPT
- [Five years of progress in GPTs](https://finbarrtimbers.substack.com/p/five-years-of-progress-in-gpts?utm_source=substack&utm_medium=email)
Excellent summary of the progress of GPT over time, revealing core components, optimizations, and essential variations to the major Foundation model architectures.


## LLM Component concepts
### Tokenization
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

### Scaling
- [The 'Chinchilla' paper of 2022](https://arxiv.org/abs/2203.15556) This paper identifies scaling laws that help to understand the volume of data that is needed to obtain 'optimal' performance for a given LLM models size. Use of it in other areas, such as for Llama reveal that the models may have been under trained.
  - Primary take away: **"All three approaches suggest that as compute budget increases, model size andthe amount of training data should be increased in approximately equal proportions." **


## GPT: * 
http://jalammar.github.io/illustrated-gpt2/
  


## Prompt engineering

### Summary: 

- Provide several examples to ground it.
  -  Good to evaluate this and see if input examples give expected scores. Modify the prompt if it isn't. 
- Consider prompt versioning to keep track of outputs more easily.
- Breag prompts into smaller prompts
- Chain of Thought Prompting
- Generate many outputs and pick final one or use LLM to pick best one. [Self consistency technique](https://arxiv.org/pdf/2203.11171.pdf)

### Links
- ‼️ [Prompt Engineering by Lillian Wang](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Best practices for prompt engineering](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)
- [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)


## Improvements and Optimizations

### Pruning

- [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774) Remove up to ~50% parameters preserving performance
- [Scaling Expert Language Models with Unsupervised Domain Discovery](https://arxiv.org/pdf/2303.14177.pdf) Cluster-Branch-Train-Merge (c-BTM), a new way to scale sparse expert LLMs on any dataset. 
 - [Github](https://github.com/kernelmachine/cbtm) 

### Memory Augmented
* [Improving language models by retrieving from trillions of tokens](https://arxiv.org/pdf/2112.04426.pdf)

### Training variations
- [LinkBERT](https://github.com/michiyasunaga/LinkBERT) places in context window hyperlinked references to achieve better performance.  
- [Cluster-Branch-Train-Merge (c-BTM)], a new way to scale sparse expert LLMs on any dataset Paper: https://arxiv.org/abs/2303.14177 Code + Models: https://github.com/kernelmachine/



## Extensions

### Multimodal
* ‼️ [Visual GPT](https://arxiv.org/pdf/2303.04671.pdf)

* [Language is not all you need](https://arxiv.org/pdf/2302.14045.pdf)


### Agentic, Recurrent and Pipelining GPT
This section describes GPT that has been enabled with more 'agency' or the ability to do better.
- [Language Models can Solve Computer Tasks](https://arxiv.org/pdf/2303.17491.pdf): 
  - Explicit RCI: "Review your previous answer and find problems with your answer." --> "Based on the problems you found, improve your answer." **R**ecursively **C**riticizes and **I**mproves it s output. This sort of prompting outperforms Chain of Thought, and combined it works even better.  
  - Implicit RCI: "
- [HuggingGPT of 2023](https://arxiv.org/pdf/2303.17580.pdf) This paper describes a paradigm where ChatGPT is enabled with the ability to launch other ML models based on input. It does so by creating a Task list, then by identifying appropriate models, and then by executing them.
  - [Github repo known as JARVIS here](https://github.com/microsoft/JARVIS)
  - [TaskMatrix.ai](https://arxiv.org/abs/2303.16434) seemingly from the same authors. 
- ‼️ [AUTO GPT](https://github.com/Torantulino/Auto-GPT) 
- ‼️ [BabyAGI](https://github.com/yoheinakajima/babyagi)
- ‼️ [ReAct](https://arxiv.org/abs/2210.03629)
  - [Github](https://github.com/ysymyth/ReAct) 
- [Reflexion](Reflexion: an autonomous agent with dynamic memory and self-reflection): "Reflexion, an approach that endows an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities"
  - [Github](https://github.com/noahshinn024/reflexion)
  - [Inspired github](https://github.com/GammaTauAI/reflexion-human-eval) 
- ‼️[Langchain](https://python.langchain.com/en/latest/#): Data aware AI that is agentic.
  - ‼️[Langflow](https://github.com/logspace-ai/langflow) 
  - ‼️[Toolkit](https://www.toolkit.club/) Generates LangChain plugins
- [Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128) `trans
<img width="865" alt="image" src="https://user-images.githubusercontent.com/76016868/231906559-758d89e4-d22a-4a3a-aa96-1d630e48651d.png">


## Applications

### Robotics

- [CLAIRIFY](https://ac-rad.github.io/clairify/) Translates english to domain specific languages like robots. 
  - https://arxiv.org/abs/2303.14100


### Computer tasks

- [Looped Transformers as Programmable Computers](https://arxiv.org/pdf/2301.13196.pdf): "We demonstrate that
a constant number of encoder layers can emulate basic computing blocks, including embedding edit operations, non-linear functions, function calls, program counters, and conditional branches. Using these building blocks, we emulate a small instruction-set computer."

### Biology

- [Evolutionary-scale prediction of atomic-level protein structure with a language model](https://www.science.org/doi/10.1126/science.ade2574) End to end Language model enabling structure sequence pairing, coupled with an equivariant transformer structure model at the end. 
-  https://arxiv.org/pdf/2303.16416.pdf
-  https://arxiv.org/abs/2304.02496

### Societal simulations
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/pdf/2304.03442.pdf): 
  They gave 25 AI agents motivations & memory, and put them in a simulated town. Not only did they engage in complex behavior (including throwing a Valentine’s Day party) but the actions were rated more human than humans roleplaying.
  Demo: https://t.co/pYNF4BBveG
  
  
## Interesting Companies:

- [e2b](https://github.com/e2b-dev/e2b) Write documentation, get code. 
- [Codium](https://www.codium.ai/blog/codiumai-powered-by-testgpt-accounces-beta-and-raised-11m/?utm_source=substack&utm_medium=email)

# Relevant + Useful 

Data sets (To be made into different document)

- https://arxiv.org/pdf/2303.14957.pdf

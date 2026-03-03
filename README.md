## TokenShip

Implementation of the **TokenSkip** paper and corresponding extensions.

- GitHub (original): https://github.com/hemingkx/TokenSkip  
- Paper: https://arxiv.org/abs/2502.12067  

Please note Llama requires `HF_TOKEN`. Make sure you are approved for:  
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct  

**Team:**

- Vybhav Chaturvedi  
- Anukul Vats  
- Kingshuk Barua  
- Lavesh Nihalani 

---

### Checklist

- [x] Ported evaluation to Kaggle H100 
- [x] Implemented prompt and truncation baselines on **MATH‑500**  
- [x] Evaluated **Qwen2.5‑7B‑Instruct** and **Llama‑3.1‑8B‑Instruct** on MATH‑500  
- [x] Logged per‑method metrics to CSVs and basic plots (accuracy, tokens, latency, efficiency, categories)
- [ ] Implement FlashAttention2, `torch.compile`, batched/length‑sorted
- [ ] **Add GSM8K evaluation** for both models (reuse prompts/metrics pipeline)  
- [ ] Export GSM8K CSVs matching MATH-500 format  
- [ ] **LLMLingua-2 integration**: Token importance scoring on CoT traces  
- [ ] Generate **compressed CoT datasets** at target ratios (0.5, 0.6, 0.7)  
- [ ] Evaluate **TokenSkip compression** vs truncation baselines  
- [ ] **LoRA/SFT on compressed CoT** (Llama-3.1-8B primary target)  

### Future/Phase 3
- [ ] **Train BERT/small-context + large-context models** to find collapse threshold  
- [ ] Compare trained vs frozen model (accuracy vs speedup)  
- [ ] Cross-model comparison plots (Qwen vs Llama)  
- [ ] Adaptive compression by difficulty level - Entropy Calculation

# Chain-of-Scheduling
This repository includes the code of paper "CoS: Towards Optimal Event Scheduling via Chain-of-Scheduling". 
# Abstract
Recommending event schedules is a key issue in Event-based Social Networks (EBSNs) in order to maintain user activity. An effective recommendation is required to maximize the user’s preference, subjecting to both time and geographical constraints. Existing methods face an inherent tradeoff among efffciency, effectiveness, and generalization, due to the NP-hard nature of the problem. This paper proposes the Chain-of-Scheduling (CoS) framework, which activates the event scheduling capability of Large Language Models (LLMs) through a guided, efffcient scheduling process. CoS enhances LLM by formulating the schedule task into three atomic stages, i.e., exploration, veriffcation and integration. Then we enable the LLMs to generate CoS autonomously via
Knowledge Distillation (KD). Experimental results show that CoS achieves near-theoretical optimal effectiveness with high efffciency on three real-world datasets in a interpretable manner. Moreover, it demonstrates strong zero-shot learning ability on out-of-domain data.
# Versions of backbone
The versions of the backbone models are Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3, which can be downloaded from the link: https://huggingface.co/
# Datasets
Due to user privacy concerns, we have only disclosed partial data. The complete data can be accessed on Meetup: https://www.meetup.com/

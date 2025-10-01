<div align="center">

# About my fork repository


This is an individual project that dives into the safety and ethical aspects of AI in gene-editing. Influenced by [CRISPR-GPT](https://www.nature.com/articles/s41551-025-01463-z), a multi-agent system capable of designing gene-editing experiments and providing general/expertise knowledge, my repository addresses safety issues in unethical prompts and successfully develops a specialized agent to overcome any jailbreaking. 


Please review the foundation project of CRISPR-GPT to understand the core structure: [base code](https://github.com/cong-lab/crispr-gpt-pub)

</div>

## Problem notice

I notice that the safety.py are using static checking with fixed keywords that could be bypass by synonym replacement in vague context.

Here is an example of tricky prompt:
<p align="center">
<img width="600" alt="CRISPR-GPT Overview" src="assets/safety_problem.png">
</p>
As you can see in the above figure, 'adult sperm' is supported while 'human sperm' gets rejected.

## Solution approach
The idea of ethical agent is simple: ultilize safety.py functions as tools, apply basic embedding for semantic vector search, provide detailed template with various criteria for contextual reasoning (potential to apply in RLHF).

<p align="center">
<img width="600" alt="CRISPR-GPT Overview" src="assets/Ethical_Agent.jpg">
</p>

## Result
Successfully detects violated prompts and provide warning records for developer.

<p align="center">
<img width="600" alt="CRISPR-GPT Overview" src="assets/safety_result.png">
</p>

## Limitation
- The public CRISPR-GPT is the lite version (compared to the one in paper). Only capable of knock-out function while prime editing and target gene activation/regression holds the real power in CRISPR.

-  Limit gene-editing document access.

-  I use Gemini 2.5 Flash because of poverty.

- No specialized training or fine-tuning (although I really want to try RLHF).

- Poor understanding of CRISPR tool and gene field to create more scenarios.

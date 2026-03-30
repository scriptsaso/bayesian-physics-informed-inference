from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

def interpret_posterior(shap_summary: dict, posterior_stats: dict) -> str:
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""
You are a materials science expert specializing in thin film morphology 
and Bayesian inference. Interpret the following results:

Posterior statistics: {posterior_stats}
SHAP attributions: {shap_summary}

Provide:
1. Physical interpretation of the dominant structural descriptors
2. Assessment of interfacial area behavior under current conditions
3. Suggested next experimental conditions to explore
4. Confidence level in current conclusions
"""
        }]
    )
    return response.content[0].text
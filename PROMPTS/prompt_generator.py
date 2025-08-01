from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    input_variables=["paper", "style", "length"],
    validate_template=True,  #checks is all place holders are present or not
    template="""
You are an AI assistant specializing in NLP research papers. Your job is to summarize the paper titled: "{paper}".

Use a **{style}** style:
- If beginner-friendly: explain using simple language and analogies, avoid jargon, and define terms.
- If technical: focus on model architecture, training process, and evaluation metrics.
- If mathematical: explain equations, theoretical motivations, and proofs if applicable.
- If code-oriented: emphasize implementation details, pseudocode, libraries, and practical usage.

Make the explanation **{length}**:
- short (1–2 paragraphs): provide only the key contributions and impact.
- medium (3–5 paragraphs): include background, method, and results.
- long (detailed explanation): cover all sections including introduction, methods, results, discussion, and applications.

Ensure clarity, structure, and completeness. Begin now.
"""
)

template.save('template.json')
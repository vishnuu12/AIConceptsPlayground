"""
A tiny knowledge base, shared by the retrieval and evaluation demos.

It describes this repository itself, which is convenient: you can check every
answer against the README, so a wrong answer is obviously wrong rather than
subtly wrong.

In a real system this would be your documentation, your tickets, your policies -
and it would be far larger, which is exactly why chunking and retrieval matter.
"""

DOCUMENTS: list[dict] = [
    {
        "id": "overview",
        "title": "What this repository is",
        "text": (
            "AIConceptsPlayground is a repository of runnable demos for machine learning and "
            "LLM engineering. Part 1 implements the algorithms from scratch with NumPy, including "
            "linear regression, logistic regression, k-means, PCA and a small neural network with "
            "hand-written backpropagation. Part 2 builds the pieces of a language model application "
            "on top of a deterministic mock model, so nothing needs an API key or a network connection."
        ),
    },
    {
        "id": "gradient-descent",
        "title": "Gradient descent and the learning rate",
        "text": (
            "Gradient descent updates the parameters by subtracting the gradient multiplied by the "
            "learning rate. If the learning rate is too small the run crawls and never converges in "
            "reasonable time. If it is too large the steps overshoot the minimum and the loss diverges. "
            "For mean squared error the update is stable while the learning rate is below two divided "
            "by the largest curvature of the loss."
        ),
    },
    {
        "id": "loss",
        "title": "Why cross-entropy for classification",
        "text": (
            "Cross-entropy is the standard loss for classification. Squared error is a poor choice "
            "because its gradient with respect to the pre-sigmoid value is multiplied by the sigmoid "
            "derivative, which goes to zero when the model is confident. A confidently wrong model "
            "therefore receives an almost zero gradient and learns very slowly. Cross-entropy does not "
            "have that factor, so a confidently wrong prediction produces a large correction."
        ),
    },
    {
        "id": "overfitting",
        "title": "Overfitting and regularisation",
        "text": (
            "Overfitting is when a model fits the noise in the training data and performs badly on new "
            "data. It is detected by comparing training error with error on a held-out test set: a large "
            "gap means overfitting. Ridge regularisation adds a penalty on the size of the weights, which "
            "shrinks them and reduces that gap. Lasso does the same but can drive weights exactly to zero."
        ),
    },
    {
        "id": "rag",
        "title": "Retrieval augmented generation",
        "text": (
            "Retrieval augmented generation grounds an answer in documents that were retrieved for the "
            "question. The pipeline is: split the documents into chunks, embed each chunk into a vector, "
            "embed the question, retrieve the chunks with the highest cosine similarity, and put those "
            "chunks into the prompt as context. Grounding reduces hallucination and lets the answer cite "
            "its source. Without retrieval the model answers from its parameters alone and may invent a "
            "confident, unsupported answer."
        ),
    },
    {
        "id": "chunking",
        "title": "Chunking",
        "text": (
            "Chunking decides how documents are split before embedding. Chunks that are too large dilute "
            "the embedding, so retrieval matches poorly. Chunks that are too small lose the surrounding "
            "context needed to answer. A common compromise is a few hundred tokens per chunk with a small "
            "overlap between neighbours, so a sentence split across a boundary still appears whole in at "
            "least one chunk."
        ),
    },
    {
        "id": "evaluation",
        "title": "Evaluation",
        "text": (
            "Evaluation is how you change a prompt or a model without breaking things. You collect a "
            "golden set of questions with known correct answers, run the system over all of them, and "
            "score each answer. Common scores are exact match, keyword or fact presence, and a rubric "
            "judged by a model. A change is only an improvement if the pass rate goes up and no previously "
            "passing case regresses."
        ),
    },
    {
        "id": "tokens",
        "title": "Tokens and context windows",
        "text": (
            "A language model reads tokens, not words. Common words are usually a single token while rare "
            "words and identifiers are split into several. Token count determines the cost of a request and "
            "how much text fits in the context window. A long retrieved context therefore competes with the "
            "question and the answer for space, which is one reason retrieval returns only the top few chunks."
        ),
    },
]


def all_text() -> str:
    return "\n\n".join(document["text"] for document in DOCUMENTS)

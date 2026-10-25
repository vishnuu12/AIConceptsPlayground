"""
Cross-cutting interview topics.

The per-concept questions live with their concept. These are the ones that do not
belong to a single page: how you would approach a problem end to end, and how you
would work on a team. They are usually the questions that separate a candidate who
has read about ML from one who has shipped something.
"""

EXTRA_TOPICS: list[dict] = [
    {
        "topic": "Approaching a problem end to end",
        "questions": [
            {
                "q": "A product manager asks you to 'add AI' to a feature. What do you do first?",
                "a": "Ask what decision the feature is making and what a wrong answer costs. Then ask whether "
                "you have data and labels, and whether a simple rule or a keyword search would already be "
                "good enough. Most 'add AI' requests are really 'make this search better' or 'classify these "
                "tickets', and both have cheaper baselines. You cannot evaluate a model without knowing what "
                "good looks like.",
            },
            {
                "q": "How would you build a document question-answering system for a company wiki?",
                "a": "Start by measuring the baseline: how often does keyword search already find the right "
                "page? Then chunk the documents on headings with overlap, embed them, index the vectors, "
                "retrieve the top k, rerank, and ground a prompt that must cite the passage. Build a golden "
                "set of real questions with known answers, measure retrieval and generation separately, and "
                "ship behind a feedback button so failures feed the golden set.",
            },
            {
                "q": "How do you decide whether a model is good enough to ship?",
                "a": "By comparing it against the current behaviour on a held-out set, and by asking what "
                "error rate the product can tolerate. A classifier that is 92% accurate may be excellent or "
                "useless depending on whether the 8% are evenly spread or all concentrated in the case that "
                "matters most. Define the threshold before you train, or you will move it to fit the result.",
            },
            {
                "q": "How would you monitor a model in production?",
                "a": "Watch the inputs and the outputs. Input drift: are the features arriving with the same "
                "distribution as in training? Output drift: is the prediction distribution shifting? Plus "
                "business metrics, latency and cost. Where labels arrive late, sample and label a small "
                "batch periodically to estimate live accuracy - without labels you are flying blind.",
            },
            {
                "q": "When would you deliberately NOT use machine learning?",
                "a": "When a rule is exact and cheap (tax calculation), when the cost of a wrong answer is "
                "high and the explanation must be auditable, when there is no data or no way to label it, or "
                "when a simple heuristic already meets the bar. Saying this in an interview is usually worth "
                "more than another model architecture.",
            },
            {
                "q": "How do you handle imbalanced classes?",
                "a": "First, stop using accuracy - a 99% majority class makes accuracy meaningless. Use "
                "precision, recall, F1 or AUC-PR, and pick the metric that matches the cost of each error "
                "type. Then try class weights, resampling, focal loss, or simply moving the decision "
                "threshold, which is often the cheapest fix of all.",
            },
        ],
    },
    {
        "topic": "Working as an engineer",
        "questions": [
            {
                "q": "How do you make an experiment reproducible?",
                "a": "Seed everything, pin your library versions, version the data as well as the code, and "
                "log the full configuration with every run. A result you cannot reproduce is not a result. "
                "This is also why writing the demo first and the write-up second matters - the numbers in "
                "this repo come from a run, not from memory.",
            },
            {
                "q": "How do you avoid leaking the test set?",
                "a": "Split once, early, and treat the test set as read-only until the final report. Do all "
                "tuning on the validation split. For time series, split by time rather than at random, or "
                "you train on the future. And be careful with preprocessing: fitting a scaler on the full "
                "dataset before splitting is a leak, and a very common one.",
            },
            {
                "q": "What would you put in a code review of an ML pull request?",
                "a": "Is the data split correct and leak-free? Is the metric the right one for the problem? "
                "Are the seeds and versions pinned? Is there a test that would fail if the model stopped "
                "learning - the kind in this repo, asserting that the loss falls or the accuracy clears a "
                "threshold? Is the evaluation on a held-out set, and is the baseline reported next to it?",
            },
            {
                "q": "How do you keep up with the field?",
                "a": "Read the primary sources for the things you actually use - the paper or the official "
                "docs, not the summary. Reimplement one small thing at a time, because implementing "
                "something is the only way to find out what you did not understand. And be sceptical of "
                "benchmarks: a leaderboard number rarely tells you how a method behaves on your data.",
            },
            {
                "q": "What is the most common mistake you see in ML projects?",
                "a": "Building the model before defining how it will be evaluated. If you cannot say what a "
                "good answer looks like and how you will measure it, you cannot tell whether any of your "
                "work helped - and you will not notice the moment it started making things worse.",
            },
            {
                "q": "How would you explain a model's decision to a non-technical stakeholder?",
                "a": "In terms of the inputs that moved the answer and how much, not the architecture. "
                "'This application was declined mainly because of the debt-to-income ratio' is useful; "
                "'the gradient boosted trees assigned a logit of -1.4' is not. If the model cannot be "
                "explained that way, that is itself a finding worth reporting.",
            },
        ],
    },
]

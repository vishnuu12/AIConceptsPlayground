"""
Part 2 content: the six LLM-engineering pages.

Same shape as Part 1. The important difference is the honesty note that appears
on every one of these pages: the model is a simulator, and what is real is the
engineering around it. That distinction is stated on the page rather than hidden
in a README, because a reader who mistakes the simulator for a model would draw
the wrong conclusions.
"""

PART_2: list[dict] = [
    {
        "number": 10,
        "slug": "embeddings",
        "part": 2,
        "title": "Embeddings and cosine similarity",
        "blurb": "Turn text into vectors so similarity becomes geometry - and see honestly where a cheap embedding fails.",
        "why": [
            "An embedding turns text into a list of numbers such that similar meanings land close together. "
            "That one idea is what makes semantic search, clustering, recommendations and RAG possible.",
            "Once text is a vector, everything from Part 1 applies: you can measure distance, find "
            "neighbours, cluster, and reduce dimensions.",
        ],
        "sections": [
            {
                "heading": "The only formula you need",
                "code": (
                    "cosine similarity = (a . b) / (|a| * |b|)\n\n"
                    "# it is the cosine of the angle between the vectors, so it ignores length:\n"
                    "# 'cat' and 'the cat' point the same way and score about 1.0"
                ),
            },
            {
                "heading": "What this demo actually computes",
                "paragraphs": [
                    "Training a real embedding model needs a lot of data and compute, so this demo uses the "
                    "honest small-scale stand-in: feature hashing. Words and word pairs are hashed into "
                    "fixed-size buckets, counted, and normalised.",
                    "That captures SHARED WORDS, not meaning. 'car' and 'automobile' score zero here because "
                    "they share no characters, where a real model would place them next to each other. Every "
                    "other lesson on this page - the geometry, the cosine, the neighbour search, the failure "
                    "modes - is identical.",
                ],
                "code": (
                    "def embed(text, dim=256):\n"
                    "    vector = zeros(dim)\n"
                    "    for word in tokens(text):\n"
                    "        vector[stable_hash(word) % dim] += 1     # crc32, not hash()\n"
                    "    return vector / norm(vector)"
                ),
            },
            {
                "heading": "Details that bite in production",
                "bullets": [
                    "Use a STABLE hash. Python's built-in hash() is randomised per process, so embeddings "
                    "would differ between runs and your index would be meaningless after a restart.",
                    "Normalise up front, so the dot product alone is the similarity.",
                    "Cosine ignores magnitude, which is what you want for text of different lengths - but "
                    "Euclidean distance is right when magnitude is meaningful.",
                    "Real embeddings have a dimension (768, 1536, 3072) and a cost. Store them once; do not "
                    "re-embed on every query.",
                    "The same model must embed both the documents and the query, or the vectors are not "
                    "comparable at all.",
                ],
            },
        ],
        "results": [
            "self-similarity of a sentence with itself: 1.0000",
            "average similarity between different sentences: 0.078",
            "short words and stopwords are dropped before hashing - otherwise every "
            "sentence matches every question on 'the' and 'is'",
            "cos('car','automobile') = 0.000 - the honest failure of shared-word matching",
        ],
        "demo": "python -m mlconcepts.llm.embeddings",
        "plot": "10-embeddings.png",
        "interview": [
            {
                "q": "Why cosine similarity rather than Euclidean distance for text?",
                "a": "Because it ignores magnitude. A long document and a short query can point in the same "
                "direction while being far apart in Euclidean distance. Cosine measures the angle, so it "
                "answers 'are these about the same thing' rather than 'are these the same size'.",
            },
            {
                "q": "What is an embedding, really?",
                "a": "A learned mapping from text to a fixed-length vector where distance encodes semantic "
                "similarity. The model is trained so that texts used in similar contexts end up close - "
                "which is why it captures meaning that shared words alone cannot.",
            },
            {
                "q": "How would you find similar documents at scale?",
                "a": "Store the vectors in a vector index (FAISS, pgvector, Qdrant) and use approximate "
                "nearest neighbour search. Exact search is O(n) per query, which is fine for thousands of "
                "documents and hopeless for millions - ANN trades a little recall for orders of magnitude.",
            },
            {
                "q": "What is the difference between an embedding and a token?",
                "a": "A token is a unit of text the model reads - a word or subword. An embedding is the "
                "vector that unit maps to. Tokenization comes first; embeddings are what the model does with "
                "the tokens.",
            },
        ],
    },
    {
        "number": 11,
        "slug": "tokenization",
        "part": 2,
        "title": "Tokenization and sampling",
        "blurb": "Byte pair encoding from scratch, plus the real softmax/temperature rule that turns scores into a choice.",
        "why": [
            "Two things sit between your text and a model, and both are usually invisible: tokenization, "
            "which decides what the model reads, and sampling, which decides what it says.",
            "Token counts determine your cost and how much fits in a context window. Temperature and top-k "
            "determine whether the model is predictable or creative. Both are implemented for real here - "
            "no simulator, no API.",
        ],
        "sections": [
            {
                "heading": "Byte pair encoding, in four lines",
                "bullets": [
                    "Start with every word split into characters.",
                    "Count how often each adjacent pair of symbols appears.",
                    "Merge the most frequent pair into a single symbol.",
                    "Repeat a few thousand times.",
                ],
                "paragraphs": [
                    "That is how GPT's vocabulary was built. Common words end up as one token, rare words stay "
                    "as several pieces, and nothing is ever unknown - because in the worst case every "
                    "character is its own token.",
                ],
            },
            {
                "heading": "Token count is not word count",
                "code": (
                    "model          -> 1 token    ['model</w>']\n"
                    "models         -> 2 tokens   ['model', 's</w>']\n"
                    "training       -> 3 tokens   ['train', 'i', 'ng</w>']\n"
                    "tokenization   -> 9 tokens\n"
                    "unbelievable  -> 11 tokens"
                ),
                "paragraphs": [
                    "A non-English word or a code identifier can cost five tokens where an English word costs "
                    "one. That is why the same prompt is more expensive in some languages, and why 'just fit "
                    "it in the context window' is a real engineering constraint rather than a figure of speech.",
                ],
            },
            {
                "heading": "Sampling: the real rule",
                "code": (
                    "probabilities = softmax(logits / temperature)\n\n"
                    "# temperature -> 0 : a spike on the largest logit (deterministic)\n"
                    "# temperature = 1 : the model's actual distribution\n"
                    "# temperature > 1 : flatter, more random\n\n"
                    "# top-k: keep only the k highest logits before sampling"
                ),
                "paragraphs": [
                    "The model outputs a score for every token in the vocabulary. Softmax turns those scores "
                    "into probabilities, temperature reshapes them, top-k or top-p trims the tail, and one "
                    "token is sampled. Then it is fed back in and the whole thing repeats.",
                ],
            },
            {
                "heading": "Practical consequences",
                "bullets": [
                    "temperature 0 for extraction, classification and anything you will parse. It is not "
                    "perfectly deterministic on real APIs, but it is the closest you get.",
                    "temperature 0.7-1.0 for writing. Above about 1.2 the output stops being coherent.",
                    "top_p (nucleus) usually beats top_k, because it adapts to how peaked the distribution is.",
                    "Long outputs cost tokens, and tokens cost money and time - max_tokens is a budget, not a "
                    "suggestion.",
                ],
            },
        ],
        "results": [
            "'model' is 1 token, 'models' is 2, 'tokenization' is 9, 'unbelievable' is 11",
            "entropy at T=0.2: 0.030, T=1.0: 1.345, T=2.0: 1.836",
            "top_k=1 always returns the argmax token",
        ],
        "demo": "python -m mlconcepts.llm.tokenization",
        "plot": "11-tokenization.png",
        "interview": [
            {
                "q": "Why do models use subword tokenization instead of words?",
                "a": "A word vocabulary cannot handle a word it has never seen, and it would be enormous. "
                "Character-level loses the benefit of common words being a single unit. Subwords are the "
                "compromise: frequent words become one token, rare ones split into pieces, and there is no "
                "out-of-vocabulary case.",
            },
            {
                "q": "What does temperature actually do?",
                "a": "It divides the logits before the softmax. Lower temperature sharpens the distribution "
                "towards the most likely token; higher flattens it towards uniform. It is a dial on "
                "randomness, not on quality - too high and the output stops being coherent.",
            },
            {
                "q": "Why does the same prompt cost different amounts in different languages?",
                "a": "Because the vocabulary was learned mostly from English text, so English words compress "
                "into single tokens while other languages need several tokens per word. Same meaning, more "
                "tokens, higher cost and less room in the context window.",
            },
            {
                "q": "top_k or top_p?",
                "a": "top_p (nucleus sampling) keeps the smallest set of tokens whose probabilities sum to p, "
                "so the cutoff adapts to how confident the model is. top_k always keeps a fixed number, "
                "which is arbitrary when the distribution is flat and too generous when it is peaked.",
            },
        ],
    },
    {
        "number": 12,
        "slug": "prompting",
        "part": 2,
        "title": "Prompting and structured output",
        "blurb": "A prompt is a program: roles, few-shot examples, and a validate-and-repair loop that knows when to give up.",
        "why": [
            "A prompt is not a sentence you type at a model - it is a program with parts that have different "
            "jobs. The system message sets the contract, the user message carries the request and the data, "
            "and few-shot examples pin down the format.",
            "The other half of this page is the part people skip: never trust the output to parse. Ask for "
            "JSON, parse it in a try/except, validate it against your schema, and when it fails, send the "
            "error back - with a hard attempt limit, because some failures cannot be repaired.",
        ],
        "sections": [
            {
                "heading": "The parts of a prompt",
                "bullets": [
                    "system - who the model is, the rules, and the output contract. The most under-used part.",
                    "user - the actual request plus any data it needs.",
                    "assistant - few-shot examples of exactly the answer you want.",
                    "tool - the results of tools the model asked you to run.",
                ],
                "paragraphs": [
                    "Two examples can pin down a format better than a paragraph of instructions - and you pay "
                    "for them on every single request. This demo measures that cost: 107 tokens with examples "
                    "against 57 without.",
                ],
            },
            {
                "heading": "Validate, then repair, then give up",
                "code": (
                    "for attempt in range(1, max_attempts + 1):\n"
                    "    response = llm.chat(messages)\n"
                    "    payload  = try_parse_json(response.text)     # may be None\n"
                    "    problems = validate(payload)                 # your schema, your rules\n"
                    "    if not problems:\n"
                    "        return payload\n"
                    "    messages += [assistant(response.text), user(f'rejected: {problems}')]\n"
                    "return None        # give up and tell the user, do not loop forever"
                ),
                "paragraphs": [
                    "json.loads succeeding does not mean the object is usable. It can be missing fields or "
                    "have nulls where you needed strings, and that failure shows up much later as a KeyError "
                    "in production. Validate immediately, at the boundary.",
                ],
            },
            {
                "heading": "What good looks like",
                "bullets": [
                    "Temperature 0 for anything you will parse.",
                    "The schema stated in the prompt AND enforced in code - the prompt is a request, not a "
                    "guarantee.",
                    "Errors reported back with the offending text, so the model has what it needs to fix it.",
                    "A hard attempt limit, and a clear user-facing error when you give up.",
                    "Log the rejected outputs. They are the best source of test cases you will ever get.",
                ],
            },
        ],
        "results": [
            "few-shot prompt: 107 tokens; without examples: 57 (the examples cost 50 on every call)",
            "clean request: extracted name, email and city on attempt 1",
            "impossible request: rejected on all 3 attempts, then gave up correctly",
        ],
        "demo": "python -m mlconcepts.llm.prompting",
        "plot": None,
        "interview": [
            {
                "q": "How do you get reliable JSON out of a model?",
                "a": "Ask for it explicitly, state the schema, use temperature 0, then parse inside a "
                "try/except and validate against your own schema. If it fails, send the validation error "
                "back with the original input and retry - with a bounded number of attempts. If the provider "
                "supports a structured-output or tool-calling mode, use it; that is stronger than asking "
                "nicely.",
            },
            {
                "q": "When are few-shot examples worth the tokens?",
                "a": "When the format is easier to show than to describe, when the task has edge cases the "
                "instructions keep missing, or when you need consistent tone. They cost tokens on every "
                "call, so measure the improvement - if two examples do not beat a clear instruction, drop them.",
            },
            {
                "q": "What is prompt injection and how do you reduce it?",
                "a": "Untrusted text that reaches the prompt can contain instructions the model then follows. "
                "Reduce it by keeping untrusted content clearly fenced and labelled as data, never letting "
                "retrieved text or user input define the system rules, and treating every model output as "
                "untrusted - validate it before acting, especially before calling a tool.",
            },
            {
                "q": "Why is a repair loop not always the answer?",
                "a": "Because some failures are not fixable by retrying - if the input genuinely lacks the "
                "field you asked for, no retry invents it. Past two or three attempts you are burning money "
                "to hide a problem the user should be told about.",
            },
        ],
    },
    {
        "number": 13,
        "slug": "rag",
        "part": 2,
        "title": "Retrieval augmented generation",
        "blurb": "Chunk, embed, retrieve, ground - and the same question answered with and without the source text.",
        "why": [
            "A model only knows what is in its parameters, and its parameters are frozen. Ask about your "
            "documentation and it can admit it does not know or invent something plausible. Retrieval fixes "
            "that by putting the relevant text into the prompt.",
            "It reduces hallucination not because the model becomes truthful, but because you stopped asking "
            "it to remember. The failure mode changes from 'confidently invented' to 'the retrieved passage "
            "did not contain the answer' - which is far easier to detect and fix.",
        ],
        "sections": [
            {
                "heading": "The pipeline",
                "code": (
                    "1. chunk   split documents into overlapping pieces\n"
                    "2. embed   turn each chunk into a vector\n"
                    "3. index   store the vectors\n"
                    "4. embed   turn the question into a vector the same way\n"
                    "5. rank    retrieve the top k chunks by cosine similarity\n"
                    "6. ground  put those chunks in the prompt as fenced context\n"
                    "7. answer  with a citation, or with an admission that the context is silent"
                ),
            },
            {
                "heading": "Chunking is the part people get wrong",
                "paragraphs": [
                    "Chunks that are too large dilute the embedding with irrelevant sentences, so retrieval "
                    "matches the wrong thing. Chunks that are too small lose the context needed to answer. "
                    "The compromise is a few hundred tokens per chunk with a small overlap, so a sentence "
                    "split across a boundary still appears whole somewhere.",
                ],
            },
            {
                "heading": "The grounding prompt",
                "code": (
                    "system: Answer using ONLY the context below. Cite the passage\n"
                    "        number. If the context does not contain the answer, say so.\n\n"
                    "user:   Context:\n"
                    "        [1] (gradient-descent#0) Gradient descent updates ...\n"
                    "        [2] (gradient-descent#1) If the learning rate is too large ...\n\n"
                    "        Question: What happens if the learning rate is too large?"
                ),
                "paragraphs": [
                    "The fence and the numbering are doing real work. Numbered passages give the model "
                    "something to cite, so the answer is checkable. Fencing the context as data - not "
                    "instructions - is also the first line of defence against prompt injection from retrieved "
                    "content.",
                ],
            },
            {
                "heading": "What to measure, and in what order",
                "bullets": [
                    "Retrieval hit rate first: was the correct chunk in the top k at all? If not, fix chunking "
                    "or the embedding - no prompt can save an answer whose source was never fetched.",
                    "Then generation: given the right context, did the answer contain the fact?",
                    "Then citation accuracy: does the cited passage actually support the claim?",
                    "Only then argue about prompt wording. Most 'RAG is bad' complaints are retrieval problems.",
                ],
            },
        ],
        "results": [
            "8 documents became 16 chunks (45 words, overlap 12)",
            "the right chunk ranked first at cosine 0.450",
            "the grounded answer quoted two passages and cited both",
            "(source: gradient-descent#0, source: gradient-descent#1)",
            "ungrounded answer invented 'section 4.2 of the handbook'",
        ],
        "demo": "python -m mlconcepts.llm.rag",
        "plot": None,
        "interview": [
            {
                "q": "Walk me through a RAG pipeline.",
                "a": "Chunk the documents with some overlap, embed each chunk with the same model you will "
                "use for queries, store the vectors in an index, embed the incoming question, retrieve the "
                "top k chunks by cosine similarity, put them into the prompt as fenced context with numbers, "
                "and ask the model to answer only from that context and cite the passage it used.",
            },
            {
                "q": "How do you choose the chunk size?",
                "a": "By evaluating retrieval, not by taste. Too large dilutes the embedding, too small loses "
                "context, so start around a few hundred tokens with 10-20% overlap and measure the retrieval "
                "hit rate on a golden set. Chunk on natural boundaries - headings, paragraphs - rather than "
                "mid-sentence.",
            },
            {
                "q": "Does RAG stop hallucination?",
                "a": "No. It changes the failure mode. The model can still misread the context or answer "
                "anyway when the context is silent - which is why the prompt must explicitly allow 'the "
                "context does not say', and why you measure citation accuracy separately from correctness.",
            },
            {
                "q": "RAG or fine-tuning?",
                "a": "RAG for knowledge that changes, that must be cited, or that is too large to fit in "
                "weights - which is most cases. Fine-tuning for behaviour: format, tone, a specialised task, "
                "or compressing a long prompt. They are not alternatives; you often fine-tune the model to "
                "be better at using retrieved context.",
            },
            {
                "q": "How do you make retrieval better?",
                "a": "In rough order: fix chunking, use a better embedding model, add a reranker over the top "
                "20 candidates, add keyword or BM25 search alongside the vectors for exact terms, and expand "
                "the query with the conversation history. Every step should be justified by the retrieval "
                "hit rate moving.",
            },
        ],
    },
    {
        "number": 14,
        "slug": "tools-agent",
        "part": 2,
        "title": "Tool calling and a small agent loop",
        "blurb": "A while loop with a budget: the model asks, you execute, the result goes back - including the errors.",
        "why": [
            "A model cannot check the weather, do arithmetic reliably, or read your database. Tool calling "
            "is how you let it ask you to do those things.",
            "That loop is an agent. It is not autonomous and it is not magic - it is a while loop with a "
            "step budget, and every framework is a variation on it.",
        ],
        "sections": [
            {
                "heading": "The loop",
                "code": (
                    "for step in range(1, max_steps + 1):\n"
                    "    response = llm.chat(messages, tools=TOOL_SCHEMAS)\n"
                    "    if not response.tool_calls:\n"
                    "        return response.text            # final answer\n"
                    "    messages.append(assistant_turn(response.tool_calls))\n"
                    "    for call in response.tool_calls:\n"
                    "        result = execute_tool(call)     # YOU run it, not the model\n"
                    "        messages.append(tool_message(call.name, result))\n"
                    "return 'could not finish within the step limit'"
                ),
            },
            {
                "heading": "The three things that bite you",
                "bullets": [
                    "THE MODEL NEVER RUNS CODE. You do, and every argument is untrusted input. This is why "
                    "the calculator here parses the expression with the ast module instead of calling eval() - "
                    "a calculator tool is the classic place someone tries 'import os and delete things'.",
                    "LOOPS MUST BE BOUNDED. A model can ask for the same tool forever. max_steps is not "
                    "optional, and neither is noticing when it is hit.",
                    "TOOLS FAIL. A bad expression, a timeout, a 404 - the error has to go back to the model "
                    "as a result, not crash the loop. A model told the tool failed can choose something else.",
                ],
            },
            {
                "heading": "The message history is the state",
                "paragraphs": [
                    "There is no hidden agent memory. The conversation list IS the state: the original user "
                    "request, each assistant turn that asked for a tool, and each tool result. That is why "
                    "the original request has to be kept around - after the first tool call, the latest "
                    "message is the tool's own output, and a loop that reads that instead will stop after one "
                    "step.",
                ],
            },
        ],
        "results": [
            "'2400 * 0.15 and the weather in Chennai' -> 2 tool calls, then an answer",
            "calculator returned 360.0; get_weather returned 'Chennai is 34C and humid'",
            "'5 apples plus 3 oranges' -> the tool errored and the loop survived",
        ],
        "demo": "python -m mlconcepts.llm.tools_agent",
        "plot": None,
        "interview": [
            {
                "q": "How does tool calling work?",
                "a": "You describe each tool with a name, a description and a JSON schema. The model replies "
                "with a structured tool call instead of prose. Your code executes the function, appends the "
                "result as a tool message, and calls the model again. It repeats until the model answers in "
                "prose or you hit your step limit.",
            },
            {
                "q": "What are the main risks of an agent loop?",
                "a": "Unbounded cost and latency, since each step is another model call. Tool arguments are "
                "untrusted input, so every tool must validate them. The model can loop on the same failing "
                "call, so you need a step limit and ideally loop detection. And side-effecting tools "
                "(payments, deletes) need human confirmation.",
            },
            {
                "q": "How do you make a tool the model actually uses?",
                "a": "Write the description for the model, not for a human - say when to use it and when not "
                "to. Keep the parameter names obvious, use enums instead of free strings, and return errors "
                "as readable text rather than stack traces. Then test it: if the model never calls it, the "
                "description is the problem.",
            },
            {
                "q": "How do you debug an agent that did something unexpected?",
                "a": "Log every step with its full arguments and result - the trace this demo prints is the "
                "minimum. Then replay it: the failure is almost always either a bad tool description, a "
                "tool returning something unhelpful, or the original request being lost from the context.",
            },
        ],
    },
    {
        "number": 15,
        "slug": "evaluation",
        "part": 2,
        "title": "Evaluation",
        "blurb": "A golden set, two variants, and the two scores that tell you what to fix - plus a regression check.",
        "why": [
            "Everything else in Part 2 is a component. This is the thing that tells you whether a change to "
            "any of them made things better or worse, and it is the piece most people skip.",
            "Without it, 'I improved the prompt' means 'I tried three examples and they looked nicer'. With "
            "it, it means a number - and a list of what broke.",
        ],
        "sections": [
            {
                "heading": "Score retrieval and generation separately",
                "paragraphs": [
                    "This is the single most useful habit in the whole of Part 2. On the golden set here, "
                    "retrieval found the right document in 6 of 6 cases while the generated answer "
                    "contained the fact in only 3.",
                    "'RAG is bad' is not a diagnosis. 'Retrieval 100%, generation 50%' tells you exactly "
                    "what to fix: the answer step, not the index.",
                ],
                "code": (
                    "retrieval hit rate  = is the right doc in the top k?      -> fix chunking / embedding\n"
                    "generation pass rate = did the answer contain the fact?  -> fix the prompt / the model\n"
                    "citation rate       = does the answer cite a source?     -> fix grounding instructions"
                ),
            },
            {
                "heading": "Start with the crudest scorer that works",
                "bullets": [
                    "Fact presence: does the answer contain the phrase we know is correct?",
                    "Exact match: rare outside structured extraction, but unbeatable when it applies.",
                    "Citation present: a regex, not a vibe. 'I was not given a source' must NOT count as a "
                    "citation - that is exactly the false pass a sloppy scorer produces.",
                    "Only then reach for a model-as-judge, and remember the judge is a model with its own "
                    "failure modes. Keep a human-checked sample.",
                ],
            },
            {
                "heading": "The regression rule",
                "paragraphs": [
                    "A change is an improvement only if the pass rate goes up AND nothing that used to pass "
                    "now fails. That second half is what people forget, and it is why you run the whole set "
                    "every time rather than the three cases you were thinking about.",
                ],
                "code": (
                    "before: 3/6 passing\n"
                    "after:  4/6 passing   <- but which three changed?\n\n"
                    "if any case went PASS -> fail, you have a regression, and the net\n"
                    "improvement may not be worth it."
                ),
            },
        ],
        "results": [
            "retrieval hit rate 6/6; generation 3/6",
            "no retrieval: 0/6 facts, 0% citations",
            "with RAG: 3/6 facts, 100% citations",
            "no regressions - every case that passed before still passes",
        ],
        "demo": "python -m mlconcepts.llm.evaluation",
        "plot": None,
        "interview": [
            {
                "q": "How do you evaluate an LLM feature?",
                "a": "Build a golden set of inputs with known correct answers, run the whole system over it, "
                "and score each output with the crudest measure that works - fact presence, exact match, "
                "citation rate. Compare variants on the same set and check for regressions. Keep the golden "
                "set growing from real failures.",
            },
            {
                "q": "What is LLM-as-judge and when is it acceptable?",
                "a": "Using a model to score outputs against a rubric. It is useful when correctness is "
                "subjective or when no exact answer exists, and it scales. It is not free: judges have "
                "position bias, prefer longer answers, and drift. Calibrate against human labels on a sample "
                "and report agreement.",
            },
            {
                "q": "How big should a golden set be?",
                "a": "Start with 20-50 cases covering the main paths and the known failure modes; that is "
                "enough to catch regressions immediately. Grow it from real user failures. A small set you "
                "actually run beats a large one you never do.",
            },
            {
                "q": "How do you know your RAG is failing because of retrieval?",
                "a": "Measure retrieval on its own: for each golden question, is the document containing the "
                "answer in the top k? If that number is low, no prompt change will help. If it is high and "
                "the answers are still wrong, the problem is generation.",
            },
        ],
    },
]

from dotenv import load_dotenv
import os
import string
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# Concepts Question 1
"""
    Scenario A:
    Best Approach: RAG (Retrieval-Augmented Generation)
    The legal team has hundreds of PDFs that change every quarter, so RAG is the best choice.
    A RAG system can split the documents into chunks, create embeddings, store them in a vector store/index, and retrieve the most relevant sections before generating an answer. This helps reduce hallucination because the model answers using real company documents.

    Scenario B:
    Best Approach: Fine-Tuning
    The startup wants a very specific writing style that is not common online.
    Fine-tuning is best because the model can learn patterns from the 3,000 examples and consistently generate product copy in the company’s unique brand voice.

    Scenario C:
    Best Approach: Prompt Engineering
    The analyst only needs to ask questions about one short two-page report.
    Prompt engineering is enough because the document can simply be pasted into the prompt as context, without needing retrieval, embeddings, or fine-tuning.
"""

# Concepts Question 2

"""
    A confidently wrong answer is more harmful because people are more likely to trust and act on it.
    When an AI says “I am not sure,” the user knows they should verify the information, but a confident hallucination can make false information sound accurate and reliable.
    For example, a medical AI assistant could confidently give the wrong medication dosage to a patient. If the patient trusts the answer without checking a doctor or pharmacist, it could cause serious health problems.
    The tone of the response matters because humans naturally trust answers that sound clear, detailed, and confident. Even if the content is incorrect, a confident tone can make the hallucination seem believable and reduce the chance that the user will question it.
"""

# Concepts Question 3
"""
    Correct Order of a RAG Pipeline:

    1. "Extract text from source documents"
    - The system reads and extracts text from PDFs, files, websites, or databases.

    2. "Split text into chunks"
    - Large documents are divided into smaller chunks so retrieval is faster and more accurate.

    3. "Convert text chunks into embeddings"
    - Each chunk is converted into an embedding (vector representation) and stored in a vector store/index.

    4. "Receive the user's query"
    - The user asks a question or sends a prompt to the system.

    5. "Embed the user's query"
    - The query is converted into an embedding so it can be compared with stored chunk embeddings.

    6. "Retrieve the most relevant chunks"
    - The system uses cosine similarity to find chunks with embeddings most similar to the user’s query.

    7. "Inject retrieved chunks into the prompt"
    - The retrieved chunks are added to the prompt as context for the LLM.

    8. "Generate a response from the LLM"
    - The LLM generates an answer grounded in the retrieved information, helping reduce hallucination.
"""

# Keyword RAG

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
# Keyword Question 1
query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

selected_document = simple_keyword_retrieval(
    query,
    documents,
    verbose=True

)

print("Selected document:", selected_document)

"""
    The selected document should be "hours.txt" because it contains keywords related to the query such as "hours" and "weekends".
    Keyword retrieval works by matching important words between the query and the documents. Since "hours.txt" discusses weekend opening and closing times, it is the most relevant match.
"""


query = "Do you have anything without caffeine?"
selected_document = simple_keyword_retrieval(
    query,
    documents,
    verbose=True
)

#Keyword Question 2
query = "Do you have anything without caffeine?"
selected_document = simple_keyword_retrieval(
    query,
    documents,
    verbose=True
)

print("Selected document:", selected_document)
"""
    The selected document will most likely be "menu.txt" because it contains words related to drinks and menu items.
    However, keyword RAG may not fully get this right because the document never directly mentions "without caffeine" or "decaf." The retrieval system is only matching keywords, not understanding meaning or intent.
    A semantic retrieval system using embeddings would do better here because it can understand that "without caffeine" is conceptually related to drink options and milk alternatives, even if the exact words are not present. Semantic retrieval compares embeddings using cosine similarity instead of relying only on keyword matches.
"""

#Keyword Question 3
"""
    Prediction:
    I predict the function will select "loyalty.txt" because the query contains words like "sign up" and "rewards," which are conceptually related to a loyalty program.

    Reasoning:
    The "loyalty.txt" document talks about earning points and rewards, so it seems like the most relevant document for this query.
"""

query = "How do I sign up for rewards?"

selected_document = simple_keyword_retrieval(
    query,
    documents,
    verbose=True

)

print("Selected document:", selected_document)
"""
    The prediction was partially correct because "loyalty.txt" is the most relevant document conceptually.
    However, the keyword retrieval system may not perform perfectly because it only checks token overlap. The query uses the word "rewards," but the document uses "loyalty program" and "points" instead. Since keyword retrieval does not understand semantic meaning, it can miss relationships between similar ideas.
    A semantic retrieval system using embeddings would work better because embeddings capture meaning, not just exact keyword matches.
"""

# Semantic RAG Concepts
# Semantic Question 1

"""
    1) What is a vector embedding?
    A vector embedding is a way of turning text into a list of numbers so that the meaning of the text is represented in a mathematical form. Texts with similar meanings end up with similar number patterns.

    2) Cosine similarity question:
    The chunk with a cosine similarity of 0.85 is more relevant because it is closer to 1, which means its meaning is much more similar to the query. The 0.30 score shows weak similarity, meaning the chunk is likely not closely related to the query.

    3) Why semantic search works without exact words:
    Semantic search works because embeddings capture meaning, not just exact words. Even if two texts use different vocabulary, they can still be close in embedding space if they talk about the same idea. This is where cosine similarity helps measure meaning-based closeness instead of keyword overlap.
"""

# Semantic Question 2
"""
    | Feature                | Keyword RAG                       | Semantic RAG                          |
    |------------------------|-----------------------------------|---------------------------------------|
    | What is compared?      | Exact word overlap                | Vector embeddings (meaning similarity)|
    | What is retrieved?     | Full document                     | Most relevant chunks                  |
    | Can it handle synonyms?| No                                | Yes                                   |
    | Storage format         | Plain text dictionary             | Vector store / index (embeddings)     |
    | Relevance score        | Number of overlapping keywords    | Cosine similarity between embeddings  |
"""

# LlamaIndex Question 1

# Load BrightLeaf Solar PDFs (adjust path if needed)

documents = SimpleDirectoryReader(
    "assignments_06/resources/brightleaf_pdfs"
).load_data()

index = VectorStoreIndex.from_documents(documents)

# Create query engine with top-k retrieval
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print("\n" + "="*80)
    print("QUESTION:", q)

    response = query_engine.query(q)
    print("\nANSWER:\n", response)
    print("\nTOP 3 RETRIEVED NODES:\n")

    for i, node in enumerate(response.source_nodes):
        score = node.score
        text_preview = node.node.text[:150].replace("\n", " ")
        print(f"[{i+1}] Similarity Score: {score}")
        print(f"    Chunk Preview: {text_preview}\n")

"""
OBSERVATIONS (Write-up)
    Query 1: "What employee benefits does BrightLeaf offer?"
    - The retrieved chunks should mostly come from HR / benefits-related sections (healthcare, PTO, retirement, etc.).
    - The model response is usually confident and specific because the documents likely contain direct policy statements.
    - I expect low hallucination risk here since the chunks are relevant and well-defined.
    - Nothing unexpected should appear if retrieval works well (like security or IT sections).

    Query 2: "What are BrightLeaf's security policies?"
    - The retrieved chunks should come from IT/security/compliance sections (access control, data protection, password rules).
    - If any HR or unrelated policy chunks appear, that would be slightly unexpected but still possible due to shared terms like "policy".
    - The model response may include phrases like "based on the provided context" because LlamaIndex typically grounds answers in retrieved nodes.
    - The tone is usually cautious (slight hedging) when the context is incomplete or partially relevant.

    Overall:
    - This is a semantic RAG system using embeddings + cosine similarity, so retrieval is meaning-based rather than keyword-based.
"""

# LlamaIndex Question 2
index = VectorStoreIndex.from_documents(documents)
query_engine_k1 = index.as_query_engine(similarity_top_k=1)
query_engine_k5 = index.as_query_engine(similarity_top_k=5)
query = "What employee benefits does BrightLeaf offer?"

# Run with k = 1
print("\n" + "="*80)
print("RUN: similarity_top_k = 1")
print("QUESTION:", query)
response_k1 = query_engine_k1.query(query)
print("\nANSWER:\n", response_k1)
print("\nTOP NODE:")

for node in response_k1.source_nodes:
    print("Score:", node.score)
    print("Chunk:", node.node.text[:150], "\n")

# Run with k = 5
print("\n" + "="*80)
print("RUN: similarity_top_k = 5")
print("QUESTION:", query)
response_k5 = query_engine_k5.query(query)
print("\nANSWER:\n", response_k5)
print("\nTOP 5 NODES:")

for i, node in enumerate(response_k5.source_nodes):
    print(f"[{i+1}] Score:", node.score)
    print("Chunk:", node.node.text[:150], "\n")

"""
OBSERVATION COMMENT:
    - With similarity_top_k=1:
    The model only sees the single most relevant chunk. The answer is usually very focused but may miss supporting details (for example, it might mention only healthcare benefits but not PTO or retirement benefits).

    - With similarity_top_k=5:
    The model receives more context chunks, so the answer becomes more complete and includes additional details from multiple sections of the policy document. However, some lower-ranked chunks may be less relevant and slightly dilute focus.

    - Conclusion:
    More retrieved context is not always better. While increasing k improves coverage, it can also introduce noise (less relevant chunks). The best value depends on document quality and how well retrieval ranks relevance. In RAG systems, balancing precision (k=1) and coverage (k>1) is important to reduce both hallucination and irrelevant information.
"""

# LlamaIndex Question 3
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

# A deliberately vague / hard query

query = "How does BrightLeaf compare to other companies in terms of work culture and innovation?"
print("\n" + "="*80)
print("QUESTION:", query)
response = query_engine.query(query)
print("\nANSWER:\n", response)
print("\nTOP RETRIEVED CHUNKS:\n")

for i, node in enumerate(response.source_nodes):
    print(f"[{i+1}] Score:", node.score)
    print("Chunk preview:", node.node.text[:200].replace("\n", " "), "\n")

"""
OBSERVATION COMMENT:
    - What I expected:
    I expected the system to struggle because the query is comparative ("compare to other companies") and requires external knowledge that is not likely inside the BrightLeaf PDFs. I also expected mixed or weakly relevant chunks.

    - What actually happened:
    The retriever still returned chunks from BrightLeaf documents that contain general HR, mission, or culture-related language. The model likely produced a best-effort answer based only on internal context, even though no real comparison data exists.

    - Tone behavior:
    The response would likely sound somewhat confident but may include hedging phrases like "based on the provided documents" or "the context suggests," because the model is forced to answer using limited retrieved information.

    - What could improve the system:
    1. Add a "missing information detection" step to reduce hallucination when retrieval scores are low.
    2. Use a hybrid retrieval system (keyword + semantic) to catch broader concepts like "culture" and "innovation."
    3. Add a reranker model to improve chunk quality before generation.
    4. Allow the system to say "not enough information in documents" when cosine similarity scores are weak.
    This would reduce hallucination and improve trustworthiness in real-world RAG systems.
"""

# LlamaIndex Question 4

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

# LLM judge setup (gpt-4o-mini)
llm = OpenAI(model="gpt-4o-mini")
faithfulness_eval = FaithfulnessEvaluator(llm=llm)
relevancy_eval = RelevancyEvaluator(llm=llm)

# Query 1 (in-domain)
q1 = "What employee benefits does BrightLeaf offer?"
response1 = query_engine.query(q1)
faith1 = faithfulness_eval.evaluate_response(response=response1)
rel1 = relevancy_eval.evaluate_response(query=q1, response=response1)

print("\n==================== QUERY 1 ====================")
print("Q:", q1)
print("Faithfulness Score:", faith1.score)
print("Relevancy Score:", rel1.score)

# Query 2 (out-of-domain / expected weak retrieval)
q2 = "What is BrightLeaf's stock price history and market performance?"
response2 = query_engine.query(q2)
faith2 = faithfulness_eval.evaluate_response(response=response2)
rel2 = relevancy_eval.evaluate_response(query=q2, response=response2)

print("\n==================== QUERY 2 ====================")
print("Q:", q2)
print("Faithfulness Score:", faith2.score)
print("Relevancy Score:", rel2.score)

"""
OBSERVATION COMMENT:
    1) What does a faithfulness score of 1.0 mean?
    A faithfulness score of 1.0 means the model's answer is fully grounded in the retrieved context with no hallucinated information. A score of 0.0 would mean the response is not supported by the retrieved documents at all and likely contains hallucinated or invented content.

    2) What does relevancy measure, and how is it different?
    Relevancy measures how well the retrieved context and final answer actually address the user’s query. Faithfulness checks whether the answer is supported by the context, while relevancy checks whether the answer is useful and on-topic for the question.

    3) Did the scores change between queries?
    Yes. The in-domain query (employee benefits) should have higher faithfulness and relevancy because the answer is directly supported by the BrightLeaf documents. The out-of-domain query (stock price history) should have lower relevancy because that information likely does not exist in the documents, and possibly lower faithfulness if the model attempts to guess.

    4) What is LLM-as-a-judge?
    LLM-as-a-judge means using another LLM (like gpt-4o-mini) to evaluate the quality of a model's response. It is used in RAG evaluation because there is often no single “correct answer,” so traditional accuracy metrics don’t work well. Instead, the judge LLM checks reasoning, grounding, and relevance based on context.
"""
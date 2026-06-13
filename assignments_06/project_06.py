from pathlib import Path
from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# Step 1: Setup

# Load environment variables from .env
load_dotenv()

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Confirm API key loaded
if api_key:
    print("API key loaded successfully.")
else:
    print("API key not found.")

# Path to Groundwork documents
docs_dir = Path("assignments_06/resources/groundwork_docs")

# Verify the directory exists before continuing
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(" Groundwork document directory found.")


# Step 2: Load the Documents

# Load all documents
documents = SimpleDirectoryReader(docs_dir).load_data()

# Print number of loaded documents
print(f"\nTotal documents loaded: {len(documents)}")

# Print each document file name
print("\n Loaded document files:")

for doc in documents:
    file_name = doc.metadata.get("file_name", "Unknown file")
    print(f"- {file_name}")

# Step 3: Build the Index and Query Engine

# Create vector index from documents
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=3)
print("\n Index built successfully. Ready to answer questions.")
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",

]

# Loop through questions
for question in questions:
    print("\n" + "=" * 60)
    print(f" Question: {question}")

    # Query the RAG system
    response = query_engine.query(question)

    # Print model answer
    print(f"\n Answer:\n{response}")

    # Get top retrieved source node
    top_node = response.source_nodes[0]

    # Extract metadata
    file_name = top_node.node.metadata.get("file_name", "Unknown file")
    similarity_score = top_node.score
    chunk_text = top_node.node.text[:200]

    # Print retrieval information
    print("\n Top Retrieved Source:")
    print(f"File Name: {file_name}")
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Chunk Preview: {chunk_text}")

# Reflection:
# The assistant sounded mostly confident and accurate because it retrieved
# information directly from the Groundwork documents before generating answers.
# The retrieval step helped reduce hallucinations by grounding the responses
# in real document chunks.
# One interesting observation was that questions about catering or wholesale
# may produce weaker answers if that information does not exist clearly in
# the documents. This shows an important RAG concept:
# retrieval quality directly affects answer quality.
# The similarity score helps measure how relevant a retrieved chunk is to
# the user query, while the chunk preview makes debugging easier by showing
# exactly what text the model used as context.


# Step 5: Find a Failure

#  This question is intentionally difficult because
# the documents do not mention online ordering,
# delivery apps, or shipping information.

failure_question = (
    "Do you deliver coffee beans nationwide and offer online subscriptions?"

)

print("\n" + "=" * 60)
print(" FAILURE TEST")
print(f"\n Question: {failure_question}")
# Query the assistant
failure_response = query_engine.query(failure_question)
# Print model response
print(f"\n Full Response:\n{failure_response}")
# Print all retrieved source nodes
print("\n Retrieved Source Nodes:")

for i, node in enumerate(failure_response.source_nodes):
    file_name = node.node.metadata.get("file_name", "Unknown file")
    similarity_score = node.score
    chunk_preview = node.node.text[:200]
    print(f"\n--- Source Node {i + 1} ---")
    print(f"File Name: {file_name}")
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Chunk Preview: {chunk_preview}")

# Reflection:
# I asked whether Groundwork delivers coffee beans nationwide
# and offers online subscriptions because this information does
# not appear anywhere in the provided documents.
# I expected this to be difficult because the retriever would
# still try to find the "closest" matching chunks even though
# the actual answer does not exist in the dataset.

# What went wrong:
# The retrieval system likely returned partially related chunks
# about coffee sourcing, locations, or customer services, but
# none of them directly answered the question. This demonstrates
# a common RAG failure mode called retrieval mismatch:
# the vector search retrieves semantically similar content,
# not necessarily correct content.
# The model may still sound confident even when the answer is
# unsupported by the documents. This is an example of AI
# hallucination. Even with retrieval augmentation, the LLM can
# still generate guesses if the retrieved context is weak or
# incomplete.
# One important observation is that confidence does not equal
# correctness. The assistant's tone may remain fluent and
# professional even when it lacks evidence. This suggests that
# users should not blindly trust AI-generated answers without
# source verification.

# To improve the system, I would:
# 1. Add a similarity score threshold so low-quality retrievals
#    are rejected instead of passed to the model.
# 2. Add a fallback response like:
#    "I could not find that information in the documents."
# 3. Improve chunking and metadata handling so retrieval becomes
#    more precise.
# 4. Use reranking models or hybrid retrieval
#    (keyword + vector search) to improve retrieval accuracy.
# 5. Add citation requirements so answers must reference retrieved
#    evidence before responding confidently.




# Step 6: Final Reflection


# Reflection:
# In the lesson, building semantic RAG manually required many

# separate steps:
# - reading files
# - chunking text
# - generating embeddings
# - storing vectors
# - performing cosine similarity search
# - retrieving chunks
# - passing context to the LLM

# The manual implementation took a large amount of code because
# every retrieval step had to be built individually.
# In contrast, the equivalent LlamaIndex implementation in this
# project only required a few lines:
#     index = VectorStoreIndex.from_documents(documents)
#     query_engine = index.as_query_engine(similarity_top_k=3)

# This demonstrates the value of frameworks like

# Frameworks abstract away repetitive infrastructure code and
# allow developers to focus on application logic instead of
# rebuilding retrieval pipelines from scratch.
# The framework still performs the same important RAG steps:
# chunking, embedding generation, vector indexing, retrieval,
# and context injection into the LLM prompt — but with far less
# code and much faster development time.
# A valuable real-world use case for RAG would be in healthcare.
# A hospital or clinic could build an assistant that answers
# questions from internal medical policies, insurance documents,
# treatment guidelines, and patient instructions.

# For example:
# - nurses could quickly retrieve protocol information,
# - support staff could answer insurance questions,
# - doctors could search internal treatment documentation.
# This would save time and improve access to institutional
# knowledge without employees manually searching through hundreds
# of documents.
# One important failure mode that RAG cannot fully prevent is
# reasoning failure or incorrect synthesis.
# Even if retrieval works correctly and the right chunks are
# provided, the LLM can still:
# - misunderstand the context,
# - combine facts incorrectly,

# - overgeneralize,
# - or generate unsupported conclusions.
# In other words, correct retrieval does not guarantee correct
# reasoning. The generation step can still hallucinate or make
# logical mistakes even when grounded in accurate source text.
# This highlights an important AI engineering principle:
# retrieval improves reliability, but it does not eliminate the
# need for verification, evaluation, and human oversight.
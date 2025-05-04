import os
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# Set your OpenAI API Key (use dotenv or Streamlit secrets in prod)
os.environ["OPENAI_API_KEY"] = ""  # Replace with your real key

# ---- Streamlit UI setup ----
st.set_page_config(page_title="Gully Bots - Retail AI Chatbot", page_icon="🤖")
st.title("🤖 Gully Bots – Retail Product Issue Assistant")

# ---- Load and Embed Data ----
@st.cache_resource
@st.cache_resource
def load_agent():
    # Load dataset once
    df = pd.read_csv("./final_combined_data.csv").sample(n=10000)
    product_counts = df['Product Purchased'].value_counts().reset_index()
    # Rename columns for clarity
    product_counts.columns = ['Product Name', 'Ticket Count']
    # Sort (optional)
    product_counts = product_counts.sort_values(by='Ticket Count', ascending=False)
    # Show the top 10 products
    print(product_counts.head(10))

    def combine_fields(row):
        return (
            f"Ticket ID: {row['Ticket ID']}\n"
            f"Product: {row['Product Purchased']}\n"
            f"Subject: {row['Ticket Subject']}\n"
            f"Description: {row['Ticket Description']}\n"
            f"Ticket Type: {row['Ticket Type']}\n"
            f"Priority: {row['Ticket Priority']}\n"
            f"Status: {row['Ticket Status']}\n"
            f"Customer Rating: {row.get('Customer Satisfaction Rating', 'N/A')}\n"
        )

    df['combined_text'] = df.apply(combine_fields, axis=1)
    documents = [Document(page_content=text) for text in df['combined_text']]

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    # TOOL 1: Product Issue Summary
    def issue_summary_tool(query):
        docs = vectorstore.similarity_search(query)
        if not docs:
            return "Sorry, I couldn't find any product issues related to that."
        prompt = (
            f"Please provide a detailed and structured summary based on support tickets for: {query}.\n"
            f"Include:\n"
            f"- Common issues or complaints\n"
            f"- Frequent return reasons\n"
            f"- Customer satisfaction insights\n"
            f"- Any supplier response patterns or escalation trends\n"
        )
        return qa_chain.run({"input_documents": docs, "question": prompt})

    # TOOL 2: Support Efficiency Tool
    def support_efficiency_tool(query: str) -> str:
        import numpy as np

        def parse_time(value):
            if "minutes" in str(value):
                return float(str(value).split()[0]) / 60
            elif "hours" in str(value):
                return float(str(value).split()[0])
            else:
                return np.nan

        df["response_time_hrs"] = df["First Response Time"].apply(parse_time)
        df["resolution_time_hrs"] = df["Time to Resolution"].apply(parse_time)

        if "channel" in query.lower():
            grouped = df.groupby("Ticket Channel")[["response_time_hrs", "resolution_time_hrs"]].mean().round(2)
            return grouped.to_string()
        elif "priority" in query.lower():
            grouped = df.groupby("Ticket Priority")[["response_time_hrs", "resolution_time_hrs"]].mean().round(2)
            return grouped.to_string()
        elif "product" in query.lower():
            top_products = df["Product Purchased"].value_counts().head(5).index.tolist()
            filtered = df[df["Product Purchased"].isin(top_products)]
            grouped = filtered.groupby("Product Purchased")[["response_time_hrs", "resolution_time_hrs"]].mean().round(2)
            return grouped.to_string()
        else:
            overall_avg = df[["response_time_hrs", "resolution_time_hrs"]].mean().round(2)
            return (
                f"Average First Response Time: {overall_avg['response_time_hrs']} hrs\n"
                f"Average Resolution Time: {overall_avg['resolution_time_hrs']} hrs"
            )

    # TOOL 3 : Return Policy Tool
    def return_policy_tool(query: str) -> str:
        # Basic keyword-based match to extract product name
        matched_products = df[df["Product Purchased"].str.contains(query, case=False, na=False)]

        if matched_products.empty:
            return "No matching product found in the return policy records."

        # Just use the first matching entry for summarization
        row = matched_products.iloc[0]

        product = row["Product Purchased"]
        days = row["return_window_days"]
        refund = "allowed" if row["refund_allowed"] == "Yes" else "not allowed"
        replacement = "allowed" if row["replacement_allowed"] == "Yes" else "not allowed"
        restock = "applies" if row["restocking_fee"] == "Yes" else "does not apply"
        policy_text = row["return_policy_description"]

        return (
            f"**Return Policy for {product}**:\n"
            f"- Return Window: {days} days\n"
            f"- Refund: {refund}\n"
            f"- Replacement: {replacement}\n"
            f"- Restocking Fee: {restock}\n"
            f"- Policy Notes: {policy_text}"
        )

    # Define LangChain tools
    tools = [
        Tool(
                name="ProductIssueSummary",
                func=issue_summary_tool,
                description="Summarizes complaints, returns, ratings, and product issues."
        ),
        Tool(
                name="SupportEfficiencyAnalyzer",
                func=support_efficiency_tool,
                description="Analyzes first response and resolution time by product, channel, or priority"
        ),
        Tool(
            name="ReturnPolicyAnalyzer",
            func=return_policy_tool,
            description="Provides return/refund/replacement policies for a given product"
        )
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    return agent

# ---- Initialize Session and Agent ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent = load_agent()

# ---- Chat Input ----
user_input = st.chat_input("Ask about product complaints, returns, or issues...")

if user_input:
    with st.spinner("Thinking..."):
        response = agent.run(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

# ---- Display Chat History ----
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)


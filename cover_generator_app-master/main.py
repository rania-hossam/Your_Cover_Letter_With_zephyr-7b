from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

import os
from pdf_reader import load_pdf
from read_job_posting import extract_text_from_url
from splitter import split_text_documents
# import dependencies
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from scripts.LLM import load_quantized_model,initialize_tokenizer
model_name = "anakin87/zephyr-7b-alpha-sharded"

# load model
model = load_quantized_model(model_name)
# initialize tokenizer
tokenizer = initialize_tokenizer(model_name)
# specify stop token ids
stop_token_ids = [0]




def get_cover_letter(url, pdf):

    pdf_doc = load_pdf(pdf)
    job_post = extract_text_from_url(url)

    pdf_doc.extend(job_post)
    documents = split_text_documents(pdf_doc)

    # specify embedding model (using huggingface sentence transformer)
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

    vectordb = Chroma.from_documents(documents, embedding=embeddings)
    retriever = vectordb.as_retriever()

    pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2048,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=pipeline)
    pdf_qa = RetrievalQA.from_llm(
            llm=llm,
            retriever=retriever,
            get_chat_history=lambda h: h,
        )


    query = 'Write a cover letter for given CV and Job posting in a conversational style and fill out the writers name in the end using cv'

    result = pdf_qa.run(query)

    return result
import { ChatOpenAI } from "@langchain/openai"
import {ChatPromptTemplate} from '@langchain/core/prompts'
import * as dotenv from "dotenv";

import {createStuffDocumentsChain} from 'langchain/chains/combine_documents'
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio"
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter'
import { OpenAIEmbeddings } from "@langchain/openai"
import {MemoryVectorStore} from 'langchain/vectorstores/memory'
import {createRetrievalChain} from 'langchain/chains/retrieval'

dotenv.config();

const model = new ChatOpenAI({
    model: "gpt-3.5-turbo",
    temperature: 0.7,
    openAIApiKey: process.env.OPENAI_API_KEY,
})

const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the user's question. 
    Context: {context}
    Question: {input}
`)

// const chain = prompt.pipe(model)
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
})

// loader -> load data from web page
const loader = new CheerioWebBaseLoader("https://js.langchain.com/v0.2/docs/concepts/")
const docs =  await loader.load()

//split
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20
})

const splitDocs = await splitter.splitDocuments(docs)

//embeddings
const embedding = new OpenAIEmbeddings()

//vectorstore
const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embedding
)

//retrieve data
const retriever = vectorStore.asRetriever({
    k: 3
})

//retrieval chain
const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever
})


const response = await retrievalChain.invoke({
    // input: "What is the concept of LCEL?",
    input: "What is the LCEL?",
    context: splitDocs
})

console.log(response);





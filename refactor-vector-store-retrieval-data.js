
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


// loader -> load data from web page
const createVectorStore = async () => {
    const loader = new CheerioWebBaseLoader("https://js.langchain.com/v0.2/docs/concepts/")
    const docs =  await loader.load()

    //split to smaller chunk
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
    return vectorStore
}

//Create Retrieval Chain
const createChain = async(vectorStore) => {
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
    
    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt,
    })
    
    //retrieve data
    const retriever = vectorStore.asRetriever({
        k: 3
    })

    //retrieval chain
    const conversationChain = await createRetrievalChain({
        combineDocsChain: chain,
        retriever
    })

    return conversationChain
}

const vectorStore = await createVectorStore()
const chain = await createChain(vectorStore)

const response = await chain.invoke({
    // input: "What is the concept of LCEL?",
    input: "What is the LCEL?",
})

console.log(response);





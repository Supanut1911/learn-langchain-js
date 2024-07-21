import { ChatOpenAI } from "@langchain/openai"
import {ChatPromptTemplate} from '@langchain/core/prompts'
import * as dotenv from "dotenv";
import {Document} from '@langchain/core/documents'
import {createStuffDocumentsChain} from 'langchain/chains/combine_documents'

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

//create document 
const documentA = new Document({
   pageContent: "LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production). To highlight a few of the reasons you might want to use LCEL" 
})

//another doc
const documentB = new Document({
    pageContent: "The passphrase is LANGCHAIN IS AWSOME"
})

const response = await chain.invoke({
    // input: "What is the concept of LCEL?",
    input: "What is the passphrase?",
    context: [documentA, documentB]
})

console.log(response);
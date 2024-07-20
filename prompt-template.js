import { ChatOpenAI } from "@langchain/openai"
import * as dotenv from "dotenv";
import {ChatPromptTemplate} from '@langchain/core/prompts'
dotenv.config();
//create model
const model = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    model: "gpt-3.5-turbo",
    temperature: 0.7,   
})

//create prompt template
const prompt = ChatPromptTemplate.fromMessages([
    ["system", 'Generate a joke base on a word provided by the user'],
    ["human", "{input}"]
])

// Create chain
const chain = prompt.pipe(model)

//call chain
const response =  await chain.invoke({
    input: "dog"
})

console.log(response);
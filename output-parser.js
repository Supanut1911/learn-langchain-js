import { ChatOpenAI } from "@langchain/openai"
import * as dotenv from "dotenv";
import {ChatPromptTemplate} from '@langchain/core/prompts'
import { StringOutputParser, CommaSeparatedListOutputParser } from "@langchain/core/output_parsers";
import { StructuredOutputParser } from "langchain/output_parsers"

import {z} from 'zod'

dotenv.config();
//create model
const model = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    model: "gpt-3.5-turbo",
    temperature: 0.7,   
})

const callStringOutputParser = async () => {
    //create prompt template
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", 'Generate a joke base on a word provided by the user'],
        ["human", "{input}"]
    ])

    //create parser
    const parser = new StringOutputParser();

    // Create chain
    const chain = prompt.pipe(model).pipe(parser)

    //call chain
    return await chain.invoke({
        input: "dog"
    })
}

const callListOutputParser = async () => {
    const prompt = ChatPromptTemplate.fromTemplate(
        `
        Provide 5 synonyms, separated by commas, for the following word {word}"
        `
    )

    const outputParser = new CommaSeparatedListOutputParser();

    const chain = prompt.pipe(model).pipe(outputParser)

    return await chain.invoke({
        word: "happy"
    })
}

// structure output parser
const callStructureOutputParser = async () => {
    const prompt = ChatPromptTemplate.fromTemplate(
        `Extract info from the following phrase. 
        Formatting instruction: {format_instruction}
        phrase: {phrase}`
    )

    const outputParser =  StructuredOutputParser.fromNamesAndDescriptions({
        name: "the name of the person",
        age: "the age of the person"
    })

    const chain = prompt.pipe(model).pipe(outputParser)

    return  await chain.invoke({
        phrase: "Max is 30 years old",
        format_instruction: outputParser.getFormatInstructions()
    })
}

// use zod output parser
const callZodOutputParser = async () => {
    const prompt = ChatPromptTemplate.fromTemplate(
        `Extract info from the following phrase. 
        Formatting instruction: {format_instruction}
        phrase: {phrase}`
    )
    
    const outputParser =  StructuredOutputParser.fromZodSchema(
        z.object({
            recipe: z.string().describe("name of recipe"),
            ingredients: z.array(z.string()).describe("list of ingredients")
        })
    )

    const chain = prompt.pipe(model).pipe(outputParser)

    return await chain.invoke({
        phrase: "ingredients for a Spaghetti Carbonara are eggs, bacon, cheese, pasta",
        format_instruction: outputParser.getFormatInstructions()
    })

}

// const response = await callStringOutputParser()
// const response = await callListOutputParser()
// const response = await callStructureOutputParser()
const response = await callZodOutputParser()
console.log(response);
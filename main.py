from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import os

load_dotenv()



def main():
    llm = GoogleGenerativeAI(model="gemini-3.5-flash")
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and former public official known for his leadership of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025; as of May 2026, Forbes estimates his net worth to be US$828 billion.
Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if he meets specific goals.
Musk is a supporter of global far-right politics, figures, and political parties. He was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated as president in January 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). Shortly before a public feud with Trump, Musk left the Trump administration in May 2025 and returned to managing his companies.
    """
    summary_template  = """
    given the information {information} about a person I want you to create:
    1. A Short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],template=summary_template,
    )
    #llm  = OllamaLLM(model="gemini-3.5-flash")
    llm  = OllamaLLM(model="gemma4:31b-cloud",temperature=0.9)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information" : information})
    llm.model.title()
    print(response)




if __name__ == "__main__":
    main()


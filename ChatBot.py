pip install langchain PyPDF2
import streamlit
import langchain
import streamlit as st
import PyPDF2
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
mistral = ChatMistralAI(api_key="Iz6iwDl4562yE22mBSduF039S8D08eNH")
def chargement_statistique(chemin_doc):
    with open(chemin_doc, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)  
        pages = len(pdf_reader.pages)
        text = ""
        for page in range(pages):
            text += pdf_reader.pages[page].extract_text()  
        return text
statistique_chemin = "J:\COURS DE STATISTIQUE DESCRIPTIVE_052231_045453.pdf"
statistique_document = chargement_statistique(statistique_chemin)
template = f"""
Tu es un assistant pédagogique spécialisé en statistique descriptive qui aide à résoudre des problèmes mais aussi à examiner les étudiants 
des petits test et exercices
Ton rôle est d'aider les étudiants de Pré-Math et Pré-Informatique à la faculté des sciences à l'université de Lubumbashi au Haut-Katanga en RDC
Tu utiliseras les notes de cours de statistique {{statistique_document}}
La question de l'étudiant {{question}}
Ta réponse :
"""
prompt = PromptTemplate(template=template, input_variables=["statistique_document", "question"])
chain = LLMChain(llm=mistral, prompt=prompt)

st.set_page_config(page_title="Assistant Statistique", page_icon="📙", layout="wide")
with st.sidebar:
    st.image("https://www.google.com/imgres?q=sidebar%20icon&imgurl=https%3A%2F%2Ficons.veryicon.com%2Fpng%2Fo%2Fmiscellaneous%2Fbig-data-regular-monochrome-icon%2Fsidebar-4.png&imgrefurl=https%3A%2F%2Fwww.veryicon.com%2Ficons%2Fmiscellaneous%2Fbig-data-regular-monochrome-icon%2Fsidebar-4.html&docid=8ik6vigXADHhsM&tbnid=ReG3oA-owSI9yM&vet=12ahUKEwi67Kan4KGHAxX3QkEAHU1BBxUQM3oECBcQAA..i&w=512&h=512&hcb=2&ved=2ahUKEwi67Kan4KGHAxX3QkEAHU1BBxUQM3oECBcQAA")
    st.title("A propos")
    st.info("Cet assistant utilise l'IA pour t'aider à maîtriser la statistique descriptive")
st.title("📙 Assistant en statistique descriptive")
st.write("Bienvenu ! Je suis là pour vous aider")
question = st.text_input("💡 Poser votre question sur la statistique descriptive : ", 
                        placeholder="Par exemple : Qu'est-ce que la moyenne géométrique ?")
if question :
    with st.spinner("Réflexion en cours..."):
        reponse = chain.run({"statistique_document":statistique_document, "question":question})
    st.success("Voici ma réponse :")
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding:20px; border-radius:10px;">
    {reponse}
    </div>
    """, unsafe_allow_html=True)


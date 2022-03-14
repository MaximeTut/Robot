from pyexpat import model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from streamlit_lottie import st_lottie
import json
import pandas as pd
import requests
import torch
import tensorflow as tf
import streamlit as st
from streamlit_option_menu import option_menu

logo = "https://www.google.com/url?sa=i&url=https%3A%2F%2Ffr.depositphotos.com%2Fvector-images%2Frobot-logo.html&psig=AOvVaw14rAtmwJQVSpRFXFY6us7z&ust=1647274982461000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCPjhzdO_w_YCFQAAAAAdAAAAABAD"
st.set_page_config(page_icon = logo, page_title ="Bonsoir !", layout = "wide")

@st.cache(allow_output_mutation=True)
def load_tokenizer():
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        return tokenizer

@st.cache(allow_output_mutation=True)
def load_model():
        model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)
        return  model

tokenizer =load_tokenizer()
model = load_model()

def reponse(question, temp=1, long=40):
    
    input_ids = tokenizer.encode(question, return_tensors='pt')
    output = model.generate(input_ids, max_length=long, temperature=temp, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    rep = tokenizer.decode(output[0], skip_special_tokens=True)
    return rep

def load_animation(url: str):
    r = requests.get(url)
    if r.status_code != 200 :
        return None
    return r.json()

url = "https://assets10.lottiefiles.com/packages/lf20_96bovdur.json"
robot = load_animation(url)


def contact_message():
    st.header(":mailbox: Let's Get In Touch !")

    name, message = st.columns((1,2))
    with name:
        contact_form = """<form action="https://formsubmit.co/maxime.letutour@gmail.com" method="POST">
     <input type="text" name="name" placeholder = "Ton Nom" required>
     <input type="email" name="email" placeholder = "Ton E-mail" required>
     </form>"""
        st.markdown(contact_form, unsafe_allow_html=True)

    with message :
        contact_form2 = """<form action="https://formsubmit.co/maxime.letutour@gmail.com" method="POST">
        <textarea name="message" placeholder="Ecris moi !"></textarea>
        <button type="submit">Send</button>
    """
        st.markdown(contact_form2, unsafe_allow_html=True)

    with open("style2.txt") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



def robot():
    robot = load_animation(url)
    col1, col2, col3 = st.columns((5,1,5))

    with col1:

        st.subheader("Choose the length of my answer")
        long = st.number_input("Be aware that long answers require more time to think !", min_value=10, max_value=250, step =10)

        st.subheader("Ask me something")
        question = st.text_input('Be aware that I speak only english for the moment !',max_chars = 60)
        question = str(question)

        ok = st.button('Ask me')
        

    with col3:
        st_lottie(robot, speed=1, loop=True, quality = "low",height =300, width = 300)
        if ok:
            rep = reponse(question, long = long)
            

            rep_style = f'<p style="font-family:Lucida Handwriting; color:#00008B; font-size: 20px;">{rep}</p>'
            st.markdown(rep_style, unsafe_allow_html=True)
            




def main():
    st.title("Shall we chat ? Ask me a question")

    with st.sidebar:

        choice = option_menu(
            menu_title = "Ask Me",
            options = ["Question", "Envoie Moi Un Message"],
            icons=["chat","envelope"],
            menu_icon="robot"
        )

    if choice == "Envoie Moi Un Message":
        contact_message()
    
    elif choice == "Question":
        robot()

    st.sidebar.subheader(":notebook_with_decorative_cover: Par Maxime Le Tutour")

    st.sidebar.write(" :blue_book: [**Mon LinkedIn**](https://www.linkedin.com/in/maxime-le-tutour-95994795/)", unsafe_allow_html =True)

    





if __name__ == '__main__':
	main()
import streamlit as st
from main import get_cover_letter


model_name = "anakin87/zephyr-7b-alpha-sharded"

st.title('Cover Letter Generator')

def generate_response(url, cv):
    try:
        output = get_cover_letter(url, cv)
        st.write(output)
    except Exception as e:
        st.error(f"An error occurred: {e}")

with st.form('my_form'):
    text = st.text_area('ENTER THE LINK OF URL YOU WANT FROM LINKEDIN:', 'Full URL with https:// ...')
    files = st.file_uploader("Upload files", type=["pdf"], accept_multiple_files=False)

    submitted = st.form_submit_button('Submit')

if submitted and text is not None and files is not None:
    generate_response(text, files)

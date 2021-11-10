import streamlit as st
from config import text_endpoint, top_k, images_path
from helper import search_by_text, UI

matches = []

# Layout
st.set_page_config(page_title="CrossModal search using sparse embeddings")
st.markdown(
    body=UI.css,
    unsafe_allow_html=True,
)
st.write(
    "<style>div.row-widget.stRadio > div{flex-direction:row; margin-left:auto; margin-right: auto; align: center}</style>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.markdown(UI.about_block, unsafe_allow_html=True)

query = st.text_input("", key="text_search_box")
search_fn = search_by_text
if st.button("Search", key="text_search"):
    matches = search_by_text(query, text_endpoint, top_k)

# Results area
cell1, cell2, cell3 = st.columns(3)
cell4, cell5, cell6 = st.columns(3)
cell7, cell8, cell9 = st.columns(3)
all_cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]

for cell, match in zip(all_cells, matches):
    cell.image(f'{images_path}/{match["tags"]["filename"]}')

import streamlit as st
from config import text_endpoint, top_k, images_path, split, split_root, image_size
from helper import search_by_text, UI


def search_and_show(text, gt_filename=None):
    def _search_and_show():
        matches = search_by_text(text, text_endpoint, top_k)
        if gt_filename is not None:
            st.title(f' Expected groundtruth for:\n {text}')
            cell_gt = st.container()
            cell_gt.image(f'{images_path}/{gt_filename}', caption=text)
        st.title(f' Matches obtained for: \n {text}')
        cell1, cell2, cell3 = st.columns(3)
        cell4, cell5, cell6 = st.columns(3)
        cell7, cell8, cell9 = st.columns(3)
        all_cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]
        for cell, match in zip(all_cells, matches):
            cell.image(f'{images_path}/{match["tags"]["filename"]}', use_column_width=True)

    return _search_and_show


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

st.header("What do you want to search with? Free Text or some text from set")
media_type = st.radio("", ["Text", "GroundTruth"])
gt_filename = None

if media_type == 'Text':
    query = st.text_input("", key="text_search_box")
    search_fn = search_by_text
    st.button("Search", key="text_search", on_click=search_and_show(query))

else:
    st.subheader("...or search from a sample")
    import random
    from src.dataset.dataset import CaptionFlickr30kDataset

    dataset = CaptionFlickr30kDataset(root=images_path, split_root=split_root, split=split)

    random_indices = [random.randint(0, len(dataset)) for _ in range(4)]
    random_data = [dataset[random_idx] for random_idx in random_indices]
    sample_texts = [
        d[1] for d in random_data
    ]
    sample_gts = [
        d[0] for d in random_data
    ]
    for i, text in enumerate(sample_texts):
        st.button(text, key=f'{i}', on_click=search_and_show(text, sample_gts[i]))

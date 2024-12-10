import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Personalized Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()


# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses ")
    st.table(results)
    return results


def train(model_name, params):
    if model_name == backend.models[0]:
        pass
    else: 
        # Start training model
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_name, params)
        st.success('Done!')
       

def predict(model_name, user_ids, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations '):
        time.sleep(0.5)
        res = backend.predict(model_name, user_ids, params)
    st.success('Recommendations generated!')

    # Only drop the 'SCORE' column for models other than the course similarity model
    if model_name != backend.models[0]:
        res = res.drop(columns=['SCORE'], errors='ignore')  # Avoid error if SCORE is not in the dataframe
    return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=1, max_value=100,
                                    value=10, step=1)
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
# KNN model
elif model_selection == backend.models[1]:
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=1, max_value=100,
                                    value=10, step=1)
    neighbors = st.sidebar.slider('Number of neighbors',
                                   min_value=1, max_value=80,
                                   value=40, step=1)
    params['top_courses'] = top_courses
    params['k'] = neighbors
# NMF model
elif model_selection == backend.models[2]:
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=1, max_value=100,
                                    value=10, step=1)
    n_factors = st.sidebar.slider('Number of factors',
                                   min_value=1, max_value=60,
                                   value=32, step=1)
    params['top_courses'] = top_courses
    params['n_factors'] = n_factors
# Neural Network model
elif model_selection == backend.models[3]:
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=1, max_value=100,
                                    value=10, step=1)
    embdedding_size = st.sidebar.slider('Embedding size',
                                   min_value=1, max_value=64,
                                   value=16, step=1)
    epochs = st.sidebar.slider('Number of epochs',
                                   min_value=1, max_value=50,
                                   value=10, step=1)
    params['top_courses'] = top_courses
    params['embdedding_size'] = embdedding_size
    params['epochs'] = epochs


# Training
st.sidebar.subheader('3. Training')
if model_selection != backend.models[0]:
    training_button = st.sidebar.button("Train Model")
else:
    training_button = None  
training_text = st.sidebar.text('')
if training_button:
    train(model_selection, params)


# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)
    
    # Check if 'SCORE' column exists before filtering
    if 'SCORE' in res_df.columns:
        res_df = res_df[['COURSE_ID', 'SCORE']]
    else:
        res_df = res_df[['COURSE_ID']]  # Only show COURSE_ID if SCORE doesn't exist

    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
    st.table(res_df)

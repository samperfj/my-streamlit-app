import pandas as pd
import numpy as np
from surprise import KNNBasic, NMF
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

models = ("Course Similarity",
          "KNN",
          "NMF",
          "Neural Network",
          )


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


# Get course similarity recommendations
def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


# Neural network architecture
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # Initialize embedding matrices and biases
        with tf.device('/CPU:0'):  # Force Embedding layer computation on CPU
            self.user_embedding_matrix = self.add_weight(
                shape=(num_users, embedding_size),
                initializer="he_normal",
                trainable=True,
                name="user_embedding_matrix",
            )
            self.user_bias = self.add_weight(
                shape=(num_users, 1),
                initializer="zeros",
                trainable=True,
                name="user_bias",
            )
            self.item_embedding_matrix = self.add_weight(
                shape=(num_items, embedding_size),
                initializer="he_normal",
                trainable=True,
                name="item_embedding_matrix",
            )
            self.item_bias = self.add_weight(
                shape=(num_items, 1),
                initializer="zeros",
                trainable=True,
                name="item_bias",
            )

    def call(self, inputs):
        user_ids = tf.cast(inputs[:, 0], tf.int32)
        item_ids = tf.cast(inputs[:, 1], tf.int32)

        # Replace invalid indices with a default valid index (e.g., 0)
        user_ids = tf.where(user_ids < 0, tf.zeros_like(user_ids), user_ids)
        item_ids = tf.where(item_ids < 0, tf.zeros_like(item_ids), item_ids)

        with tf.device('/CPU:0'):
            user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, user_ids)
            user_bias = tf.nn.embedding_lookup(self.user_bias, user_ids)
            item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, item_ids)
            item_bias = tf.nn.embedding_lookup(self.item_bias, item_ids)

        dot_user_item = tf.reduce_sum(user_embedding * item_embedding, axis=1, keepdims=True)
        x = dot_user_item + user_bias + item_bias
        return tf.nn.relu(x)

  
    
# Process dataset and map user and item ids to indices
def process_dataset(raw_data):
    encoded_data = raw_data.copy()
    
    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}
    
    # Mapping item ids to indices
    course_list = encoded_data["item"].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    # Convert user ids to indices
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert course ids to indices
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return encoded_data, user_idx2id_dict, course_idx2id_dict


# Split the encoded dataset to use the entire dataset for training
def generate_train__dataset(dataset, scale=True):

    min_rating = min(dataset["rating"])
    max_rating = max(dataset["rating"])

    # Shuffle the dataset
    dataset = dataset.sample(frac=1, random_state=42)

    # Prepare features and target variable
    x = dataset[["user", "item"]].values
    if scale:
        y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    else:
        y = dataset["rating"].values

    # Use 100% of the data for training
    x_train, y_train = x, y

    return x_train, y_train


# Model training
def train(model_name, params):
    """Train the selected model."""
    # Course Similarity model
    if model_name == models[0]:
        pass

    # KNN model
    elif model_name == models[1]:
        global knn_model
        # Read the ratings dataset with Surprise Reader
        reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))
        course_dataset = Dataset.load_from_file("ratings.csv", reader=reader)

        # Train the KNN model on the full dataset
        trainset = course_dataset.build_full_trainset()
        k = params.get('k', 40)  # Default to k=40 if not provided
        min_k = 1
        sim_options = {
            'name': 'cosine',  # Cosine similarity
            'user_based': False,  # Item-based collaborative filtering
        }
        model = KNNBasic(k=k, min_k=min_k, sim_options=sim_options)
        model.fit(trainset)

        # Save the trained model for later predictions
        knn_model = model

    
    # NMF model
    elif model_name == models[2]:
        global nmf_model
        # Read the ratings dataset with Surprise Reader
        reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))
        course_dataset = Dataset.load_from_file("ratings.csv", reader=reader)

        # Train the NMF model on the full dataset
        trainset = course_dataset.build_full_trainset()
        n_factors = params.get('n_factors', 32)  # Default to 32 if not provided
        init_low = 0.5
        init_high = 5.0
        model = NMF(init_low=init_low, init_high=init_high, n_factors=n_factors)
        model.fit(trainset)

        # Save the trained model for later predictions
        nmf_model = model


    # Neural Network model
    elif model_name == models[3]:  # Neural Network model
        global ann_model, user_idx2id_dict, course_idx2id_dict  # Declare global variables
        # Load and process ratings
        rating_df = load_ratings()
        num_users = len(rating_df['user'].unique())
        num_items = len(rating_df['item'].unique())

        # Process dataset and prepare training data
        encoded_data, user_idx2id_dict, course_idx2id_dict = process_dataset(rating_df)
        x_train, y_train = generate_train__dataset(encoded_data)

        # Model initialization
        embedding_size = params.get('embedding_size', 16)
        epochs = params.get('epochs', 10)
        model = RecommenderNet(num_users, num_items, embedding_size)
        model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                      optimizer=keras.optimizers.Adam())  

        # Train the model
        model.fit(x=x_train, y=y_train, batch_size=64, epochs=epochs, verbose=0)

        # Save the trained model globally
        ann_model = model    
        

# Model predicting
def predict(model_name, user_ids, params):
    """Generate predictions for the selected model."""
    sim_threshold = params.get("sim_threshold", 60) / 100.0  # Default 0.6
    top_courses = params.get("top_courses", 10)  # Default top 10 courses

    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    ratings_df = load_ratings()

    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            # Filter recommendations by similarity threshold
            filtered_res = {k: v for k, v in res.items() if v >= sim_threshold}
            # Sort and limit results to top_courses
            sorted_res = sorted(filtered_res.items(), key=lambda item: item[1], reverse=True)[:top_courses]
            for course_id, score in sorted_res:
                users.append(user_id)
                courses.append(course_id)
                scores.append(score)

        # KNN model
        elif model_name == models[1]:
            global knn_model
            if knn_model is not None:
                # Predict ratings for all items for the given user
                user_ratings = ratings_df[ratings_df['user'] == user_id]
                enrolled_course_ids = set(user_ratings['item'].to_list())
                all_courses = set(idx_id_dict.values())

                unselected_course_ids = all_courses.difference(enrolled_course_ids)
                preds = []
                for course_id in unselected_course_ids:
                    pred = knn_model.predict(uid=user_id, iid=course_id, verbose=False)
                    preds.append((pred.iid, pred.est))
                # Sort predictions and limit to top_courses
                sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_courses]
                for course_id, score in sorted_preds:
                    users.append(user_id)
                    courses.append(course_id)
                    scores.append(score)

        # NMF model
        elif model_name == models[2]:
            global nmf_model
            if nmf_model is not None:
                # Predict ratings for all items for the given user
                user_ratings = ratings_df[ratings_df['user'] == user_id]
                enrolled_course_ids = set(user_ratings['item'].to_list())
                all_courses = set(idx_id_dict.values())

                unselected_course_ids = all_courses.difference(enrolled_course_ids)
                preds = []
                for course_id in unselected_course_ids:
                    pred = nmf_model.predict(uid=user_id, iid=course_id, verbose=False)
                    preds.append((pred.iid, pred.est))
                # Sort predictions and limit to top_courses
                sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_courses]
                for course_id, score in sorted_preds:
                    users.append(user_id)
                    courses.append(course_id)
                    scores.append(score)

        # Neural Network model
        elif model_name == models[3]:  
            global ann_model, user_idx2id_dict, course_idx2id_dict
            if ann_model is not None:
                # Handle new users during prediction
                if user_id not in user_idx2id_dict:
                    # New user case: Assign new index
                    new_user_idx = len(user_idx2id_dict)  # Assign the next available index
                    user_idx2id_dict[user_id] = new_user_idx  # Add to dictionary
                    user_encoded = new_user_idx
                else:
                    user_encoded = user_idx2id_dict[user_id]  # Get existing index

                # Handle new courses during prediction
                all_courses = set(course_idx2id_dict.values())
                unselected_course_ids = list(all_courses.difference(set(ratings_df[ratings_df['user'] == user_id]['item'].tolist())))

                course_encoded = []
                for course_id in unselected_course_ids:
                    if course_id in course_idx2id_dict:
                        course_encoded.append(course_idx2id_dict[course_id])
                    else:
                        # New or unknown course: assign fallback index (-1)
                        course_encoded.append(-1)

                # Prepare user and course input arrays for predictions
                user_input = np.array([user_encoded] * len(course_encoded))  # Repeat for all courses
                course_input = np.array(course_encoded)  # All unselected course IDs

                # Stack inputs for prediction
                input_data = np.stack([user_input, course_input], axis=-1)

                # Make predictions
                predictions = ann_model.predict(input_data, verbose=0)
                preds = list(zip(unselected_course_ids, predictions.flatten()))

                # Sort and filter predictions based on score
                sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_courses]
                for course_id, score in sorted_preds:
                    users.append(user_id)
                    courses.append(course_id)
                    scores.append(score)


    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
